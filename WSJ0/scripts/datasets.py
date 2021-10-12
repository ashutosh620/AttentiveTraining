# KIANA
# -*- coding: utf-8 -*-
# from create_eval_metadata import get_spk_dict, mix_data_with_overlap, normalizeLoudness, extend_noise
import h5py
import soundfile as sf
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import sys
import scipy
import scipy.signal
import librosa
import torchvision.transforms as transforms
import random
import pyloudnorm as pyln
import warnings
from constants import *
import pandas as pd
import timeit

MIN_LENGTH = 2
MAX_LENGTH = 3


def to_tensor(x):
    return torch.from_numpy(x).float()


def extend_noise(noise, max_length):
    """ Concatenate noise using hanning window"""
    noise_ex = noise
    window = np.hanning(SRATE + 1)
    # Increasing window
    i_w = window[: len(window) // 2 + 1]
    # Decreasing window
    d_w = window[len(window) // 2 :: -1]
    # Extend until max_length is reached
    while len(noise_ex) < max_length:
        noise_ex = np.concatenate(
            (
                noise_ex[: len(noise_ex) - len(d_w)],
                np.multiply(noise_ex[len(noise_ex) - len(d_w) :], d_w)
                + np.multiply(noise[: len(i_w)], i_w),
                noise[len(i_w) :],
            )
        )
    noise_ex = noise_ex[:max_length]
    return noise_ex


def normalizeLoudness(x, x_loudness, target_loudness):
    # normalize loudness
    delta_loudness = target_loudness - x_loudness
    gain = np.power(10.0, delta_loudness / 20.0)
    x = x * gain

    # avoid clipping
    if np.max(np.abs(x)) >= 1.0:
        scale = MAX_WAV_AMP / np.max(np.abs(x))
        gain *= scale
        x *= scale
    return x, gain


def get_spk_dict(speech_list):
    spk_id_to_path = {}
    path_to_spk_id = {}
    for speech_file in speech_list:
        lst = speech_file.split("/")
        spk_id = lst[-2]
        # print(spk_id)
        if spk_id in spk_id_to_path:
            spk_id_to_path[spk_id].add(speech_file)
        else:
            spk_id_to_path[spk_id] = set([speech_file])
        path_to_spk_id[speech_file] = spk_id
    return spk_id_to_path, path_to_spk_id


def pad_and_mix(
    speech_list,
    speech_length_list,
    speech_gain_list,
    pad_left_list,
    max_size,
    spk_pattern,
    return_targets=False,
):
    pad_right_list = []
    mixture = 0
    final_speech_dict = {}
    for i, pad_left in enumerate(pad_left_list):
        pad_right = max_size - pad_left - speech_length_list[i]
        speech = speech_list[i]
        final_speech = np.pad(speech, (pad_left, pad_right), mode="constant")
        key = spk_pattern[i]
        if key in final_speech_dict:
            final_speech_dict[key] += final_speech
        else:
            final_speech_dict[key] = final_speech

        pad_right_list.append(pad_right)
        mixture += final_speech

    # avoid clipping
    scale = 1.0
    max_amp = np.max(np.abs(mixture))
    for key, value in final_speech_dict.items():
        max_amp = max(max_amp, np.max(np.abs(value)))

    if max_amp >= 1.0:
        scale = MAX_WAV_AMP / max_amp
    mixture *= scale

    updated_gain_list = []
    for i, gain in enumerate(speech_gain_list):
        updated_gain_list.append(gain * scale)

    if return_targets:
        spk_keys = list(set(spk_pattern))
        spk_keys.sort()
        target_list = []
        for spk_key in spk_keys:
            target_list.append(final_speech_dict[spk_key] * scale)
        return pad_left_list, pad_right_list, updated_gain_list, mixture, target_list

    return pad_left_list, pad_right_list, updated_gain_list, mixture


def mix_data_with_no_overlap(
    speech_list, speech_length_list, speech_gain_list, spk_pattern, return_targets=False
):

    pad_left_list = []
    max_size = 0
    for i, _ in enumerate(spk_pattern):
        speech_length = speech_length_list[i]
        # print(speech_length, speech_list[i].size)
        if i == 0:
            pad_left = 0
        else:
            gap = int(random.uniform(MIN_GAP, MAX_GAP) * SRATE)
            pad_left = prev_end + gap
        pad_left_list.append(pad_left)
        prev_end = pad_left + speech_length
        max_size = max(prev_end, max_size)

    return pad_and_mix(
        speech_list,
        speech_length_list,
        speech_gain_list,
        pad_left_list,
        max_size,
        spk_pattern,
        return_targets,
    )


def mix_data_with_overlap(
    speech_list,
    speech_length_list,
    speech_gain_list,
    spk_pattern,
    overlap_type,
    return_targets=False,
    min_offset=16000,
):

    end_list = []
    key_to_prev_end = {}
    max_size = 0
    pad_left_list = []
    for i, key in enumerate(spk_pattern):
        if i == 0:
            pad_left = 0
        else:
            if key in key_to_prev_end and end_list[-1] == key_to_prev_end[key]:
                pad_left = key_to_prev_end[key] + int(
                    random.uniform(MIN_GAP, MAX_GAP) * SRATE
                )
            else:
                if overlap_type == "random":
                    if random.uniform(0, 1) < OVERLAP_PROB:
                        if i == 1:
                            pad_left = random.randint(
                                min_offset, max(end_list[-1], min_offset)
                            )
                        else:
                            gap = int(MIN_GAP * SRATE)
                            left = end_list[-2] + gap
                            right = max(left, end_list[-1])
                            pad_left = random.randint(left, right)
                    else:
                        pad_left = end_list[-1] + int(
                            random.uniform(MIN_GAP, MAX_GAP) * SRATE
                        )
                elif overlap_type == "max":
                    if i == 1:
                        pad_left = min_offset
                    else:
                        pad_left = end_list[-2] + int(MIN_GAP * SRATE)
                elif overlap_type == "half":
                    if i == 1:
                        pad_left = int((min_offset + end_list[-1]) // 2)

                    else:
                        gap = int(random.uniform(MIN_GAP, MAX_GAP) * SRATE)
                        left = end_list[-2] + gap
                        right = max(left, end_list[-1])
                        pad_left = int((left + right) // 2)
        curr_end = pad_left + speech_length_list[i]
        end_list.append(curr_end)
        end_list.sort()
        end_list = end_list[-2:]
        key_to_prev_end[key] = curr_end
        max_size = max(curr_end, max_size)
        pad_left_list.append(pad_left)
    return pad_and_mix(
        speech_list,
        speech_length_list,
        speech_gain_list,
        pad_left_list,
        max_size,
        spk_pattern,
        return_targets,
    )


def sample_spk_pattern(num_spk, num_iter=4):
    tmp_set = set([1])
    spk_pattern = []
    seen_set = set([])
    for i in range(num_iter):
        curr = random.choice(list(tmp_set))
        seen_set.add(curr)

        spk_pattern.append(str(curr))
        if len(tmp_set) < num_spk:
            tmp_set.add(len(seen_set) + 1)
    return "".join(spk_pattern)


class TrainingDataset(Dataset):
    r"""Training dataset."""

    def __init__(
        self,
        speech_list,
        speech_dir,
        noise_list=None,
        noise_dir=None,
        num_spk=2,
        num_out_stream=1,
        num_chunks=4,
        nsamples=80000,
        length=100000,
        overlap_type="random",
        min_offset=1.0,
    ):
        self.num_out_stream = num_out_stream
        with open(speech_list, "r") as f:
            speech_list = [
                line.strip().replace(".wav", ".samp") for line in f.readlines()
            ]
        self.speech_list = speech_list
        self.speech_dir = speech_dir
        self.noise_dir = noise_dir
        if self.noise_dir is not None:
            assert noise_list is not None
            self.noise_list = noise_list
            with open(self.noise_list, "r") as f:
                noise_list = [line.strip() for line in f.readlines()]
            self.noise_list = noise_list

        self.spk_id_to_path, self.path_to_spk_id = get_spk_dict(self.speech_list)
        self.spk_set = set(self.spk_id_to_path.keys())
        print("number of training speakers", len(self.spk_set))
        self.num_spk = num_spk
        self.num_out_stream = num_out_stream
        self.num_chunks = num_chunks
        self.nsamples = nsamples
        self.length = length
        self.overlap_type = overlap_type
        self.min_offset = int(min_offset * SRATE)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        feature, target, num_spk = self.generate_data(index)

        feature *= FEAT_SCALE
        target *= FEAT_SCALE

        feature = to_tensor(feature + 1e-6)
        target = to_tensor(target + 1e-6)

        return feature, target, torch.tensor(num_spk).long()

    def generate_data(self, index):
        spk_pattern = sample_spk_pattern(self.num_spk, self.num_chunks)

        spk_keys = set(spk_pattern)
        num_spk = len(spk_keys)
        spk_ids = random.sample(self.spk_set, num_spk)
        key_to_id = {}
        key_to_files = {}
        for key, spk_id in zip(spk_keys, spk_ids):
            key_to_id[key] = spk_id
            key_to_files[key] = random.sample(
                self.spk_id_to_path[spk_id], spk_pattern.count(key)
            )

        speech_list = []
        speech_length_list = []
        speech_gain_list = []
        for key in spk_pattern:
            curr_file = key_to_files[key].pop()
            reader = h5py.File(os.path.join(self.speech_dir, curr_file), "r")
            speech = reader["speech"][:]
            speech_loudness = reader["loudness"][()]
            reader.close()

            start, end = librosa.effects.trim(
                speech,
                top_db=TOP_DB,
                frame_length=int(FRAME_SIZE * SRATE / 1000),
                hop_length=int(FRAME_SHIFT * SRATE / 1000),
            )[1]
            speech_length = int(random.uniform(MIN_LENGTH, MAX_LENGTH) * SRATE)
            end = min(end, start + speech_length)
            speech_length = end - start
            speech = speech[start:end]
            target_speech_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
            speech, speech_gain = normalizeLoudness(
                speech, speech_loudness, target_speech_loudness
            )
            speech_list.append(speech)
            speech_length_list.append(speech_length)
            speech_gain_list.append(speech_gain)

        _, _, _, mix_speech, target_list = mix_data_with_overlap(
            speech_list,
            speech_length_list,
            speech_gain_list,
            spk_pattern,
            self.overlap_type,
            return_targets=True,
            min_offset=self.min_offset,
        )

        mix_speech = mix_speech[: self.nsamples]
        for i in range(len(target_list)):
            target_list[i] = target_list[i][: self.nsamples]

        if self.noise_dir is not None:
            mix_speech = self.add_noise(mix_speech)
            scale = 1.0
            if np.max(np.abs(mix_speech)) >= 1.0:
                scale = MAX_WAV_AMP / np.max(np.abs(mix_speech))
            mix_speech *= scale
            for i in range(len(target_list)):
                target_list[i] *= scale

        num_spk = len(target_list)
        feature = np.reshape(mix_speech, [1, -1])
        if self.num_out_stream == 1:
            target = np.reshape(target_list[0], [1, -1])
        elif self.num_out_stream == 2:
            target = np.reshape(target_list[0], [1, -1])
            if num_spk > 1:
                background = np.stack(target_list[1:], axis=0)
                background = np.sum(background, axis=0, keepdims=True)
            else:
                background = np.zeros_like(target)
            target = np.concatenate([target, background], axis=0)
        else:
            assert num_spk <= self.num_out_stream
            target = np.stack(target_list, axis=0)
            pad = self.num_out_stream - target.shape[0]
            if pad > 0:
                target = np.pad(target, ((0, pad), (0, 0)), mode="constant")
        return feature, target, num_spk

    def add_noise(self, mix_speech):
        mix_speech_size = mix_speech.size
        noise_ind = random.randint(0, len(self.noise_list) - 1)
        noise_file = self.noise_list[noise_ind]
        reader = h5py.File(os.path.join(self.noise_dir, noise_file), "r")
        noise = reader["noise"][:]
        noise_loudness = reader["loudness"][()]
        reader.close()
        assert np.sum(noise ** 2) > 0.0

        if noise.size < mix_speech_size:
            noise = extend_noise(noise, mix_speech_size)
        else:
            noise_start = random.randint(0, noise.size - mix_speech_size)
            noise = noise[noise_start : noise_start + mix_speech_size]

        target_noise_loudness = random.uniform(NOISE_MIN_LOUDNESS, NOISE_MAX_LOUDNESS)

        noise, _ = normalizeLoudness(noise, noise_loudness, target_noise_loudness)

        mix_speech += noise

        return mix_speech


class EvalDataset(Dataset):
    r"""Evaluation dataset."""

    def __init__(
        self,
        metadata_file,
        speech_dir,
        noise_dir=None,
        spk_pattern_list=None,
        overlap_type_list=None,
    ):
        self.df = pd.read_csv(metadata_file)
        # print(type(self.df.iloc[0]["spk_pattern"]))
        self.speech_dir = speech_dir
        self.noise_dir = noise_dir
        self.spk_pattern_list = spk_pattern_list
        self.overlap_type_list = overlap_type_list
        if self.spk_pattern_list is not None:
            assert type(self.spk_pattern_list) == list
            self.spk_pattern_list = [int(_) for _ in self.spk_pattern_list]
            print("speaker_pattern_list", self.spk_pattern_list)
            self.df = self.df[self.df["spk_pattern"].isin(self.spk_pattern_list)]

        if self.overlap_type_list is not None:
            assert type(self.overlap_type_list) == list
            self.df = self.df[self.df["overlap_type"].isin(self.overlap_type_list)]

        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        curr_row = self.df.iloc[index]
        spk_pattern = str(curr_row["spk_pattern"])
        speech_path_list = curr_row["speech_path_list"].split(" ")
        speech_start_list = [int(_) for _ in curr_row["speech_start_list"].split(" ")]
        speech_length_list = [int(_) for _ in curr_row["speech_length_list"].split(" ")]
        pad_left_list = [int(_) for _ in curr_row["pad_left_list"].split(" ")]
        pad_right_list = [int(_) for _ in curr_row["pad_right_list"].split(" ")]
        speech_gain_list = [float(_) for _ in curr_row["speech_gain_list"].split(" ")]

        target_dict = {}
        mix_speech = 0
        for i, spk_id in enumerate(spk_pattern):
            reader = h5py.File(os.path.join(self.speech_dir, speech_path_list[i]), "r")
            speech = reader["speech"][:]
            reader.close()

            speech = speech[
                speech_start_list[i] : speech_start_list[i] + speech_length_list[i]
            ]
            speech = np.pad(
                speech, (pad_left_list[i], pad_right_list[i]), mode="constant"
            )
            speech *= speech_gain_list[i]

            mix_speech += speech
            if spk_id in target_dict:
                target_dict[spk_id] += speech
            else:
                target_dict[spk_id] = speech

        keys = list(target_dict.keys())
        num_spk = len(keys)
        keys.sort()
        target_list = []
        for i, key in enumerate(keys):
            target_list.append(target_dict[key])

        if self.noise_dir is not None:
            noise_path = curr_row["noise_path"]
            reader = h5py.File(os.path.join(self.noise_dir, noise_path), "r")
            noise = reader["noise"][:]
            reader.close()

            noise_start = curr_row["noise_start"]
            noise_gain = curr_row["noise_gain"]
            ext_noise = bool(curr_row["extend_noise"])
            if ext_noise:
                noise = extend_noise(noise, mix_speech.size)
            else:
                noise = noise[noise_start : noise_start + mix_speech.size]

            noise *= noise_gain
            mix_speech += noise

        feature = np.reshape(mix_speech, [1, -1])
        target = np.reshape(target_list[0], [1, -1])

        # amplify as during training
        feature = feature * FEAT_SCALE
        target = target * FEAT_SCALE

        feature = to_tensor(feature + 1e-6)
        label = to_tensor(target + 1e-6)

        d = dict(curr_row)

        return feature, label, torch.tensor(num_spk).long(), d


class TrainCollate(object):
    def __init__(self):
        self.name = "collate"

    def __call__(self, batch):
        if isinstance(batch, list):
            feat_nchannels = batch[0][0].shape[0]
            label_nchannels = batch[0][1].shape[0]
            sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
            lengths = list(map(lambda x: (x[0].shape[1], x[1].shape[1]), sorted_batch))

            padded_feature_batch = torch.zeros(
                (len(lengths), feat_nchannels, lengths[0][0])
            )
            padded_label_batch = torch.zeros(
                (len(lengths), label_nchannels, lengths[0][1])
            )
            lengths1 = torch.zeros((len(lengths),), dtype=torch.int32)
            num_spks = torch.zeros((len(lengths),), dtype=torch.int32)
            for i in range(len(lengths)):
                padded_feature_batch[i, :, 0 : lengths[i][0]] = sorted_batch[i][0]
                padded_label_batch[i, :, 0 : lengths[i][1]] = sorted_batch[i][1]
                lengths1[i] = lengths[i][1]
                num_spks[i] = sorted_batch[i][2]

            return padded_feature_batch, padded_label_batch, num_spks, lengths1
        else:
            raise TypeError("`batch` should be a list.")


class EvalCollate(object):
    def __init__(self):
        self.name = "TestCollate"

    def __call__(self, batch):
        if isinstance(batch, list):

            feat_nchannels = batch[0][0].shape[0]
            label_nchannels = batch[0][1].shape[0]
            sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
            lengths = list(map(lambda x: (x[0].shape[1], x[1].shape[1]), sorted_batch))

            padded_feature_batch = torch.zeros(
                (len(lengths), feat_nchannels, lengths[0][0])
            )
            padded_label_batch = torch.zeros(
                (len(lengths), label_nchannels, lengths[0][1])
            )
            lengths1 = torch.zeros((len(lengths),), dtype=torch.int32)
            num_spks = torch.zeros((len(lengths),), dtype=torch.int32)
            dicts = []
            for i in range(len(lengths)):
                padded_feature_batch[i, :, 0 : lengths[i][0]] = sorted_batch[i][0]
                padded_label_batch[i, :, 0 : lengths[i][1]] = sorted_batch[i][1]
                lengths1[i] = lengths[i][1]
                num_spks[i] = sorted_batch[i][2]
                dicts.append(sorted_batch[i][3])

            return padded_feature_batch, padded_label_batch, num_spks, lengths1, dicts
        else:
            raise TypeError("`batch` should be a list.")
