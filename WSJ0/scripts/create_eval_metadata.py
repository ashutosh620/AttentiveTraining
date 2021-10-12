import argparse
import os
import random
from typing import Optional, final
import numpy as np
import soundfile as sf
import pandas as pd
import h5py
import sys
import librosa
from constants import *
from datasets import (
    extend_noise,
    normalizeLoudness,
    mix_data_with_no_overlap,
    mix_data_with_overlap,
    get_spk_dict,
)
from multiprocessing import Pool
from tqdm import tqdm

MIN_LENGTH = 2
MAX_LENGTH = 4
parser = argparse.ArgumentParser("get test and validation metadata")
parser.add_argument("--split", type=str, default="valid", choices=["test", "valid"])
parser.add_argument("--min_offset", type=float, default=1)
parser.add_argument("--num_samples", type=int, default=3000)
parser.add_argument("--nthreads", type=int, default=30)
args = parser.parse_args()
random.seed(args.split)
print("offset is ", args.min_offset)


speech_dir = "/fs/scratch/PAS0774/ashutosh/WSJ0_ATT_TRAIN/data/speech/{}kHz".format(
    int(SRATE // 1000)
)
noise_dir = "/fs/scratch/PAS0774/ashutosh/WSJ0_ATT_TRAIN/data/noise/{}kHz".format(
    int(SRATE // 1000)
)
if args.split == "test":
    speech_list = "../filelists/test_filelists.txt"
    noise_list = "../filelists/test_noise_list.txt"
elif args.split == "valid":
    speech_list = "../filelists/valid_filelists.txt"
    noise_list = "../filelists/valid_noise_list.txt"

with open(speech_list, "r") as f:
    speech_list = [line.strip().replace(".wav", ".samp") for line in f]
with open(noise_list, "r") as f:
    noise_list = [line.strip() for line in f]
print("Number of speech files: {}".format(len(speech_list)))
print("Number of noise files: {}".format(len(noise_list)))

random.shuffle(noise_list)
random.shuffle(speech_list)
min_offset = int(args.min_offset * SRATE)

spk_id_to_path, path_to_spk_id = get_spk_dict(speech_list)
spk_set = set(spk_id_to_path.keys())
print("number of speakers", len(spk_set))
spk_pattern_list = [
    "1111",
    "1212",
    "1221",
    "122221",
    "1231",
    "123231",
    "12341",
    "123451",
]
overlap_type_list = ["none", "half", "max"]


def generate_data(data_ind):

    cond = True
    while cond:
        noise_ind = random.randint(0, len(noise_list) - 1)
        noise_file = noise_list[noise_ind]
        reader = h5py.File(os.path.join(noise_dir, noise_file), "r")
        noise = reader["noise"][:]
        noise_loudness = reader["loudness"][()]
        reader.close()
        if np.sum(noise ** 2) > 0.0:
            cond = False
    dict_list = []
    for spk_pattern in spk_pattern_list:
        spk_keys = set(spk_pattern)
        # spk_loudness = {}
        # for key in spk_keys:
        #     spk_loudness[key] = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)

        num_spk = len(spk_keys)
        spk_ids = random.sample(spk_set, num_spk)
        key_to_id = {}
        key_to_files = {}
        for key, spk_id in zip(spk_keys, spk_ids):
            key_to_id[key] = spk_id
            # curr_set = set(random.sample(spk_id_to_path[spk_id], spk_pattern.count(key)))
            key_to_files[key] = random.sample(
                spk_id_to_path[spk_id], spk_pattern.count(key)
            )
            # print(type(spk_id_to_path[spk_id]))

        speech_list = []
        speech_length_list = []
        speech_gain_list = []
        speech_loudness_list = []
        speech_start_list = []
        speech_path_list = []
        speech_max_list = []
        for key in spk_pattern:
            curr_file = key_to_files[key].pop()
            reader = h5py.File(os.path.join(speech_dir, curr_file), "r")
            speech = reader["speech"][:]
            speech_loudness = reader["loudness"][()]
            reader.close()
            # print(curr_file, "speech", speech.shape,
            #       "loudness", speech_loudness)
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
            speech_max_list.append(np.max(np.abs(speech)))
            speech_length_list.append(speech_length)
            speech_loudness_list.append(speech_loudness)
            speech_start_list.append(start)
            speech_path_list.append(curr_file)
            target_speech_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
            speech, speech_gain = normalizeLoudness(
                speech, speech_loudness, target_speech_loudness
            )
            speech_list.append(speech)
            speech_gain_list.append(speech_gain)

        for overlap_type in overlap_type_list:
            if overlap_type == "none":
                (
                    pad_left_list,
                    pad_right_list,
                    gain_list,
                    mix_speech,
                ) = mix_data_with_no_overlap(
                    speech_list, speech_length_list, speech_gain_list, spk_pattern
                )
            else:
                (
                    pad_left_list,
                    pad_right_list,
                    gain_list,
                    mix_speech,
                ) = mix_data_with_overlap(
                    speech_list,
                    speech_length_list,
                    speech_gain_list,
                    spk_pattern,
                    overlap_type,
                    min_offset=min_offset,
                )

            # print(gain_list, curr_noise_gain)
            mix_speech_size = mix_speech.size
            ext_noise = False
            if noise.size < mix_speech_size:
                noise_start = 0
                curr_noise = extend_noise(noise, mix_speech_size)
                ext_noise = True
            else:
                noise_start = random.randint(0, noise.size - mix_speech_size)
                curr_noise = noise[noise_start : noise_start + mix_speech_size]

            noise_max = np.max(np.abs(curr_noise))
            target_noise_loudness = random.uniform(
                NOISE_MIN_LOUDNESS, NOISE_MAX_LOUDNESS
            )

            curr_noise, curr_noise_gain = normalizeLoudness(
                curr_noise, noise_loudness, target_noise_loudness
            )

            # target_noise_loudness1 = noise_loudness + 20 * np.log10(curr_noise_gain)

            # print(tmp, target_noise_loudness, target_noise_loudness1, scale)
            # assert target_noise_loudness >= target_noise_loudness1
            # sf.write("tmp_mix1_{}.wav".format(data_ind), mix_speech, samplerate=SRATE)
            # mix_speech_max = np.max(np.abs(mix_speech))
            mix_speech += curr_noise

            scale = 1.0
            if np.max(np.abs(mix_speech)) >= 1.0:
                scale = MAX_WAV_AMP / np.max(np.abs(mix_speech))
                # print("scale", scale, noise_loudness, target_noise_loudness)
            mix_speech *= scale
            final_noise_gain = curr_noise_gain * scale
            final_noise_loudness = noise_loudness + 20 * np.log10(final_noise_gain)
            final_gain_list = []
            for gain in gain_list:
                final_gain_list.append(gain * scale)
            final_loudness_list = []
            for loudness, gain in zip(speech_loudness_list, final_gain_list):
                final_loudness_list.append(loudness + 20 * np.log10(gain))
            # print("{}: ({:.2f}, {:.2f}), ({:.3f}, {:.3f})".format(data_ind, final_loudness_list[0], final_noise_loudness, np.max(np.abs(mix_speech)), np.max(np.abs(curr_noise))))
            # sf.write("tmp_mix_{}.wav".format(data_ind), mix_speech, samplerate=SRATE)
            # sf.write("tmp_noise_{}.wav".format(data_ind), curr_noise, samplerate=SRATE)
            # sys.exit()
            # assert final_noise_loudness <= final_loudness_list[0]
            #  assert (final_loudness_list[0] >= MIN_LOUDNESS - 4) and (final_loudness_list[0] <= MAX_LOUDNESS)
            final_noise_max = noise_max * final_noise_gain
            final_max_list = []
            for max_, gain in zip(speech_max_list, final_gain_list):
                final_max_list.append(max_ * gain)
            mix_speech_max = np.max(np.abs(mix_speech))
            curr_dict = {}
            curr_dict["data_ind"] = data_ind
            curr_dict["spk_pattern"] = spk_pattern
            curr_dict["overlap_type"] = overlap_type
            curr_dict["speech_start_list"] = " ".join(
                [str(i) for i in speech_start_list]
            )
            curr_dict["speech_length_list"] = " ".join(
                [str(i) for i in speech_length_list]
            )
            curr_dict["speech_path_list"] = " ".join([str(i) for i in speech_path_list])
            curr_dict["speech_gain_list"] = " ".join(
                ["{:.4f}".format(i) for i in final_gain_list]
            )
            curr_dict["speech_loudness_list"] = " ".join(
                ["{:.4f}".format(i) for i in final_loudness_list]
            )
            curr_dict["speech_max_list"] = " ".join(
                ["{:.4f}".format(i) for i in final_max_list]
            )
            curr_dict["pad_left_list"] = " ".join([str(i) for i in pad_left_list])
            curr_dict["pad_right_list"] = " ".join([str(i) for i in pad_right_list])
            curr_dict["noise_path"] = noise_file
            curr_dict["noise_start"] = noise_start
            curr_dict["extend_noise"] = ext_noise
            curr_dict["noise_gain"] = final_noise_gain
            curr_dict["noise_loudness"] = final_noise_loudness
            curr_dict["noise_max"] = final_noise_max
            curr_dict["mix_speech_length"] = mix_speech_size
            curr_dict["mix_speech_max"] = mix_speech_max
            dict_list.append(curr_dict)
    return dict_list


pbar = tqdm()
dict_list = []


def update(*a):
    # print(a[0])
    dict_list.extend(a[0])
    pbar.update()


pool = Pool(processes=args.nthreads)
try:
    for data_ind in range(args.num_samples):
        pool.apply_async(generate_data, args=(data_ind,), callback=update)
        # dict_list.extend(generate_data(data_ind))
except Exception as e:
    print(str(e))
    pool.close()
pool.close()
pool.join()


df = pd.DataFrame.from_dict(dict_list)
print(df)
# sys.exit()
df.to_csv(
    "../filelists/{}_metadata_{:.1f}.csv".format(args.split, args.min_offset),
    index=True,
    index_label="index",
)
print(df["noise_loudness"].min(), df["noise_loudness"].max())
df["first_speech_chunk_loudness"] = df["speech_loudness_list"].apply(
    lambda x: float(x.split(" ")[0])
)
df["first_speech_max"] = df["speech_max_list"].apply(lambda x: float(x.split(" ")[0]))
print(df["first_speech_chunk_loudness"].min(), df["first_speech_chunk_loudness"].max())
bins = np.arange(-49, -24, 2)
labels = np.arange(-49, -25, 2)
print("#" * 10)
binned_noise_loudness = pd.cut(df["noise_loudness"], bins)
print(binned_noise_loudness.value_counts(sort=False))

binned_speech_loudness = pd.cut(df["first_speech_chunk_loudness"], bins)
print(binned_speech_loudness.value_counts(sort=False))

bins = np.arange(0, 1.1, 0.1)
labels = np.arange(0.05, 1.0, 0.1)
binned_speech_max = pd.cut(df["first_speech_max"], bins)
print(binned_speech_max.value_counts(sort=False))

binned_noise_max = pd.cut(df["noise_max"], bins)
print(binned_noise_max.value_counts(sort=False))

binned_mix_max = pd.cut(df["mix_speech_max"], bins)
print(binned_mix_max.value_counts(sort=False))


# sys.exit()

