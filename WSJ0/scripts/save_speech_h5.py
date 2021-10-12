# -*- coding: utf-8 -*-

import sys
import os
import timeit
import h5py
import soundfile as sf
from scipy.signal import resample_poly
import numpy as np
import argparse
import pyloudnorm as pyln
from multiprocessing import Pool
from tqdm import tqdm
from constants import SRATE

start_init = timeit.default_timer()
########################### 1. Configurations
######### parse commands
parser = argparse.ArgumentParser("save speech to h5 file")
parser.add_argument("--split", type=str)
parser.add_argument("--nthreads", type=int, default=16)
args = parser.parse_args()

######### file paths
input_wav_dir = "/fs/scratch/PAS0774/ashutosh/premix_data/csr_1_wav/wsj0/"
output_h5_dir = "/fs/scratch/PAS0774/ashutosh/WSJ0_ATT_TRAIN/data/speech/{}kHz/".format(
    SRATE // 1000
)
meter = pyln.Meter(SRATE)


def save_to_h5(count, path):
    input_path = os.path.join(input_wav_dir, path)
    output_path = os.path.join(output_h5_dir, path.replace(".wav", ".samp"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = h5py.File(output_path, "w")
    s, srate_ = sf.read(input_path)
    assert srate_ == 16000
    if srate_ != SRATE:
        s = resample_poly(s, SRATE, srate_)
    # print("s", s.dtype, s.shape, s.min(), s.max())
    loudness = meter.integrated_loudness(s)
    writer.create_dataset("speech", data=s.astype(np.float32), chunks=True)
    writer.create_dataset("loudness", data=loudness)
    writer.close()


speech_list_file = "../filelists/{}_filelists.txt".format(args.split)
with open(speech_list_file, "r") as f:
    speech_list = [line.strip() for line in f.readlines()]
N = len(speech_list)
print("Number of wavfiles are {}".format(N))

sum_time = 0
pbar = tqdm(total=len(speech_list))


def update(*a):
    pbar.update()


pool = Pool(processes=args.nthreads)
try:
    for count, path in enumerate(speech_list):
        pool.apply_async(save_to_h5, args=(count, path), callback=update)
except Exception as e:
    print(str(e))
    pool.close()
pool.close()
pool.join()
