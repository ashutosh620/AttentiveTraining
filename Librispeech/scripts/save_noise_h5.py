# -*- coding: utf-8 -*-

import sys
import os
import timeit
import h5py
import soundfile as sf
import librosa
import numpy as np
import argparse
import pyloudnorm as pyln
import pandas as pd
from scipy.signal import resample_poly
from multiprocessing import Pool
from tqdm import tqdm

MIN_LENGTH = 3.0 # remove all the noises with length less than MIN_LENGTH seconds
CHUNK_SIZE = 10. # chunk noises to a size of 10 seconds
########################### 1. Configurations
######### parse commands
parser = argparse.ArgumentParser("save noise to h5 file")
parser.add_argument("--split", type=str, default="train", choices=["train", "test", "valid"])
parser.add_argument("--srate", type=int, default=16000)
parser.add_argument("--nthreads", type=int, default=8)
args = parser.parse_args()
if args.split in ["test", "valid"]:
    CHUNK_SIZE = 15.
else:
    CHUNK_SIZE = 10.
N_CHUNK = int(args.srate*CHUNK_SIZE)
FOLDER_CAP = 5000
######### file paths
input_dir = "/fs/scratch/PAS0774/ashutosh/premix_data/high_res_wham/"
input_wav_dir = os.path.join(input_dir, "audio")
metadata_csv = os.path.join(input_dir, "high_res_metadata.csv")
output_h5_dir = "/fs/scratch/PAS0774/ashutosh/ATT_TRAIN/data/noise/{}kHz/".format(args.srate//1000)
filelists_path = "../filelists"

df = pd.read_csv(metadata_csv)

if args.split == "train":
    df = df[df["WHAM! Split"] == "Train"]
elif args.split == "valid":
    df = df[df["WHAM! Split"] == "Valid"]
elif args.split == "test":
    df = df[df["WHAM! Split"] == "Test"]

df = df[df["File Length (sec)"] >= MIN_LENGTH]
df = df["Filename"]


N = len(df)
print("Number of wavfiles is {}".format(N))
meter = pyln.Meter(args.srate)

def save_to_h5(count, n):
    filefolder = "{}-{}".format((count//FOLDER_CAP)*FOLDER_CAP,(count//FOLDER_CAP+1)*FOLDER_CAP-1)
    filename = "{}_noise_{}.samp".format(args.split, count)
    output_dir = os.path.join(output_h5_dir, args.split, filefolder)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    loudness = meter.integrated_loudness(n)
    writer = h5py.File(output_path, "w")
    writer.create_dataset("noise", data=n.astype(np.float32), chunks=True)
    writer.create_dataset("loudness", data=loudness)
    writer.close()
    return os.path.join(args.split, filefolder, filename)
    
# pbar = tqdm(total=len(speech_list))
pbar = tqdm()
noise_list = []
def update(*a):
    # print(len(a))
    noise_list.append(a[0])
    pbar.update()
pool = Pool(processes=args.nthreads)
try:
    count = 0
    
    for i in range(N):
        # write all examples into h5py files
        start = timeit.default_timer()
        input_path = os.path.join(input_wav_dir, df.iloc[i])
        n, sr = sf.read(input_path)
        if len(n.shape) == 2:
            n = n[:, 0]
        if sr != args.srate:
            n = resample_poly(n, args.srate, sr)
        n_list = np.array_split(n, list(range(N_CHUNK, n.size, N_CHUNK)))
        for j, n in enumerate(n_list):
            if n.size >= int(MIN_LENGTH*args.srate):
                pool.apply_async(save_to_h5, args=(count, n), callback=update)
                count += 1

except Exception as e:
    print(str(e))
    pool.close()
pool.close()
pool.join()


os.makedirs(filelists_path, exist_ok=True)
filename = os.path.join(filelists_path, "{}_noise_list.txt".format(args.split))
print("writing noise list")
with open(filename, "w") as f:
    for line in noise_list:
        f.write(line+"\n")

