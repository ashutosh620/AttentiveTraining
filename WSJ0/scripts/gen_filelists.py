import os
import glob
import argparse
import random

random.seed(0)

parser = argparse.ArgumentParser("get test and validation metadata")
parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
args = parser.parse_args()

if args.split == "train":
    split_dir_list = ["si_tr_s"]
else:
    split_dir_list = ["si_et_05", "si_dt_05"]

raw_dir = "/fs/scratch/PAS0774/ashutosh/premix_data/csr_1_wav/wsj0/"
speech_files = []
for split_dir in split_dir_list:
    speech_files.extend(glob.glob(os.path.join(raw_dir, split_dir, "*/*.wav")))
speech_files.sort()
print("Number of speech files: {}".format(len(speech_files)))

os.makedirs("../filelists", exist_ok=True)
if args.split == "train":
    spk_id_to_path = {}
    for path in speech_files:
        spk_id = path.split("/")[-2]
        if spk_id in spk_id_to_path:
            spk_id_to_path[spk_id].append(path)
        else:
            spk_id_to_path[spk_id] = [path]
    N_spk = len(spk_id_to_path)
    spk_list = list(spk_id_to_path)
    random.shuffle(spk_list)
    train_spk_list = spk_list[:80]
    valid_spk_list = spk_list[80:]
    split_list = ["train", "valid"]
    spk_list = {}
    spk_list["train"] = train_spk_list
    spk_list["valid"] = valid_spk_list
    for split in split_list:
        with open("../filelists/{}_filelists.txt".format(split), "w") as f:
            curr_spk_list = spk_list[split]
            for spk in curr_spk_list:
                curr_path_list = spk_id_to_path[spk]
                for path in curr_path_list:
                    f.write(path.replace(raw_dir, "") + "\n")
else:
    with open("../filelists/{}_filelists.txt".format(args.split), "w") as f:
        for path in speech_files:
            f.write(path.replace(raw_dir, "") + "\n")
