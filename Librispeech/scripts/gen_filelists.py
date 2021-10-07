import os
import glob

raw_dir = "/fs/scratch/PAS0774/ashutosh/premix_data/LibriSpeech/"
split_list = ["train-clean-100", "test-clean", "dev-clean"]
os.makedirs("../filelists", exist_ok=True)
for split in split_list:
    speech_files = glob.glob(os.path.join(raw_dir, split, "*/*/*.flac"))
    print("Split: {}, Number of files: {}".format(split, len(speech_files)))
    with open("../filelists/{}_filelists.txt".format(split), "w") as f:
        for path in speech_files:
            f.write(path.replace(raw_dir, "") + "\n")
