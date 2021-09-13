import os
import argparse
import glob
from tqdm import tqdm
import torchaudio
from math import ceil
import shutil
import torch
from random import sample, shuffle

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir", type=str, default=".", help="Path where librispeech dataset is present")

parser.add_argument("--chunk_size", type=int, default=16000, help="Required for the creating chunks of audio")

parser.add_argument("--chunk_thresh", type=int, default=1000, help="Noise threshold for a chunk to be selected")

args = parser.parse_args()

dataset_dir = os.path.join(args.dataset_dir, "LibriSpeech/train-clean-100")
chunk_size = args.chunk_size
chunk_thresh = args.chunk_thresh

training_set = []
validation_set = []
testing_set = []

for cls in tqdm(os.listdir(dataset_dir)):
    for subf in os.listdir(dataset_dir + str(cls)):
        try:
            for fil in os.listdir("{}/{}/{}".format(dataset_dir, cls, subf)):
                waveform, sample_rate = torchaudio.load(dataset_dir + str(cls) + "/{}/{}".format(subf, fil))
                if waveform.size()[1] <= chunk_size:
                    continue
                for i, chunk in enumerate(torch.split(waveform, chunk_size, dim = 1)):
                    if chunk.size()[1] < chunk_thresh:
                        continue
                    name = dataset_dir + "/" + str(cls) + "/{}".format(fil[:-5]) + "_{}.flac".format(i)
                    torchaudio.save(name, chunk, sample_rate)
                os.remove(dataset_dir + "/" + str(cls) + "/{}/{}".format(subf, fil))
        except:
            continue
    for subf in os.listdir(dataset_dir + str(cls)):
        try:
            os.rmdir("{}{}/{}".format(dataset_dir, cls, subf))
        except:
            continue

for cls in tqdm(os.listdir(dataset_dir)):
    fils = []
    for u in os.listdir("{}{}".format(dataset_dir, cls)):
        fils.append("{}/{}\n".format(cls, u))
    u = len(fils)
    testing_set.extend(sample(fils, int(u/100)))
    fils = list(set(fils).difference(set(testing_set)))
    validation_set.extend(sample(fils, int(u/100)))
    fils = list(set(fils).difference(set(validation_set)))
    training_set.extend(fils)

shuffle(training_set)
shuffle(validation_set)
shuffle(testing_set)

f = open("{}/training_list.txt".format(dataset_dir), "w")
for l in training_set:
    f.write(l)
f.close()

f = open("{}/validation_list.txt".format(dataset_dir), "w")
for l in validation_set:
    f.write(l)
f.close()

f = open("{}/testing_list.txt".format(dataset_dir), "w")
for l in testing_set:
    f.write(l)
f.close()
