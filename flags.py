import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from collections import OrderedDict
from random import shuffle
import os
from datetime import datetime

from models import *
from datasets import *


#FLAGS and Hyperparameters
num_epochs = 50
epoch_resume = 1
batch_size = 32
lr = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_name = "speaker_classification"
model_loss = nn.CrossEntropyLoss()
type_class = 3    #0: 10 classes      1: 20 classes        2: All Classes
type_kernel = 3     #1: Short kernel      2: Long kernel        3:Multi-feature
if task_name == "speech_classification":
    classes = [['go', 'right', 'on', 'down', 'left', 'up', 'yes', 'stop', 'off', 'no'], 
               ['go', 'right', 'on', 'down', 'left', 'up', 'yes', 'stop', 'off', 'no', 'eight', 'five', 'cat', 'dog', 'bed', 'wow', 'three', 'visual', 'two', 'learn'],
               None]
if task_name == "speaker_identification":
    clss = sorted(os.listdir("/ssd-scratch/devansh19160/LibriSpeech/train-clean-100/"))
    clss = [i for i in clss if i[-3:]!="txt"]
    classes = [clss[:50], clss[:100], clss[:150], clss[:200], None]
print(device)


kernel = None

if type_kernel == 3:
    print("Multifeature kernel")
    kernel = "multifeat"

if type_kernel == 1:
    print("Short kernel")
    kernel = "short"

if type_kernel == 2:
    print("Long kernel")
    kernel = "long"

final_model_path = "/scratch/devansh19160/{}/Final Model/{}_dataset_{}_{}_final_model.pt".format(kernel, task_name, type_class, kernel)
checkpoint_path = "/scratch/devansh19160/{}/Checkpoints/{}_dataset_{}_{}_ckpt.pt".format(kernel, task_name, type_class, kernel)
analysis_path = "/scratch/devansh19160/{}/Final Model/Analysis/{}_dataset_{}_{}_ckpt_".format(kernel, task_name, type_class, kernel)

#Dataset and Dataloader
dataset_train = SpeechClassificationDataset(split = "train", specified_classes = classes[type_class])
dataset_validation = SpeechClassificationDataset(split = "val", specified_classes = classes[type_class])
dataset_test = SpeechClassificationDataset(split = "test", specified_classes = classes[type_class])

dataloader_train = Data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 2)
dataloader_validation = Data.DataLoader(dataset_validation, batch_size = batch_size, shuffle = True, num_workers = 2)
dataloader_test = Data.DataLoader(dataset_test, batch_size = batch_size, shuffle = True, num_workers = 2)

print("Train: ", len(dataloader_train), " Val: ", len(dataloader_validation), " Test: ", len(dataloader_test))

num_classes = dataset_train.num_classes
print("Num Classes: {}".format(num_classes))

timestamp = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
logfile = "/scratch/devansh19160/{}/{}_{}_{}_logger_{}.json".format(kernel, task_name, kernel, num_classes, timestamp)

if type_kernel == 1:
    model = RawWaveFormCNN_SingleRes_Short(num_classes = num_classes).to(device)

if type_kernel == 2:
    model = RawWaveFormCNN_SingleRes_Long(num_classes = num_classes).to(device)

if type_kernel == 3:
    model = RawWaveFormCNN_MultiRes(num_classes = num_classes).to(device)

def init_weights(m):
    if type(m) == Linear or type(m)==Conv1d:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

optimizer = optim.AdamW(model.parameters(), lr = lr)
