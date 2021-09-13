import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv1d, ELU, MaxPool1d, BatchNorm1d, Dropout, Linear, Flatten, Softmax, ReLU
import torch.utils.data as Data
from collections import OrderedDict
import os
from tqdm import tqdm
import json

from models import *
from datasets import *
from flags import *

if os.path.exists(checkpoint_path):
    print("Loading from the previous checkpoint")
    checkpoint = torch.load(checkpoint_path)
    epoch_resume = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print("New model created")

model = model.to(device)

training_loss = []
validation_loss = []
validation_accuracy = []
for epoch in range(epoch_resume, num_epochs + 1):
    if(epoch%3==0):
        #We validate
        total, correct = 0, 0
        for audio_waveform, label in tqdm(dataloader_validation, "Validation Epoch: {}".format(epoch)):
            a_w_cuda = audio_waveform.to(device)
            final_out = model(a_w_cuda)
            loss = model_loss(final_out, label.to(device)).detach().cpu().item()
            validation_loss.append(loss)
            model.eval()
            class_wise_out = model(a_w_cuda)
            _, predicted = torch.max(class_wise_out.detach().cpu(), axis = 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            model.train()
        accuracy = (float(correct)/total) * 100
        validation_accuracy.append(accuracy)
        
    for audio_waveform, label in tqdm(dataloader_train, "Training Epoch: {}".format(epoch)):
        optimizer.zero_grad()
        a_w_cuda = audio_waveform.to(device)
        final_out = model(a_w_cuda)
        loss = model_loss(final_out, label.to(device))
        loss.backward(retain_graph = True)
        optimizer.step()
        training_loss.append(loss.detach().cpu().item())
    
    a_name = analysis_path + str(epoch) + ".pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
    model.eval()
    torch.save(model.state_dict(), a_name)
    model.train()

with open(logfile, "w") as f:
    json.dump({"Validation Loss": validation_loss,
          "Validation Accuracy": validation_accuracy,
          "Training Loss": training_loss}, f)

model.eval()
torch.save(model.state_dict(), final_model_path)