import torch
from math import ceil, floor
import matplotlib.pyplot as plt
import json

def manual_pad(t, pad_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b_size, chan, _ = t.size()
    front = torch.zeros(b_size, chan, ceil(pad_val/2)).to(device)
    back = torch.zeros(b_size, chan, floor(pad_val/2)).to(device)
    t = torch.cat([back, t, front], axis = 2).to(device)
    return t

def plot(logfile):
    with open(logfile, "r") as f:
        log = json.load(f)

    training_loss = log["Training Loss"]
    validation_loss = log["Validation Loss"]
    validation_accuracy = log["Validation Accuracy"]

    plt.plot(range(len(training_loss)), training_loss)
    plt.title("Training Loss")
    plt.savefig("training_loss.jpg")
    plt.cla()

    plt.plot(range(len(validation_loss)), validation_loss)
    plt.title("Validation Loss")
    plt.savefig("validation_loss.jpg")
    plt.cla()

    plt.plot(range(len(validation_accuracy)), validation_accuracy)
    plt.title("Validation Accuracy")
    plt.savefig("validation_acc.jpg")
    plt.cla()
