import torch
import torch.utils.data as Data
import torchaudio
import os

class SpeechClassificationDataset(Data.Dataset):
    def __init__(self, max_len = 16000 ,data_folder = "/ssd-scratch/devansh19160/LibriSpeech/train-clean-100/", split = "train", specified_classes = None):
        self.max_len = max_len
        self.data_folder = data_folder
        self.labels_dict = {}
        self.num_classes = 0
        self.classes = os.listdir(data_folder)
        if specified_classes:
            self.classes = specified_classes
        for class_name in self.classes:
            if(class_name == "_background_noise_" or class_name[-3:] == "txt"):
                continue
            try:
                self.labels_dict[class_name] = self.num_classes
                self.num_classes+=1
            except:
                pass
        
        self.labels_text_file = data_folder
        if(split == "train"):
            self.labels_text_file += "training_list.txt"
        if(split == "test"):
            self.labels_text_file += "testing_list.txt"
        if(split == "val"):
            self.labels_text_file += "validation_list.txt"
        
        self.file_labels = None
        
        with open(self.labels_text_file) as f:
            self.file_labels = f.readlines()
            f.close()
        
        self.final_file_labels = []
        for i, file_lab in enumerate(self.file_labels):
            try:
                file_name, label_ = file_lab, self.labels_dict[file_lab[:file_lab.find("/")]]
                self.final_file_labels.append((file_name[:-1], label_))
            except:
                pass
    
    def __len__(self):
        return len(self.final_file_labels)
    
    def __getitem__(self, idx):
        file_path, label = self.final_file_labels[idx]
        waveform, sample_rate = torchaudio.load(self.data_folder + file_path)
        if(waveform.size()[1]>self.max_len):
            waveform = waveform[:,:self.max_len]
        else:
            waveform = torch.cat([waveform, torch.zeros(1, self.max_len - waveform.size()[1])], axis = 1)
        return waveform, label
