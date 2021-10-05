import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ELU, MaxPool1d, BatchNorm1d, Dropout, Linear, Flatten, Softmax, ReLU
import torch.utils.data as Data
from collections import OrderedDict

''' Examples of model definition for raw-waveform based CNNs 
    with segmental (1-3 pitch period), subsegmental and multiresolution kernels
'''    

def manual_pad(t, pad_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b_size, chan, _ = t.size()
    front = torch.zeros(b_size, chan, ceil(pad_val/2)).to(device)
    back = torch.zeros(b_size, chan, floor(pad_val/2)).to(device)
    t = torch.cat([back, t, front], axis = 2).to(device)
    return t

class RawWaveFormCNN_Segmental(nn.Module):
    def __init__(self, input_dim = 16000, num_classes = 10):
        super(RawWaveFormCNN_Segmental, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_net = nn.Sequential(OrderedDict([
            ('conv_1', Conv1d(in_channels = 1, out_channels = 80, kernel_size = 300, stride = 10)),
            ('elu_1', ELU(inplace = True)),
            ('batch_norm_1', BatchNorm1d(num_features = 80)),
            ('maxpool_1', MaxPool1d(kernel_size = 2)),
            ('conv_2', Conv1d(in_channels = 80, out_channels = 100, kernel_size = 7, stride = 1)),
            ('elu_2', ELU(inplace = True)),
            ('batch_norm_2', BatchNorm1d(num_features = 100)),
            ('maxpool_2', MaxPool1d(kernel_size = 2)),
            ('conv_3', Conv1d(in_channels = 100, out_channels = 100, kernel_size = 3, stride = 1)),
            ('elu_3', ELU(inplace = True)),
            ('batch_norm_3', BatchNorm1d(num_features = 100)),
            ('maxpool_3', MaxPool1d(kernel_size = 2)),
            ('flatten', Flatten()),
            ('Linear_1', Linear(in_features = 19300, out_features = 1024)),
            ('relu', ReLU(inplace = True)),
            ('dropout', Dropout(0.25)),
            ('Linear_2', Linear(in_features = 1024, out_features = self.num_classes))
        ]))
    
    def forward(self, x):
        if self.training:
            return self.seq_net(x)
        else:
            return F.softmax(self.seq_net(x))



class RawWaveFormCNN_SubSegmental(nn.Module):
    def __init__(self, input_dim = 16000, num_classes = 10):
        super(RawWaveFormCNN_SubSegmental, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_net = nn.Sequential(OrderedDict([
            ('conv_1', Conv1d(in_channels = 1, out_channels = 80, kernel_size = 30, stride = 10)),
            ('elu_1', ELU(inplace = True)),
            ('batch_norm_1', BatchNorm1d(num_features = 80)),
            ('maxpool_1', MaxPool1d(kernel_size = 2)),
            ('conv_2', Conv1d(in_channels = 80, out_channels = 100, kernel_size = 7, stride = 1)),
            ('elu_2', ELU(inplace = True)),
            ('batch_norm_2', BatchNorm1d(num_features = 100)),
            ('maxpool_2', MaxPool1d(kernel_size = 2)),
            ('conv_3', Conv1d(in_channels = 100, out_channels = 100, kernel_size = 3, stride = 1)),
            ('elu_3', ELU(inplace = True)),
            ('batch_norm_3', BatchNorm1d(num_features = 100)),
            ('maxpool_3', MaxPool1d(kernel_size = 2)),
            ('flatten', Flatten()),
            ('Linear_1', Linear(in_features = 19700, out_features = 1024)),
            ('relu', ReLU(inplace = True)),
            ('dropout', Dropout(0.25)),
            ('Linear_2', Linear(in_features = 1024, out_features = self.num_classes))
        ]))
    
    def forward(self, x):
        if self.training:
            return self.seq_net(x)
        else:
            return F.softmax(self.seq_net(x))


class RawWaveFormCNN_MultiRes(nn.Module):
    def __init__(self, input_dim = 16000, num_classes = 10):
        super(RawWaveFormCNN_MultiRes, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.layer_1_1 = Conv1d(in_channels = 1, out_channels = 10, kernel_size = 10, stride = 10)
        self.layer_1_2 = Conv1d(in_channels = 1, out_channels = 10, kernel_size = 30, stride = 10, padding = 10)
        self.layer_1_3 = Conv1d(in_channels = 1, out_channels = 10, kernel_size = 50, stride = 10, padding = 20)
        self.layer_1_4 = Conv1d(in_channels = 1, out_channels = 10, kernel_size = 70, stride = 10, padding = 30)
        self.layer_1_5 = Conv1d(in_channels = 1, out_channels = 10, kernel_size = 90, stride = 10, padding = 40)
        self.layer_1_6 = Conv1d(in_channels = 1, out_channels = 10, kernel_size = 150, stride = 10, padding = 70)
        self.layer_1_7 = Conv1d(in_channels = 1, out_channels = 10, kernel_size = 300, stride = 10, padding = 145)
        self.layer_1_8 = Conv1d(in_channels = 1, out_channels = 10, kernel_size = 500, stride = 10, padding = 245)
        
        self.elu_1 = ELU()
        self.batch_norm_1 = BatchNorm1d(num_features = 80)
        self.maxpool_1 = MaxPool1d(kernel_size = 3)
        
        self.layer_2_1 = Conv1d(in_channels = 80, out_channels = 20, kernel_size = 3, stride = 1)
        self.layer_2_2 = Conv1d(in_channels = 80, out_channels = 20, kernel_size = 6, stride = 1)
        self.layer_2_3 = Conv1d(in_channels = 80, out_channels = 20, kernel_size = 9, stride = 1, padding = 3)
        self.layer_2_4 = Conv1d(in_channels = 80, out_channels = 20, kernel_size = 12, stride = 1)
        self.layer_2_5 = Conv1d(in_channels = 80, out_channels = 20, kernel_size = 15, stride = 1, padding = 6)
        
        self.elu_2 = ELU()
        self.batch_norm_2 = BatchNorm1d(num_features = 100)
        self.maxpool_2 = MaxPool1d(kernel_size = 3)
        
        self.layer_3_1 = Conv1d(in_channels = 100, out_channels = 25, kernel_size = 3, stride = 1)
        self.layer_3_2 = Conv1d(in_channels = 100, out_channels = 25, kernel_size = 5, stride = 1, padding = 1)
        self.layer_3_3 = Conv1d(in_channels = 100, out_channels = 25, kernel_size = 7, stride = 1, padding = 2)
        self.layer_3_4 = Conv1d(in_channels = 100, out_channels = 25, kernel_size = 9, stride = 1, padding = 3)
        
        self.elu_3 = ELU()
        self.batch_norm_3 = BatchNorm1d(num_features = 100)
        
        self.dense_part = nn.Sequential(OrderedDict([('flatten', Flatten()),
            ('Linear_1', Linear(in_features = 17500, out_features = 1024)),
            ('relu', ReLU(inplace = True)),
            ('dropout', Dropout(0.25)),
            ('Linear_2', Linear(in_features = 1024, out_features = self.num_classes))
        ]))

    def forward(self, x):
        layer_1_out_1 = self.layer_1_1(x)
        layer_1_out_2 = self.layer_1_2(x)
        layer_1_out_3 = self.layer_1_3(x)
        layer_1_out_4 = self.layer_1_4(x)
        layer_1_out_5 = self.layer_1_5(x)
        layer_1_out_6 = self.layer_1_6(x)
        layer_1_out_7 = self.layer_1_7(x)
        layer_1_out_8 = self.layer_1_8(x)
        
        layer_1_out = torch.cat([layer_1_out_1, layer_1_out_2, layer_1_out_3, layer_1_out_4, layer_1_out_5, layer_1_out_6, layer_1_out_7, layer_1_out_8], axis = 1)
        layer_1_out = self.elu_1(layer_1_out)
        layer_1_out = self.batch_norm_1(layer_1_out)
        layer_1_out = self.maxpool_1(layer_1_out)
        
        layer_2_out_1 = self.layer_2_1(layer_1_out)
        layer_2_out_2 = self.layer_2_2(manual_pad(layer_1_out, 3))
        layer_2_out_3 = self.layer_2_3(layer_1_out)
        layer_2_out_4 = self.layer_2_4(manual_pad(layer_1_out, 9))
        layer_2_out_5 = self.layer_2_5(layer_1_out)
        
        layer_2_out = torch.cat([layer_2_out_1, layer_2_out_2, layer_2_out_3, layer_2_out_4, layer_2_out_5], axis = 1)
        layer_2_out = self.elu_2(layer_2_out)
        layer_2_out = self.batch_norm_2(layer_2_out)
        layer_2_out = self.maxpool_2(layer_2_out)
        
        layer_3_out_1 = self.layer_3_1(layer_2_out)
        layer_3_out_2 = self.layer_3_2(layer_2_out)
        layer_3_out_3 = self.layer_3_3(layer_2_out)
        layer_3_out_4 = self.layer_3_4(layer_2_out)
        
        layer_3_out = torch.cat([layer_3_out_1, layer_3_out_2, layer_3_out_3, layer_3_out_4], axis = 1)
        layer_3_out = self.elu_3(layer_3_out)
        layer_3_out = self.batch_norm_3(layer_3_out)
        
        final_out = self.dense_part(layer_3_out)
        
        if self.training:
            return final_out
        else:
            return F.softmax(final_out)


