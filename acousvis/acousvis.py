import torch
from torch.nn import Conv1d
import torchaudio
import torch.fft as fft

import numpy as np

def make_spectral_map(inp_signal,
                      model = None,
                      n_final_bins = 512,
                      shift = 10,
                      device = None,
                      weights = None):
    """
        inp_signal: Input signal in form of an input tensor of size [1, signal_len]
        model: Raw Waveform model for which we are to extract the spectral information
        n_final_bins: Final output frequency resolution of the spectral map
        shift: Shift required for the kernel while computing the outer product on the inp_signal
    """
    n_final_bins = 2*n_final_bins  #Output number of bins are halved while taking FFT
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(model.training):
        model.eval()
        model = model.to(device)
    frequency_data = []
    time_data = []
    for filters in model.modules():
        if(isinstance(filters, Conv1d)):
            num_filters, in_channels, filter_size = filters.weight.size()
            
            if in_channels == 1:
                with torch.no_grad():
                    filter_weights = filters.weight.squeeze().numpy()
                    pad_zeros_right = torch.zeros(1, filter_size - 1).to(device)
                    modif_inp = torch.cat([inp_signal, pad_zeros_right], axis = 1).unsqueeze(0) # Pad the filters to match the frequency resolution
                    modif_inp = modif_inp.to(device)
                    
                    conv_operator = Conv1d(1, num_filters, filter_size, stride = shift, bias = False) # Taking the outer product for response calculation
                    conv_operator.weight = filters.weight
                    output_conv_map = conv_operator(modif_inp)
                    
                    time_data.append(output_conv_map.squeeze().numpy())
                    dft_filters = np.fft.rfft(filter_weights, n = n_final_bins, axis = 1) # Finding the frequency response of the filters
                    
                    frequency_data.append(dft_filters)
    frequency_data = np.concatenate(frequency_data, axis = 0)
    time_data = np.concatenate(time_data, axis = 0)
    
    spectral_map = frequency_data.T@time_data
    return spectral_map, inp_signal

def prepare_inps_spectral_maps(inp_signal, 
                               sample_rate = None,
                               model = None, 
                               n_final_bins = 512,
                               shift = 10,
                               device = None,
                               weights = None,
                               minmax_norm = True,
                               min_n = -0.1,
                               max_n = 0.1):
    """
        inp_signal: Path of the input signal or input signal in form of an input 
                        tensor of size [1, signal_len]
        sample_rate: Sampling rate of the signal to be specified in case of a tensor
                        as an input for inp_signal
        model: Raw Waveform model for which we are to extract the spectral information
        n_final_bins: Final output frequency resolution of the spectral map
        shift: Shift required for the kernel while computing the outer product on the inp_signal
        device: Specifying the device on which we have to carry out the operations, sets
                    cuda if a cuda device is present
        minmax_norm: Min-Max linear normalization required for normalizing a signal
        min_n: Max-value that the min-max norm has to normalize to
        max_n: Min-value that the min-max norm has to normalize to
    """
    if isinstance(inp_signal, str):
        inp_signal, sample_rate = torchaudio.load(inp_signal)
        inp_signal = inp_signal.to(device)
    else:
        inp_signal = torch.tensor(inp_signal, dtype = torch.float32).to(device)
    
    if minmax_norm:
        min_ = torch.min(inp_signal)
        max_ = torch.max(inp_signal)
        inp_signal = (max_n - min_n)*(inp_signal - min_)/(max_ - min_) + min_n
    
    spectral_map, inp_signal = make_spectral_map(inp_signal, \
                                                            model, n_final_bins, \
                                                            shift, device, weights)
    
    time_values_present = np.arange(spectral_map.shape[1]) * float(shift)/sample_rate
    frequencies_values_present = np.linspace(0, sample_rate/2, spectral_map.shape[0])
    time, frequency = np.meshgrid(time_values_present, frequencies_values_present)
    return spectral_map, time, frequency, time_values_present, \
            frequencies_values_present, inp_signal, sample_rate
