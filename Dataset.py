import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
import re
from params import *
import pretty_midi
import librosa
import os

class AudioDataset(Dataset):
    def __init__(self,data_paths,labels:pd.DataFrame,mode,device='cpu',preload=False, cache_size=0):
        super().__init__()
        self.rng = np.random.default_rng(seed=SEED)
        
        self.data_paths = data_paths
        self.cache_size = cache_size
        self.preload = preload
        self.device = device
        if preload or cache_size == -1:
            self.data = [torch.flatten(torch.load(path,map_location=self.device)) for path in self.data_paths]
            self.preload = True
            self.cache_size = -1
        elif cache_size == 0:
            self.data = []
        else:
            self.data = {}
        
        self.labels = labels.drop_duplicates().drop(columns=["path"])
        self.mode = mode
        
        
            
    def _get_data(self,index,start):

        if self.cache_size == 0:
            
            sample = torch.flatten(torch.load(self.data_paths[index],map_location=self.device))
            return self._load_sample(sample=sample,sr=SAMPLING_RATE,start=start,duration=SAMPLE_LENGTH)
         
        if self.preload:
            return self._load_sample(sample=self.data[index],sr=SAMPLING_RATE,start=start,duration=SAMPLE_LENGTH)
        
        if index not in self.data:
            if self.cache_size <= len(self.data):
                self.data.popitem()
            self.data[index] = torch.flatten(torch.load(self.data_paths[index],map_location=self.device))

        return self._load_sample(sample = self.data[index],sr = SAMPLING_RATE,start=start,duration=SAMPLE_LENGTH)
        

    def __getitem__(self, index):
        
        sample_id = int(re.sub('\..*',"",os.path.basename(self.data_paths[index])))
        

        label = self.labels.loc[sample_id]
        n_channels = 1
        if self.mode == 2:
            n_channels = N_CHANNELS
        
        start = np.sort(self.rng.random(n_channels)*(label.seconds - SAMPLE_LENGTH))
        #start = np.sort(self.rng.random(n_channels)*np.array([(i+1)*(label.seconds - SAMPLE_LENGTH)/n_channels for i in range(n_channels)]))

        sample = self._get_data(index,start)

        
        label1 = label["composer"]
        label2 = label.drop(["composer","seconds"]).to_numpy()

        return sample, (label1.astype(int), label2.astype(int))
        
    def __len__(self):
        return len(self.data_paths)
    
    def _load_sample(self,path = None ,sample=None, sr=SAMPLING_RATE,start=[0],duration=SAMPLE_LENGTH) -> torch.Tensor: 
        
        if path != None:
            sample, _ = librosa.load(path,sr=sr,offset=start[0],duration=duration)
            sample = torch.tensor([sample])
            
        elif sample != None:
            sample = torch.stack([sample[int(start[i]*sr):int(start[i]*sr)+sr*duration] for i in range(len(start))])
            
            
        
        if self.mode == 0:
            return sample.flatten().type(torch.float32)
        
        sample = spectrogram(sample,sample_rate=sr,device=self.device)
    
        
        if self.mode == 1:
            return sample.flatten().type(torch.float32)
        

        t = transforms.Compose([
            transforms.Resize(size=(128,312)),
            transforms.ConvertImageDtype(torch.float32)
        ])
        return t(sample)
    

def get_dataloader(data,labels, batchSize, shuffle=True,mode=0,device='cpu',preload=False, cache_size=0):
	# create a dataset and use it to create a data loader
    ds = AudioDataset(data,labels,mode,device=device,preload=preload, cache_size=cache_size)
    
    if device == DEVICE:
        loader = DataLoader(ds, 
                        batch_size=batchSize,
                        shuffle=shuffle)
                        
        # return a tuple of  the dataset and the data loader
        return (ds, loader)
    loader = DataLoader(ds, batch_size=batchSize,shuffle=shuffle)
    return (ds, loader)

    
    


def fft_filter(sample,tol=0.1):
    newSample = np.fft.fft(sample)
    mask = np.abs(newSample)/len(sample)<=tol*np.max(np.abs(newSample)/len(sample))
    newSample[mask] = 0
    newSample = np.fft.ifft(newSample).real
    return newSample



def spectrogram_custom(samples, sample_rate, stride_ms = 16.0, 
                          window_ms = 32.0, max_freq = None, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram


def spectrogram(sample,sample_rate, stride_ms = 16.0, 
                          window_ms = 32.0, power=2,device='cpu'):
    

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)
    n_bins = window_size

    trans = torchaudio.transforms.Spectrogram(n_bins,win_length=window_size,hop_length=stride_size, power=power).to(device=device)
    sample = trans(sample)
    mag = sample.abs()

    return torch.log(mag + 1e-14)