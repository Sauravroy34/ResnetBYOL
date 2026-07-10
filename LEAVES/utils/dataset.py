from distutils.command.config import config
from random import random
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from utils.data_utils import normalize_data, Catergorical2OneHotCoding
from utils import augmentation as aug
import random
import configs
import copy
import torch

class TransDataset(Dataset):
    def __init__(self, filename, data_normalization=True, is_training=True):
        super(TransDataset).__init__()
        self.is_training = is_training
        
        # first, check whether or not the file exist
        # filename_data must not be none.
        if not os.path.isfile(filename):
            print(filename + "doesn't exist!\n")
            exit(0)
        # then load the data.
        if filename.split('.')[-1] == "csv":
            data = pd.read_csv(filename, sep='\t', header=None).values
            self.data_y = data[:, 0]
            self.data_x = data[:, 1:]
            self.data_y = Catergorical2OneHotCoding(self.data_y.astype(np.int8))
        else:
            data_dict = np.load(filename, allow_pickle=True).item();
            self.data_x = data_dict['ECG_signal']
            self.data_x = np.transpose(self.data_x, (0,2,1))
            self.data_y = data_dict['label']
            self.data_y = Catergorical2OneHotCoding(self.data_y.astype(np.int8).reshape(-1,), 
                                                    num_class=configs.num_classes)

        if data_normalization:
            std_ = self.data_x.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            self.data_x = (self.data_x - self.data_x.mean(axis=1, keepdims=True)) / std_
        if len(self.data_x) == 1:
            self.data_x = np.expand_dims(self.data_x, axis=-1)

    def __len__(self):
        return self.data_x.shape[0]

    def normalization(self, x):
        # x = x.reshape((1,-1))
        x = (x - np.min(x, axis=1, keepdims=True))/(np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True) + 0.00000001)
        return x
    
    def transformation(self, x):
        x = x.T
        # args = random.choice(['jitter', 'scaling', 'permutation', 'rotation', 'magnitudewarp', 'timewarp', 'windowslice', 'windowwarp'])
        args = random.choice(['jitter', 'scaling', 'permutation', 'rotation', 'magnitudewarp', 'timewarp', 'original'])
        if args == 'jitter':
            x = aug.jitter(x, sigma=configs.noise_sigma)
        elif args == 'scaling':
            x = aug.scaling(x, sigma=configs.noise_sigma)
        elif args == 'permutation':
            x = aug.permutation(x)
        elif args == 'rotation':
            x = aug.rotation(x)
        elif args == 'magwarp':
            x = aug.magnitude_warp(x, sigma=configs.warp_sigma)
        elif args == 'timewarp':
            x = aug.time_warp(x, sigma=configs.warp_sigma)
        elif args == 'windowslice':
            x = aug.window_slice(x)
        elif args == 'windowwarp':
            x = aug.window_warp(x)
        else:
            pass
        x = x.T
        x = self.normalization(x)
        return x
    
    def __getitem__(self, index):
        x =  self.data_x[index]
        if len(x.shape) == 1:
            x = x.reshape((configs.in_channel,-1))
        y =  self.data_y[index]
        # self.is_training = False
        if self.is_training:
            if not configs.leaves_configs["use_leaves"]:
                x1 = self.transformation(x)
                x2 = self.transformation(x)
                return x1, x2, y
            else:
                x = self.normalization(x)
                return x, x.copy(), y
        else:
            x = self.normalization(x)
            return x, y
            
class SleepEDFE_Dataset(Dataset):
    def __init__(self, filename, data_normalization=True, is_training=True):
        super(TransDataset).__init__()
        self.is_training = is_training
        
        # then load the data.
        if is_training:
            filename_eog = "/home/hanyu/data/sleep_edfe/train_EOG.csv"
            filename_eeg = "/home/hanyu/data/sleep_edfe/train_EEG.csv"
        else:
            filename_eog = "/home/hanyu/data/sleep_edfe/test_EOG.csv"
            filename_eeg = "/home/hanyu/data/sleep_edfe/test_EEG.csv"
        data_eog = pd.read_csv(filename_eog, sep='\t', header=None).values
        data_eeg = pd.read_csv(filename_eeg, sep='\t', header=None).values
        self.data_y = data_eog[:, 0]
        self.data_x_eog = data_eog[:, 1:]
        self.data_x_eeg = data_eeg[:, 1:]
        self.data_y = Catergorical2OneHotCoding(self.data_y.astype(np.int8))

        if data_normalization:
            std_ = self.data_x_eog.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            self.data_x_eog = (self.data_x_eog - self.data_x_eog.mean(axis=1, keepdims=True)) / std_
            
            std_ = self.data_x_eeg.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            self.data_x_eeg = (self.data_x_eeg - self.data_x_eeg.mean(axis=1, keepdims=True)) / std_
            
        if len(self.data_x_eeg) == 1:
            self.data_x_eeg = np.expand_dims(self.data_x_eeg, axis=-1)
            
        if len(self.data_x_eog) == 1:
            self.data_x_eog = np.expand_dims(self.data_x_eog, axis=-1)

    def __len__(self):
        return self.data_x_eog.shape[0]

    def normalization(self, x):
        # x = x.reshape((1,-1))
        x = (x - np.min(x, axis=1, keepdims=True))/(np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True) + 0.00000001)
        return x
    
    def transformation(self, x):
        x = x.T
        # args = random.choice(['jitter', 'scaling', 'permutation', 'rotation', 'magnitudewarp', 'timewarp', 'windowslice', 'windowwarp'])
        args = random.choice(['jitter', 'scaling', 'permutation', 'rotation', 'magnitudewarp', 'timewarp', 'original'])
        if args == 'jitter':
            x = aug.jitter(x, sigma=configs.noise_sigma)
        elif args == 'scaling':
            x = aug.scaling(x, sigma=configs.noise_sigma)
        elif args == 'permutation':
            x = aug.permutation(x)
        elif args == 'rotation':
            x = aug.rotation(x)
        elif args == 'magwarp':
            x = aug.magnitude_warp(x, sigma=configs.warp_sigma)
        elif args == 'timewarp':
            x = aug.time_warp(x, sigma=configs.warp_sigma)
        elif args == 'windowslice':
            x = aug.window_slice(x)
        elif args == 'windowwarp':
            x = aug.window_warp(x)
        else:
            pass
        x = x.T
        x = self.normalization(x)
        return x
    
    def __getitem__(self, index):
        x_eeg =  self.data_x_eeg[index]
        x_eog =  self.data_x_eog[index]
        if len(x_eog.shape) == 1:
            x_eog = x_eog.reshape((configs.in_channel1,-1))
        if len(x_eeg.shape) == 1:
            x_eeg = x_eeg.reshape((configs.in_channel2,-1))
        y =  self.data_y[index]
        # self.is_training = False
        if self.is_training:
            if not configs.leaves_configs["use_leaves"]:
                x1 = self.transformation(x_eog)
                x2 = self.transformation(x_eeg)
                return x1, x2, y
            else:
                x_eog = self.normalization(x_eog)
                x_eeg = self.normalization(x_eeg)
                return x_eog, x_eeg, y
        else:
            x_eog = self.normalization(x_eog)
            x_eeg = self.normalization(x_eeg)
            return x_eog, x_eeg, y
            
            
class SemiSupDatasetSMILE(Dataset):
    def __init__(self, mode="semi", augmentation=False):
        super().__init__()

        self.augmentation = augmentation
        self.mode = mode
        self.ecg = np.load("/home/hanyu/data/SMILE/semi_supervised/segments/all_segments/ecg_train.npy")
        self.gsr = np.load("/home/hanyu/data/SMILE/semi_supervised/segments/all_segments/gsr_train.npy")

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        ecg = np.array(self.ecg[idx]).reshape(1,-1)
        
        gsr = np.array(self.gsr[idx]).reshape(1,-1)
        return torch.Tensor(ecg), torch.Tensor(gsr), torch.Tensor([1])
    
    
class SupervisedDataset(Dataset):
    def __init__(self, mode="train", onehot=False):
        super().__init__()
        
        if mode == "train":
            self.ecg = np.load("/home/hanyu/data/SMILE/semi_supervised/segments/window_{}_min/ecg_train.npy".format(configs.supResolution))
            self.gsr = np.load("/home/hanyu/data/SMILE/semi_supervised/segments/window_{}_min/gsr_train.npy".format(configs.supResolution))
            self.label = np.load("/home/hanyu/data/SMILE/semi_supervised/segments/window_{}_min/label_train.npy".format(configs.supResolution))
        elif mode == "test":
            self.ecg = np.load("/home/hanyu/data/SMILE/semi_supervised/segments/ecg_test_.npy")
            self.gsr = np.load("/home/hanyu/data/SMILE/semi_supervised/segments/gsr_test_.npy")
            self.label = np.load("/home/hanyu/data/SMILE/semi_supervised/segments/label_test_.npy")
        else:
            print("Incorrect mode")
            
        self.label = self.label - 1
        self.label = self.label.astype(int)
        self.label[self.label > 0] = 1
        if onehot:
            self.label = Catergorical2OneHotCoding(self.label)
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        ecg = torch.Tensor(self.ecg[idx].reshape((1,-1)))
        gsr = torch.Tensor(self.gsr[idx].reshape((1,-1)))
        label = torch.LongTensor([self.label[idx]])
        return ecg, gsr, label

def test():
    # something need to be test here.
    print("Test a function!")

if __name__ == "__main__":
    test()
    print("Everything passed")