import torch
import numpy as np
from torch.utils.data import Dataset


class PatientDrugDataset(Dataset):

    def __init__(self, npy_file, label_file=None, transform=None):
        self.bow = np.load(npy_file)
        self.transform = transform
        self.target = False
        if label_file:
            self.target = True
            self.label = np.load(label_file)

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.bow[idx]
        if self.target:
            target = self.label[idx]
            sample = {'Data': torch.from_numpy(data), 'Target': torch.from_numpy(target)}

        else:
            sample = {'Data': torch.from_numpy(data)}


        if self.transform:
            sample = self.transform(sample)

        return sample, idx


class PatientDrugTestDataset(Dataset):

    def __init__(self, npy_file, label_file=None, transform=None):
        self.bow = np.load(npy_file)
        self.partition = int(len(self.bow) // 2)
        self.bow1 = self.bow[:self.partition]
        self.bow2 = self.bow[self.partition:]
        self.target = False
        self.transform = transform
        if label_file:
            self.target = True
            self.label = np.load(label_file)
            self.label = self.label[self.partition:]

    def __len__(self):
        return min(len(self.bow1),len(self.bow2))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data1 = self.bow1[idx]
        data2 = self.bow2[idx]
        if self.target:
            target = self.label[idx]
            sample = {'First': torch.from_numpy(data1), 'Second': torch.from_numpy(data2), 'Target': torch.from_numpy(target)}
        else:
            sample = {'First': torch.from_numpy(data1), 'Second': torch.from_numpy(data2)}

        if self.transform:
            sample = self.transform(sample)

        return sample, idx