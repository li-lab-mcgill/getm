import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix

from IPython import embed

def coo2tensor(coo):
    # FIXME: not sparse tensor
    # return torch.from_numpy(coo.toarray())
    i = torch.LongTensor(np.vstack((coo.row, coo.col)))
    v = torch.FloatTensor(coo.data)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

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

class MixDataset(Dataset):

    def __init__(self, npy_file, npy_t, transform=None):
        self.bow = np.load(npy_file, allow_pickle=True)
        self.bow_t = np.load(npy_t, allow_pickle=True)
        self.transform = transform


    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.bow[idx]
        data_t = self.bow_t[idx]
        
        if isinstance(data, coo_matrix):
            data = coo2tensor(data)
            data_t = coo2tensor(data_t)
        else:
            data = torch.from_numpy(data)
            data_t = torch.from_numpy(data_t)

        sample = {'Data': data, 'Data_t': data_t}


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


class MTDataset(Dataset):

    def __init__(self, npy_file, age_file, visit_file, transform=None):
        self.bow = np.load(npy_file)
        self.age = np.load(age_file)
        self.visit = np.load(visit_file)
        self.transform = transform

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.bow[idx]
        ages = self.age[idx]
        visits = self.visit[idx]

        sample = {'Data': torch.from_numpy(data), "Age": ages, "Visits": visits}

        if self.transform:
            sample = self.transform(sample)

        return sample, idx

class LSTMDataset(Dataset):

    def __init__(self, npy_file, label_file, mask_file, transform=None):
        self.bow = np.load(npy_file)
        self.transform = transform
        self.label = np.load(label_file)
        self.mask = np.load(mask_file)

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.bow[idx]
        target = self.label[idx]
        mask = self.mask[idx]
        sample = {'Data': torch.from_numpy(data), 'Target': torch.from_numpy(target),
                  'Mask': torch.from_numpy(mask)}

        if self.transform:
            sample = self.transform(sample)

        return sample, idx

class SimulationDataset(Dataset):

    def __init__(self, npy_file, npy_t, theta_file=None, eta_file=None, transform=None):
        self.bow = np.load(npy_file)
        self.bow_t = np.load(npy_t)
        self.transform = transform
        self.theta_bool = False
        if theta_file:
            self.theta_bool = True
            self.theta = np.load(theta_file)
            self.eta = np.load(eta_file)

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.bow[idx]
        data_t = self.bow_t[idx]
        if self.theta_bool:
            theta = self.theta[:, idx, :]
            eta = self.eta[:, idx, :]
            sample = {'Data': torch.from_numpy(data), 'Theta': torch.from_numpy(theta),
                  'Eta': torch.from_numpy(eta), "Data_t": data_t}
        else:
            sample = {'Data': torch.from_numpy(data), "Data_t": data_t}
        if self.transform:
            sample = self.transform(sample)

        return sample, idx