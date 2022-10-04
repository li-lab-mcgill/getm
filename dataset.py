import torch
import numpy as np
from torch.utils.data import Dataset


class PatientDrugDataset(Dataset):

    def __init__(self, npy_file, y_file=None, mask_file=None, transform=None):
        self.bow = np.load(npy_file)
        self.transform = transform
        if y_file:
            self.e2e = True
            self.y = np.load(y_file)
            self.mask = np.load(mask_file)
        else:
            self.e2e = False
    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.bow[idx]
        if self.e2e:
            y = self.y[idx]
            mask = self.mask[idx]
            sample = {'Data': torch.from_numpy(data), "Y": torch.from_numpy(y)
                      , "Mask": torch.from_numpy(mask)}

        else:
            sample = {'Data': torch.from_numpy(data)}


        if self.transform:
            sample = self.transform(sample)

        return sample, idx

class MixDataset(Dataset):

    def __init__(self, npy_file, npy_t, y_file=None, mask_file=None, transform=None):
        self.bow = np.load(npy_file)
        self.bow_t = np.load(npy_t)
        self.transform = transform
        if y_file:
            self.e2e = True
            self.y = np.load(y_file)
            self.mask = np.load(mask_file)
        else:
            self.e2e = False


    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.bow[idx]
        data_t = self.bow_t[idx]

        if not self.e2e:
            sample = {'Data': torch.from_numpy(data), 'Data_t': torch.from_numpy(data_t)}
        else:
            y = self.y[idx]
            mask = self.mask[idx]
            sample = {'Data': torch.from_numpy(data), 'Data_t': torch.from_numpy(data_t), "Y": torch.from_numpy(y)
                      , "Mask": torch.from_numpy(mask)}


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