import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch.fft as fft

class Load_Dataset(Dataset):
    def __init__(self, dataset, finetune=False, percent=1):
        super().__init__()
        if finetune:
            x_data = dataset['samples'][:int(1000*percent)]
            y_data = dataset['labels'][:int(1000*percent)]
        else:
            x_data = dataset['samples']
            y_data = dataset['labels']
        if isinstance(x_data, np.ndarray):
            self.t_data = torch.from_numpy(x_data).float()
            self.y_data = torch.from_numpy(y_data).long()
        else:
            self.t_data = x_data.float()
            self.y_data = y_data.long()

        '''Frequency domain'''
        self.f_data = fft.rfft(self.t_data, 1023).abs()

    def __getitem__(self, index):
        return self.t_data[index], self.f_data[index], self.y_data[index]
    
    def __len__(self):
        return self.t_data.shape[0]
    
def Data_Loader(path, args, mode):
    if mode == 'train':
        data = torch.load(os.path.join(path, 'train.pt'))
        dataset = Load_Dataset(data)
        loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    elif mode == 'finetune':
        data = torch.load(os.path.join(path, 'finetune.pt'))
        dataset = Load_Dataset(data, True, args.percent)
        loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    else:
        data = torch.load(os.path.join(path, 'test.pt'))
        dataset = Load_Dataset(data)
        loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)

    return loader
