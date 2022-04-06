import os
import pandas as pd
import numpy as np

import torch

DATA_PATH = "/home/songpengcao/Autox/Thesis/time-series"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, w=24, h=1):
        self.data_dir = os.path.join(DATA_PATH, 'Data', mode)
        self.load_path = os.path.join(self.data_dir, "load.csv")
        self.branch_path = os.path.join(self.data_dir, "branch.csv")
        self.label_path = os.path.join(self.data_dir, "label.csv")

        self.load = np.loadtxt(self.load_path, delimiter=',')      # (17520, 41)
        self.branch = np.loadtxt(self.branch_path, delimiter=',')  # (17520, 41)
        self.label = np.loadtxt(self.label_path, delimiter=',').T  # (17520, 41)

        self.bus_num = self.load.shape[1]
        self.branch_num = self.branch.shape[1]
        self.sample_num = self.branch.shape[0]

        self.window = w
        self.horizon = h

        self.get_data()

    def get_data(self):
        rng = range(self.window + self.horizon - 1, self.sample_num)
        n = len(rng)

        self.load_X = np.zeros([n, self.window, self.bus_num])
        self.branch_X = np.zeros([n, self.window, self.branch_num])

        self.Y = np.zeros([n, self.branch_num])  # shape = (17496, 41)
        self.Y_label = np.zeros([n, self.branch_num])
        
        for i in range(n):
            end = rng[i] - self.horizon + 1
            start = end - self.window

            self.load_X[i, :, :] = self.load[start:end, :]
            self.branch_X[i, :, :] = self.branch[start:end, :]
            self.Y[i, :]    = self.branch[rng[i], :]
            self.Y_label[i, :] = self.label[rng[i], :]

        self.X = np.concatenate((self.load_X, self.branch_X), axis=2)  # shape = (17496, 24, 71)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :], self.Y[idx, :], self.Y_label[idx, :]


def main():
    dataset = Dataset("train")
    x, y, l = dataset[0]
    print(x.shape)  # (24, 71)
    print(y.shape)  # (41, )
    print(l.shape)  # (41, )

if __name__ == "__main__":
    print('Start running: {}'.format(os.path.basename(__file__)))
    main()