import os
import pandas as pd
import numpy as np

import torch

DATA_PATH = "/home/colorful/jupyter_workspace/Albert/Congestion-long term/Data"

def series_to_supervised(data, n_in=24, n_out=1, dropnan=True):
    """
    将时间序列重构为监督学习数据集
    :params data: 观测值序列, 类型为Dataframe
    :params n_in: 输入的滞后观测值(X)长度
    :params n_out: 输出观测值(y)的长度
    :params dropnan: 是否丢弃含有NaN值的行, 类型为布尔值
    :return 经过重组后的Pandas DataFrame序列
    """
    cols, names = list(), list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(data.shape[1])]
        
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(data.shape[1])]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(data.shape[1])]
    # 将列名和数据拼接在一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 丢弃含有NaN值的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, n_in=24, n_out=24):
        if mode == "Train":
            self.data_dir = os.path.join(DATA_PATH, "bus30_train.csv")
        else:
            self.data_dir = os.path.join(DATA_PATH, "bus30_test.csv")
        self.df = pd.read_csv(self.data_dir, header=None).T

        self.data_list = np.zeros([self.df.shape[1], self.df.shape[0]-n_in-n_out+1, n_in+n_out])
        for line_num in range(self.df.shape[1]):
            temp = pd.DataFrame(self.df[line_num])
            temp = series_to_supervised(temp, n_in, n_out, dropnan=True)
            self.data_list[line_num, :, :] = np.array(temp)
            
        print(self.data_list.shape)

    def __len__(self):
        return self.data_list.shape[1]

    def __getitem__(self, idx):
        return self.data_list[:, idx, :]


def main():
    dataset = Dataset("Train")


if __name__ == "__main__":
    print('Start running: {}'.format(os.path.basename(__file__)))
    main()
