import os
import logging
import time

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from LineNet import LineNet
from Dataset import Dataset


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# const
BATCH_SIZE = 1
LINE_NUM = 29
MODEL_PATH = "model_reg_cls_alpha=10.pth"


def get_model(device):
    model_path = os.path.join(os.getcwd(), MODEL_PATH)
    logger.info("Load model from: {}".format(str(model_path)))
    model = LineNet(128).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device")

    test_dataset = Dataset("test")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)

    model = get_model(device)

    pred_result = []

    cls_result = []

    for batch, (x, y, l) in enumerate(tqdm(test_dataloader)):
        x = x.to(device).to(torch.float32)
        y = y.to(device).to(torch.float32)
        l = l.to(device).to(torch.float32)

        pred_cls, pred_reg = model(x, BATCH_SIZE, device)
        congestion_prob = pred_cls[0][LINE_NUM].item()
        cls_result.append(congestion_prob)
        pred_result.append(pred_reg)

    pred_line = []
    true_line = []

    loss = 0
    for idx in range(len(pred_result)):
        pred = pred_result[idx]
        pred_line_one = float(pred[0][LINE_NUM].item())
        true_line_one = float(test_dataset[idx][1][LINE_NUM])
        pred_line.append(pred_line_one)
        true_line.append(true_line_one)
    
    loss = mean_squared_error(true_line, pred_line)
    print(loss)
    
    print(len(pred_line))
    print(len(true_line))

    plt.figure(figsize=(18,8), dpi=200)
    plt.plot(cls_result)
    plt.savefig('plot_cls_result.png')

    plt.figure(figsize=(18,8), dpi=200)
    plt.plot(pred_line[-500:], label='pred')
    plt.plot(true_line[-500:], label='true')
    plt.legend(fontsize=20)
    plt.savefig('plot_pre_result.png')

if __name__ == "__main__":
    main()
