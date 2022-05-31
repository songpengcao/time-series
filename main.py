import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random

from Dataset import Dataset
from LineNet import HIDDEN_SIZE, LineNet


writer = SummaryWriter()

logging.basicConfig(level=logging.INFO,
                    filename='output.log', 
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

# const
BATCH_SIZE = 32
EPOCH_NUM = 10
ALPHA = 10
MODEL_NAME = "LineNet"

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def train(dataloader, iter_num, model, device, reg_loss_fn, cls_loss_fn, optimizer):
    model.train()
    for batch, (x, y, l) in enumerate(tqdm(dataloader)):
        x = x.to(device).to(torch.float32)  # (x.shape) == (32, 24, 71)
        y = y.to(device).to(torch.float32)  # (y.shape) == (32, 41)
        l = l.to(device).to(torch.float32)  # (l.shape) == (32, 41)
        
        pred_cls, pred_reg = model(x, BATCH_SIZE, device) # (pred.shape) == (32, 41)
        reg_loss = reg_loss_fn(pred_reg, y)
        cls_loss = cls_loss_fn(pred_cls, l)
        loss = reg_loss + cls_loss * ALPHA
        # loss = cls_loss
        writer.add_scalar('Loss/train', loss, iter_num * 546 + batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, device, reg_loss_fn, cls_loss_fn):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for batch, (x, y, l) in enumerate(tqdm(dataloader)):
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32)
            l = l.to(device).to(torch.float32)
            
            pred_cls, pred_reg = model(x, BATCH_SIZE, device)
            reg_loss = reg_loss_fn(pred_reg, y)
            cls_loss = cls_loss_fn(pred_cls, l)
            test_loss += reg_loss
    logger.info(test_loss)
    test_loss /= len(dataloader)
    logger.info(f"Avg Reg loss: {test_loss:>8f} \n")
    print(f"Avg Reg loss: {test_loss:>8f} \n")


def main():
    setup_seed(888)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device")

    # load data
    train_dataset = Dataset("train")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    test_dataset = Dataset("test")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)

    # build model
    linenet = LineNet(HIDDEN_SIZE).to(device)
    print(linenet)
    # design loss function
    reg_loss_fn = nn.SmoothL1Loss(reduction='mean')
    cls_loss_fn = nn.BCELoss(reduction='mean')

    raw_lr = 1e-5

    for epoch in range(EPOCH_NUM):
        # learning_rate decay
        # if t % 5 == 0:
        #     raw_lr = raw_lr * 0.9
        logger.info(f"Epoch {epoch+1}\n-------------------------------")
        print(f"Epoch {epoch+1}\n-------------------------------")
        optimizer = torch.optim.Adam(linenet.parameters(), lr=raw_lr)
        train(train_dataloader, epoch, linenet, device, reg_loss_fn, cls_loss_fn, optimizer)
        test(test_dataloader, linenet, device, reg_loss_fn, cls_loss_fn)
    logger.info("Done!")

    # save model
    torch.save(linenet.state_dict(), "{}_model_reg_cls_alpha={}.pth".format(MODEL_NAME, ALPHA))

    writer.flush()
    writer.close()

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)