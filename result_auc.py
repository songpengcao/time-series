import os
import logging
import argparse
import torch
import numpy as np
from tqdm import tqdm

from Dataset import Dataset
from LineNet import LINE_NUM, LineNet


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# CONST
LINE_NUM_SET = [9, 28, 29, 34]
BATCH_SIZE = 1
THRESHOLD = 0.5

MODEL_PATH = "model_reg_cls_alpha=10.pth"


def get_model(args, device):
    model_path = os.path.join(os.getcwd(), args.model_name)
    logger.info("Load model from: {}".format(str(model_path)))
    model = LineNet(128).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def plot_roc_curve(fper, tper):
    plt.figure()
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig('ROC.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default=MODEL_PATH)
    parser.add_argument("--line_num",
                        type=int,
                        default=9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device")

    model = get_model(args, device)
    
    test_dataset = Dataset("test")
    test_label = np.loadtxt('./Data/test/label.csv', delimiter=',').T[24:]

    line = args.line_num
    line_label = []
    line_prob = []
    for i in tqdm(range(len(test_dataset))):
        x, _, _ = test_dataset[i]
        x = torch.Tensor(x)
        x = x.to(device).to(torch.float32).unsqueeze(0)
        y_cls, y_res= model(x, BATCH_SIZE, device)
        line_prob.append(y_cls[0][line].item())
        line_label.append(test_label[i][line])

    fpr, tpr, thresholds = roc_curve(line_label, line_prob)
    plot_roc_curve(fpr, tpr)
    auc_result = auc(fpr, tpr)
    print(auc_result)