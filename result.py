import os
import logging
import argparse
import torch
import numpy as np
from tqdm import tqdm

from Dataset import Dataset
from LineNet import LineNet

from sklearn.metrics import mean_squared_error

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# CONST
LINE_NUM_SET = [9, 28, 29, 34]
BATCH_SIZE = 1


def get_model(args, device):
    model_path = os.path.join(os.getcwd(), args.model_name)
    logger.info("Load model from: {}".format(str(model_path)))
    model = LineNet(128).to(device)
    print(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def compute_multi_error(i, model, args, visualize=False):
    horizon = args.horizon
    now_error = np.zeros([horizon, 41])
    congestion_status = np.zeros([horizon, 41])
    congestion_label = np.zeros([horizon, 41])
    congestion_result = np.zeros([horizon, 41])

    temp_input, _, _ = test_dataset[i] # np.ndarray (24, 71)
    for j in range(horizon):
        x = temp_input.copy()
        x = torch.Tensor(x)
        x = x.to(device).to(torch.float32).unsqueeze(0)

        pred_cls, pred_reg = model(x, BATCH_SIZE, device)
        # pred_cls is (1, 41) tensor on cuda
        # pred_reg is (1, 41) tensor on cuda
        for line_num in LINE_NUM_SET:
            line_predict = pred_reg[0][line_num].item()
            congestion_prob = pred_cls[0][line_num].item()
            # print(line_predict, congestion_prob)
            if congestion_prob >= args.threshold:
                pre_label = 1
            else:
                pre_label = 0

            # get ground_truth
            _, y, _ = test_dataset[i+j]
            gt_value = y[line_num]
            gt_label = test_label[i+j][line_num]
            # print(gt_value, gt_label)

            # get result
            congestion_status[j][line_num] = pre_label
            congestion_label[j][line_num] = gt_label
            congestion_result[j][line_num] = 0 if pre_label == gt_label else 1
            now_error[j][line_num] = mean_squared_error([line_predict], [gt_value])

        # next_data, _, _ = test_dataset[i+1]  # np.ndarray (24, 71)  raw
        next_data, _, _ = test_dataset[i+j]  # np.ndarray (24, 71)  new TODO
        temp_input = next_data.copy()
        temp_input[-1, 30:] = pred_reg.cpu().detach().numpy()[0]

    for line_num in LINE_NUM_SET:
        # compute inner recall
        if np.array(congestion_label[:, line_num]).sum() > 0:
            m = 0
            n = 0
            for j in range(horizon):
                if congestion_label[j][line_num] == 1:
                    m += 1
                    if congestion_status[j][line_num] == 1:
                        n += 1
            all_recall[i][line_num] = n / m
            all_recall_count[line_num] += 1
    
        # compute inner precision
        m = 0
        n = 0
        for j in range(horizon):
            if congestion_status[j][line_num] == 1:
                m += 1
                if congestion_label[j][line_num] == 1:
                    n += 1
        if m != 0:
            all_precision[i][line_num] = n / m
            all_precision_count[line_num] += 1

        inner_acc = (horizon - np.array(congestion_result[:, line_num]).sum()) / horizon
        acc_count[line_num] = acc_count[line_num] + 1 if inner_acc == 1 else acc_count[line_num]
        all_error[i][line_num] = now_error[:, line_num].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default=None)
    parser.add_argument("--horizon",
                        type=int,
                        default=12)
    parser.add_argument("--threshold",
                        type=float,
                        default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device")

    model = get_model(args, device)
    
    test_dataset = Dataset("test")
    test_label = np.loadtxt('./Data/test/label.csv', delimiter=',').T[24:]

    logger.info("Horizon: {}".format(args.horizon))

    SAMPLE_LEN = 8760 - args.horizon

    all_recall = np.zeros([SAMPLE_LEN, 41])
    all_recall_count = np.zeros(41)
    all_precision = np.zeros([SAMPLE_LEN, 41])
    all_precision_count = np.zeros(41)
    all_error = np.zeros([SAMPLE_LEN, 41])
    acc_count = np.zeros(41)

    # for i in tqdm(range(SAMPLE_LEN)):
    #     compute_multi_error(i, model, args)
    compute_multi_error(490, model, args)

    for line_num in LINE_NUM_SET:
        recall_score = all_recall[:, line_num].sum() / all_recall_count[line_num]
        precision_score = all_precision[:, line_num].sum() / all_precision_count[line_num]
        acc_rate = acc_count[line_num] / SAMPLE_LEN
        error_score = all_error[:, line_num].mean()

        print('line: {}, [error] -- {}'.format(line_num, round(error_score, 4)))
        print('line: {}. [all_recall] -- {}'.format(line_num, round(recall_score, 4)))
        print('line: {}, [all_precision] -- {}'.format(line_num, round(precision_score, 4)))
        print('line: {}, [acc_count] -- {}, [acc_rate] -- {}'.format(line_num, acc_count[line_num], round(acc_rate, 4)))
        print()
