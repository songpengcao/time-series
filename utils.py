from matplotlib import pyplot as plt
import numpy as np

def plot_attention_score(atten_score, name, title):
    atten_score = np.concatenate(
        (np.zeros((atten_score.shape[0], 1)), atten_score), axis=1)
    atten_score = np.concatenate(
        (np.zeros((1, atten_score.shape[1])), atten_score), axis=0)
    x_len, y_len = atten_score.shape
    fig = plt.figure(figsize=(12, 12), dpi=75)
    ax = fig.add_subplot(111)
    ax.set_xticks(range(0, y_len+1, 5))
    ax.set_yticks(range(0, x_len+1, 5))
    ax.set_xlabel("Bus Number")
    ax.set_ylabel("Line Number")
    ax.tick_params(labeltop=False, labelbottom=True, labelleft=True, labelright=False)
    im = ax.imshow(atten_score, origin='lower', cmap=plt.cm.hot_r)
    # im = ax.matshow(atten_score, cmap='ocean')
    if name == 2:
        plt.colorbar(im)
    # plt.title(title)
    plt.savefig('plt_atten_score_{}.png'.format(name))
    plt.close()

