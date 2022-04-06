from matplotlib import pyplot as plt

def plot_attention_score(atten_score, name):
    x_len, y_len = atten_score.shape
    fig = plt.figure(figsize=(20, 20), dpi=75)
    ax = fig.add_subplot(111)
    ax.set_xticks(range(y_len))
    ax.set_yticks(range(x_len))
    im = ax.imshow(atten_score, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title('title')
    plt.savefig('plt_atten_score_{}.png'.format(name))

