
#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def ion():
    plt.ion()


def ioff():
    plt.ioff()
    plt.show()


def plot_heatmap(s_dim, a_dim, table):
    if s_dim > a_dim:
        x_dim = s_dim
        y_dim = a_dim
        table = table.T
        title = "Q-Table x: s, y: a"
    else:
        x_dim = a_dim
        y_dim = s_dim
        title = "Q-Table x: a, y: s"

    x = np.arange(x_dim)
    y = np.arange(y_dim)

    fig, ax = plt.subplots()
    im = ax.imshow(table)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('colorbar', rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(x)
    ax.set_yticks(y)
    # ... and label them with the respective list entries
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title(title)
    fig.tight_layout()
    plt.pause(0.0001)
    plt.close()


def show_np_img(data: np.ndarray):
    plt.imshow(data)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
