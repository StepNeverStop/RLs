# !/usr/bin/env python3
# encoding: utf-8

from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_np_img(data: np.ndarray):
    plt.imshow(data)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()


def show_img(img_arr, normalized=True):
    img_arr = np.array(img_arr)
    if normalized:
        img_arr *= 255
    im = Image.fromarray(np.uint8(img_arr))
    im.show()
    sleep(0.5)


if __name__ == "__main__":
    img = np.random.uniform(size=[800, 800, 3])
    show_img(img)
