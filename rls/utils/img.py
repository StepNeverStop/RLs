#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from PIL import Image
from time import sleep


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
