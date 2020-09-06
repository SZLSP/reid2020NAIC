# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import os
import os.path as osp

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from fastreid.utils.file_io import PathManager


def edge_detection(img_root, old='naic', _new='edge'):
    for dirpath, dirnames, filenames in os.walk(img_root):
        for fn in tqdm(filenames):
            if fn[-3:] in {'png', 'PNG', 'jpg', 'JPG', 'JPEG'}:
                new_dir = dirpath.replace(old, _new)
                os.makedirs(new_dir, exist_ok=True)
                img = cv2.imread(osp.join(dirpath, fn), 0)
                x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
                y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
                absX = cv2.convertScaleAbs(x)
                absY = cv2.convertScaleAbs(y)

                dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
                cv2.imwrite(osp.join(new_dir, fn), dst)


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.
    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        image = Image.fromarray(image)
        return image
