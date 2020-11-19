# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .autoaugment import AutoAugment
from .transforms import *


def build_transforms(cfg, is_train=True):
    res = []

    if is_train:

        size_train = cfg.INPUT.SIZE_TRAIN
        # Randomly alters the intensities of RGB channels
        do_coloraugment = cfg.INPUT.CJ.ENABLED
        ca_prob = cfg.INPUT.CA.PROB
        # Turn all image gray
        do_greyscale = cfg.INPUT.DO_GRAYSCALE

        # augmix augmentation
        do_augmix = cfg.INPUT.DO_AUGMIX

        # auto augmentation
        do_autoaug = cfg.INPUT.DO_AUTOAUG
        total_iter = cfg.SOLVER.MAX_ITER

        # horizontal filp
        do_flip = cfg.INPUT.DO_FLIP
        flip_prob = cfg.INPUT.FLIP_PROB

        # padding
        do_pad = cfg.INPUT.DO_PAD
        padding = cfg.INPUT.PADDING
        padding_mode = cfg.INPUT.PADDING_MODE

        # color jitter
        do_cj = cfg.INPUT.CJ.ENABLED
        cj_prob = cfg.INPUT.CJ.PROB
        cj_brightness = cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = cfg.INPUT.CJ.CONTRAST
        cj_saturation = cfg.INPUT.CJ.SATURATION
        cj_hue = cfg.INPUT.CJ.HUE

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_mean = cfg.INPUT.REA.MEAN
        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        # color transpose
        do_ct = cfg.INPUT.CT.ENABLED
        ct_color_offset = cfg.INPUT.CT.COLOR_OFFSET
        ct_invert = cfg.INPUT.CT.INVERT

        #Random2DTranslation
        do_trans2d = cfg.INPUT.T2.ENABLED


        if do_ct:
            res.append(ColorTranspose(ct_color_offset, ct_invert))

        if do_greyscale:
            res.append(GrayScale())

        if do_autoaug:
            res.append(AutoAugment(total_iter))
        res.append(T.Resize(size_train, interpolation=3))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)])
        if do_trans2d:
            res.append(Random2DTranslation(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]))

        if do_cj:
            res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_augmix:
            res.append(AugMix())
        if do_rea:
            res.append(RandomErasing(probability=rea_prob, mean=rea_mean))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))


    else:
        # Turn all image gray
        do_greyscale = cfg.INPUT.DO_GRAYSCALE

        if do_greyscale:
            res.append(GrayScale())
        size_test = cfg.INPUT.SIZE_TEST
        res.append(T.Resize(size_test, interpolation=3))
    res.append(ToTensor())
    if is_train:
        do_coloraugment = cfg.INPUT.CJ.ENABLED
        ca_prob = cfg.INPUT.CA.PROB
        if do_coloraugment:
            res.append(ColorAugmentation(ca_prob))
        # Lighting
        do_lighting = cfg.INPUT.LI.ENABLED
        alpha_std = cfg.INPUT.LI.ALPHA
        if do_lighting:
            res.append(Lighting(alphastd=alpha_std))

    return T.Compose(res)
