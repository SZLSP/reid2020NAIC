# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

import os

import torch
from torch.utils.data import DataLoader

from fastreid.data import samplers
from fastreid.data.build import fast_batch_collator
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms
from .common import FaceCommDataset

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_face_train_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)

    train_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root)
        # if comm.is_main_process():
        #    dataset.show_train()
        train_items.extend(dataset.train)

    train_set = FaceCommDataset(train_items, train_transforms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH

    data_sampler = samplers.TrainingSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )
    return train_loader


'''
def build_face_test_loader(cfg, dataset_name):
    test_transforms = build_transforms(cfg, is_train=False)
    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
    #if comm.is_main_process():
    #    dataset.show_test()
    test_items = dataset.query + dataset.gallery
    test_set = CommDataset(test_items, test_transforms, relabel=False)
    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=4,  # save some memory
        collate_fn=fast_batch_collator)
    return test_loader, len(dataset.query)
'''
