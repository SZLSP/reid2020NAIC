# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

import os

from fastreid.data.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CASIAWebFace(object):
    dataset_dir = 'CASIA-WebFace'
    dataset_name = "casia_webface"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.file_list = os.path.join(self.dataset_dir, 'casiawebface_age_id_label.txt')
        required_files = [
            self.dataset_dir
        ]
        # check_before_run(required_files)
        train_list = []
        with open(self.file_list) as f:
            img_label_list = f.read().splitlines()

        for info in img_label_list:
            foldername, filename, age, fid = info.split(' ')
            fid = self.dataset_name + '_' + fid
            train_list.append((os.path.join(root, self.dataset_dir, foldername, filename), fid))

        self.train = train_list
