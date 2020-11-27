import sys

sys.path.insert(0, './')
import os
import os.path as osp
from collections import defaultdict

from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY
import numpy as np
import json

_NAIC_TRAIN_RATIO = 8.0
_NAIC_VAL_RATIO = 9.0
_NAIC_RANDOM_SEED = 2020
_NAIC_TESTING = False
_NAIC_MIN_INSTANCE = 1


@DATASET_REGISTRY.register()
class NAICReID(ImageDataset):

    def __init__(self, root='datasets', **kwargs):
        global _NAIC_TESTING, _NAIC_TRAIN_RATIO, _NAIC_VAL_RATIO, _NAIC_MIN_INSTANCE
        _NAIC_TESTING = (kwargs.get('use_testing') or _NAIC_TESTING)
        _NAIC_TRAIN_RATIO = (kwargs.get('train_ratio') or _NAIC_TRAIN_RATIO)
        _NAIC_VAL_RATIO = kwargs.get('val_ratio') or _NAIC_VAL_RATIO
        _NAIC_MIN_INSTANCE = kwargs.get('min_instance') or _NAIC_MIN_INSTANCE
        self.json_name = f'naic_{int(_NAIC_TRAIN_RATIO):02d}{int(_NAIC_VAL_RATIO):02d}{int(_NAIC_MIN_INSTANCE):02d}.json'
        _NAIC_TRAIN_RATIO /= 10.0
        _NAIC_VAL_RATIO /= 10.0
        self.naic_root, self.img_root, self.label_dir, self.splitor, self.train_prefix = self.get_datainfo(root)
        required_files = [
            self.naic_root
        ]
        self.check_before_run(required_files)

        naic_json = osp.join(self.naic_root, self.json_name)
        if not osp.exists(naic_json):
            self.preprocess(self.naic_root)

        with open(naic_json, 'r') as f:
            naic = json.load(f)
        train = naic['train']
        query = naic['val_query'] if not _NAIC_TESTING else naic['test_query']
        gallery = naic['val_gallery'] if not _NAIC_TESTING else naic['test_gallery']

        train = self.process_data(train, True)
        query = self.process_data(query)
        gallery = self.process_data(gallery)

        super(NAICReID, self).__init__(train, query, gallery, **kwargs)

    def process_data(self, data, is_train=False):
        new_data = []
        for identity in data:
            new_data.append(
                (
                    osp.join(self.img_root, identity[0]),
                    f'{self.train_prefix}_{identity[1]}' if is_train else int(identity[1]),
                    int(identity[2])
                )
            )
        return new_data

    def get_datainfo(self, root):
        naic_root = osp.join(root, 'naic')
        img_root = osp.join(root, 'naic', 'train', 'images')
        label_dir = osp.join(root, 'naic', 'train', 'label.txt')
        spiltor = ':'
        train_prefix = 'naic'
        return naic_root, img_root, label_dir, spiltor, train_prefix

    def preprocess(self, naic_root):

        np.random.seed(_NAIC_RANDOM_SEED)
        data = defaultdict(list)
        with open(self.label_dir, 'r') as f:
            for line in f.readlines():
                img_name, pid = line.strip().split(self.splitor)
                img_name = osp.split(img_name)[-1]
                #                image_dir,   pid   ,    camid
                data[pid].append([img_name, int(pid), len(data[pid]) + 1])
        pids = sorted(list(data.keys()))
        np.random.shuffle(pids)
        total_id = len(pids)
        assert 0 < _NAIC_TRAIN_RATIO <= _NAIC_VAL_RATIO <= 1
        s1, s2 = int(total_id * _NAIC_TRAIN_RATIO), int(total_id * _NAIC_VAL_RATIO)
        train = pids[:s1]
        val = pids[s1:s2]
        test = pids[s2:]

        naic_json = defaultdict(list)
        for key in train:
            if len(data[key]) >= _NAIC_MIN_INSTANCE:
                naic_json['train'].extend(data[key])

        def split_query_gallery(keys):
            query = []
            gallery = []
            for key in keys:
                identity = data[key]
                query_idx = np.random.randint(0, len(identity))
                query.append((identity.pop(query_idx)))
                gallery.extend(identity)
            return query, gallery

        naic_json['val_query'], naic_json['val_gallery'] = split_query_gallery(val)
        naic_json['test_query'], naic_json['test_gallery'] = split_query_gallery(test)
        with open(osp.join(naic_root, self.json_name), 'w') as f:
            json.dump(naic_json, f)


@DATASET_REGISTRY.register()
class NAIC19_PRE(NAICReID):
    def get_datainfo(self, root):
        naic_root = osp.join(root, 'naic19_pre')
        img_root = osp.join(naic_root, 'train_set')
        label_dir = osp.join(naic_root, 'train_list.txt')
        spiltor = ' '
        train_prefix = 'naic19pre'
        return naic_root, img_root, label_dir, spiltor, train_prefix


@DATASET_REGISTRY.register()
class NAIC19_REP(NAICReID):
    def get_datainfo(self, root):
        naic_root = osp.join(root, 'naic19_rep')
        img_root = osp.join(naic_root, 'train_set')
        label_dir = osp.join(naic_root, 'train_list (1).txt')
        spiltor = ' '
        train_prefix = 'naic19rep'
        return naic_root, img_root, label_dir, spiltor, train_prefix

@DATASET_REGISTRY.register()
class NAICReID_REP(NAICReID):
    def get_datainfo(self, root):
        naic_root = osp.join(root, 'naic_rep')
        naic_root = osp.join(naic_root, 'train')
        img_root = osp.join(naic_root, 'images')
        label_dir = osp.join(naic_root, 'label.txt')
        spiltor = ':'
        train_prefix = 'naicrep'
        return naic_root, img_root, label_dir, spiltor, train_prefix

@DATASET_REGISTRY.register()
class NAICSubmit(ImageDataset):
    def __init__(self, root='datasets', **kwargs):
        self.naic_root = osp.join(root, 'naic_rep')
        required_files = [
            self.naic_root
        ]
        self.check_before_run(required_files)
        imgs_root = osp.join(self.naic_root, 'image_B_v1.1')
        train = []
        query = [(osp.join(imgs_root, 'query', img_name), 1, 1) for img_name
                 in sorted(os.listdir(osp.join(imgs_root, 'query')))]
        gallery = [(osp.join(imgs_root, 'gallery', img_name), 1, 1) for img_name
                   in sorted(os.listdir(osp.join(imgs_root, 'gallery')))]

        super(NAICSubmit, self).__init__(train, query, gallery, **kwargs)


@DATASET_REGISTRY.register()
class NAIC19Test(ImageDataset):

    def __init__(self, root='datasets', **kwargs):
        global _NAIC_TESTING
        _NAIC_TESTING = kwargs.get('use_testing') or _NAIC_TESTING

        self.naic_root, self.img_root, self.label_dir, self.splitor, self.train_prefix = self.get_datainfo(root)
        required_files = [
            self.naic_root
        ]
        self.check_before_run(required_files)

        naic_json = osp.join(self.naic_root, 'naic19test.json')
        if not osp.exists(naic_json):
            self.preprocess(self.naic_root)

        with open(naic_json, 'r') as f:
            naic = json.load(f)
        train = naic['train']
        query = naic['val_query'] if not _NAIC_TESTING else naic['test_query']
        gallery = naic['val_gallery'] if not _NAIC_TESTING else naic['test_gallery']

        train = self.process_data(train, True)
        query = self.process_data(query)
        gallery = self.process_data(gallery)

        super(NAIC19Test, self).__init__(train, query, gallery, **kwargs)

    def process_data(self, data, is_train=False):
        new_data = []
        for identity in data:
            new_data.append(
                (
                    osp.join(self.img_root, identity[0]),
                    f'{self.train_prefix}_{identity[1]}' if is_train else int(identity[1]),
                    int(identity[2])
                )
            )
        return new_data

    def get_datainfo(self, root):
        naic_root = osp.join(root, 'naic19_rep')
        img_root = osp.join(naic_root, 'train_set')
        label_dir = osp.join(naic_root, 'train_list (1).txt')
        spiltor = ' '
        train_prefix = 'naic19rep'
        return naic_root, img_root, label_dir, spiltor, train_prefix

    def preprocess(self, naic_root):

        np.random.seed(_NAIC_RANDOM_SEED)
        data = defaultdict(list)
        with open(self.label_dir, 'r') as f:
            for line in f.readlines():
                img_name, pid = line.strip().split(self.splitor)
                img_name = osp.split(img_name)[-1]
                #                image_dir,   pid   ,    camid
                data[pid].append([img_name, int(pid), len(data[pid]) + 1])
        pids = sorted(list(filter(lambda k: len(data[k]) >= 2, data.keys())))
        np.random.shuffle(pids)
        total_id = len(pids)
        s1, s2 = total_id - 2900 * 2, total_id - 2900
        train = pids[:s1]
        val = pids[s1:s2]
        test = pids[s2:]

        naic_json = defaultdict(list)
        for key in train:
            if len(data[key]) >= _NAIC_MIN_INSTANCE:
                naic_json['train'].extend(data[key])

        def split_query_gallery(keys):
            query = []
            gallery = []
            for key in keys:
                identity = data[key]
                query_idx = np.random.randint(0, len(identity))
                query.append((identity.pop(query_idx)))
                gallery.extend(identity)
            return query, gallery

        naic_json['val_query'], naic_json['val_gallery'] = split_query_gallery(val)
        naic_json['test_query'], naic_json['test_gallery'] = split_query_gallery(test)

        with open(osp.join(naic_root, 'naic19test.json'), 'w') as f:
            json.dump(naic_json, f)
