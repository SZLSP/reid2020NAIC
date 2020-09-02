import sys
sys.path.insert(0,'.')
import glob
import os
import os.path as osp
from collections import defaultdict

from .bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY
import numpy as np
import json

_NAIC_TRAIN_RATIO = 0.6
_NAIC_VAL_RATIO = 0.8
_NAIC_RANDOM_SEED = 2020
_NAIC_TESTING = False


@DATASET_REGISTRY.register()
class NAICReID(ImageDataset):
    def __init__(self, root='datasets', **kwargs):
        self.naic_root = osp.join(root, 'naic')
        required_files = [
            self.naic_root
        ]
        self.check_before_run(required_files)

        naic_json = osp.join(self.naic_root, 'naic.json')
        if not osp.exists(naic_json):
            self.preprocess(self.naic_root)

        with open(naic_json, 'r') as f:
            naic = json.load(f)
        train = naic['train']
        query = naic['val_query'] if not _NAIC_TESTING else naic['test_query']
        gallery = naic['val_gallery'] if not _NAIC_TESTING else naic['test_gallery']

        train = self.process_data(train,True)
        query = self.process_data(query)
        gallery = self.process_data(gallery)

        super(NAICReID, self).__init__(train, query, gallery, **kwargs)

    def process_data(self, data, is_train=False):
        new_data = []
        for identity in data:
            new_data.append(
                (
                    osp.join(self.naic_root,'train','images', identity[0]),
                    f'naic_{identity[1]}' if is_train else int(identity[1]),
                    int(identity[2])
                )
            )
        return new_data

    def preprocess(self, naic_root):

        label_dir = osp.join(naic_root, 'train', 'label.txt')
        np.random.seed(_NAIC_RANDOM_SEED)
        data = defaultdict(list)
        with open(label_dir, 'r') as f:
            for line in f.readlines():
                img_name, pid = line.strip().split(':')
                #                image_dir,   pid   ,    camid
                data[pid].append([img_name, int(pid), len(data[pid]) + 1])
        pids = sorted(list(data.keys()))
        np.random.shuffle(pids)
        total_id = len(pids)
        s1, s2 = int(total_id * _NAIC_TRAIN_RATIO), int(total_id * _NAIC_VAL_RATIO)
        train = pids[:s1]
        val = pids[s1:s2]
        test = pids[s2:]

        naic_json = defaultdict(list)
        for key in train:
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
        with open(osp.join(naic_root, 'naic.json'), 'w') as f:
            json.dump(naic_json, f)


@DATASET_REGISTRY.register()
class NAICSubmit(ImageDataset):
    def __init__(self, root='datasets', **kwargs):
        self.naic_root = osp.join(root, 'naic')
        required_files = [
            self.naic_root
        ]
        self.check_before_run(required_files)
        imgs_root = osp.join(self.naic_root,'image_A')
        train = []
        query = [(osp.join(imgs_root,'query',img_name),1,1) for img_name
                 in sorted(os.listdir(osp.join(imgs_root,'query')))]
        gallery = [(osp.join(imgs_root,'gallery',img_name),1,1) for img_name
                 in sorted(os.listdir(osp.join(imgs_root,'gallery')))]


        super(NAICSubmit, self).__init__(train, query, gallery, **kwargs)



if __name__ == '__main__':
    dataset = NAICSubmit()
