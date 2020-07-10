# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

from torch.utils.data import Dataset

from fastreid.data.data_utils import read_image


class FaceCommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        self.pid_dict = {}
        if self.relabel:
            fids = list()
            for i, item in enumerate(img_items):
                if item[1] in fids: continue
                fids.append(item[1])
            self.fids = fids
            self.fid_dict = dict([(p, i) for i, p in enumerate(self.fids)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, fid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel: fid = self.fid_dict[fid]
        return {
            "images": img,
            "targets": fid,
            "img_path": img_path
        }

    @property
    def num_classes(self):
        return len(self.fids)
