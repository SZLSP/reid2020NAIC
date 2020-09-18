import json
import os
from collections import OrderedDict

import numpy as np

dist_path = [
    '/home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2/R101-ibn-arcF-cj-st-amp-gem-all-ranger2_it0225599_6558_aqe_cos_rerank.npy',
    '/home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-iebn2/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-iebn2_it0225599_6643_cos_rerank.npy',
    '/home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-ntl/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-ntl_it0115199_0119999_0110399_6512_aqe_cos_rerank.npy']
output_path = '/home/dgy/project/pytorch/fast-reid/logs/NAIC/ensemble'

img_order_path = [path[:-4] + '_order.json' for path in dist_path]
img_order = []
for path in img_order_path[1:]:
    with open(path, 'r') as f:
        order_dict = json.load(f)
        img_order.append(order_dict)

query_paths = img_order[0]['query_paths']
gallery_paths = img_order[0]['gallery_paths']
for order_dict in img_order[1:]:
    assert order_dict['query_paths'] == query_paths and order_dict['gallery_paths'] == gallery_paths

dist_matrices = [np.load(path) for path in dist_path]
dist_matrices = sum(dist_matrices)
indices = np.argsort(dist_matrices, axis=1)

gallery_paths = np.asarray(gallery_paths)
submit = OrderedDict()
for i, key in enumerate(query_paths):
    submit[key] = gallery_paths[indices[i, :200]].tolist()

saved_name = '_'.join([os.path.split(path)[-1][:-4] for path in dist_path])
with open(os.path.join(output_path, saved_name + '.json'), 'w') as f:
    json.dump(submit, f)
