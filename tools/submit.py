#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils import comm
from collections import OrderedDict
from fastreid.data import build_reid_test_loader, build_reid_train_loader
from fastreid.evaluation import (DatasetEvaluator, ReidEvaluator,
                                 inference_on_dataset, print_csv_format)
from fastreid.evaluation.query_expansion import aqe
from fastreid.evaluation.rerank import re_ranking
from fastreid.evaluation import inference_context
import faiss
import numpy as np
import os
import os.path as osp
import torch
from torch.functional import F
import json
from tqdm import tqdm

class NAICSubmiter(DefaultPredictor):
    def __init__(self,cfg):
        super(NAICSubmiter,self).__init__(cfg)

    def evaluation(self):
        dataset_name = 'NAICReID'
        logger = logging.getLogger(__name__)

        results = OrderedDict()

        logger.info("Prepare testing set")
        data_loader, num_query = build_reid_test_loader(self.cfg, dataset_name)
        # When evaluators are passed in as arguments,
        # implicitly assume that evaluators can be created before data_loader.

        evaluator = ReidEvaluator(self.cfg, num_query)

        results[dataset_name] = {}
        results_i = inference_on_dataset(self.model, data_loader, evaluator)
        results[dataset_name] = results_i

        if comm.is_main_process():
            assert isinstance(
                results, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results
            )
            print_csv_format(results)

            if len(results) == 1: results = list(results.values())[0]

            return results

    def submit(self,use_dist=False,save_path=None):
        dataset_name = 'NAICSubmit'
        logger = logging.getLogger(__name__)


        logger.info("Prepare testing set")
        data_loader, num_query = build_reid_test_loader(self.cfg, dataset_name)
        # When evaluators are passed in as arguments,
        # implicitly assume that evaluators can be created before data_loader.
        img_paths = []
        features = []

        with inference_context(self.model),torch.no_grad():
            for idx, inputs in tqdm(enumerate(data_loader),total=len(data_loader)):
                outputs = self.model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                features.append(outputs.cpu())
                img_paths.extend(inputs['img_path'])
        features = torch.cat(features, dim=0)
        img_paths = [osp.split(p)[-1] for p in img_paths]

        query_paths = img_paths[:num_query]
        query_features = features[:num_query]

        gallery_paths = img_paths[num_query:]
        gallery_features = features[num_query:]

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
            dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)

        query_features = query_features.numpy()
        gallery_features = gallery_features.numpy()


        if use_dist:
            indices = np.argsort(dist, axis=1)
        else:
            dim = query_features.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(gallery_features)
            _, indices = index.search(query_features, k=200)

        gallery_paths = np.asarray(gallery_paths)
        submit = OrderedDict()
        for i,key in enumerate(query_paths):
            submit[key] = gallery_paths[indices[i,:200]].tolist()

        if save_path is None: save_path = self.cfg.OUTPUT_DIR
        with open(osp.join(save_path,'submit.json'),'w') as f:
            json.dump(submit,f)





    @staticmethod
    def cal_dist(metric: str, query_feat: torch.tensor, gallery_feat: torch.tensor):
        assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
        if metric == "cosine":
            dist = 1 - torch.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(query_feat, gallery_feat.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.cpu().numpy()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    submiter = NAICSubmiter(cfg)
    # submiter.evaluation()
    submiter.submit()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )