#!/usr/bin/env python
# encoding: utf-8


import logging
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from fastreid.utils import comm
from collections import OrderedDict
from fastreid.data import build_reid_test_loader
from fastreid.evaluation import (ReidEvaluator,
                                 print_csv_format, extract_feature)
from fastreid.evaluation.query_expansion import aqe
from fastreid.evaluation.rerank import re_ranking
from fastreid.evaluation.fast_reranking import re_ranking as fast_re_ranking
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
    def __init__(self, cfg):
        super(NAICSubmiter, self).__init__(cfg)
        self.cached_info = {}

    def evaluation(self):
        logger = logging.getLogger(__name__)

        results = OrderedDict()

        for idx, dataset_name in enumerate(self.cfg.DATASETS.TESTS):
            logger.info("Prepare testing set")
            cached_info = self.cached_info.get(dataset_name)
            evaluator = None
            if cached_info is not None:
                data_loader, num_query, evaluator = cached_info
                if evaluator._num_query != num_query:
                    evaluator = None
                else:
                    evaluator = evaluator.recover()
            if evaluator is None:
                data_loader, num_query = build_reid_test_loader(self.cfg, dataset_name, use_testing=False)
                # When evaluators are passed in as arguments,
                # implicitly assume that evaluators can be created before data_loader.

                evaluator = ReidEvaluator(self.cfg, num_query)
                self.cached_info[dataset_name] = (data_loader, num_query, evaluator)
                extract_feature(self.model, data_loader, evaluator, self.cfg)
                evaluator.cached()

            results_i = evaluator.evaluate()
            results[dataset_name] = {}
            results[dataset_name] = results_i

            results_path = osp.join(self.cfg.OUTPUT_DIR, 'results.csv')
            score = 0.5 * (results[dataset_name]['Rank-1'] + results[dataset_name]['mAP@200'])
            results[dataset_name]['NAIC'] = score
            results[dataset_name]['AQE'] = self.cfg.TEST.AQE.ENABLED
            results[dataset_name]['METRIC'] = self.cfg.TEST.METRIC
            results[dataset_name]['RERANK'] = False # self.cfg.TEST.RERANK.ENABLED
            results[dataset_name]['ITERATION'] = self.iteration
            results[dataset_name]['FLIP_FEATS'] = self.cfg.TEST.FLIP_FEATS
            # results[dataset_name]['RERANK_K1'] = self.cfg.TEST.RERANK.K1
            # results[dataset_name]['RERANK_K2'] = self.cfg.TEST.RERANK.K2
            outputs = [dataset_name, score] + list(results[dataset_name].values())
            outputs = ','.join(list(map(str, outputs)))
            if not osp.exists(results_path):
                column = ['Datasets', 'NAIC'] + list(results[dataset_name].keys())
                column = ','.join(list(map(str, column)))
                with open(results_path, 'a') as f:
                    f.write(column + '\n' + outputs + '\n')
            else:
                with open(results_path, 'a') as f:
                    f.write(outputs + '\n')

        if comm.is_main_process():
            assert isinstance(
                results, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results
            )
            print_csv_format(results)

        return results

    def submit(self, use_dist=False, save_path=None, postfix=''):
        dataset_name = 'NAICSubmit'
        logger = logging.getLogger(__name__)

        logger.info("Prepare testing set")
        data_loader, num_query = build_reid_test_loader(self.cfg, dataset_name)
        # When evaluators are passed in as arguments,
        # implicitly assume that evaluators can be created before data_loader.
        img_paths = []
        features = []

        with inference_context(self.model), torch.no_grad():
            for idx, inputs in tqdm(enumerate(data_loader), total=len(data_loader)):
                if self.cfg.TEST.FLIP_FEATS == 'on':
                    in_feat = self.cfg.HEADS.REDUCTION_DIM \
                        if self.cfg.MODEL.HEADS.NAME == 'ReductionHead' else self.cfg.MODEL.HEADS.IN_FEAT
                    feat = torch.FloatTensor(inputs["images"].size(0), in_feat).zero_().cuda()
                    for i in range(2):
                        if i == 1:
                            inv_idx = torch.arange(inputs["images"].size(3) - 1, -1, -1).long()
                            inputs["images"] = inputs["images"].index_select(3, inv_idx)
                        f = self.model(inputs)
                        feat = feat + f
                else:
                    feat = self.model(inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                features.append(feat.cpu())
                img_paths.extend(inputs['img_path'])
        features = torch.cat(features, dim=0)
        img_paths = [osp.split(p)[-1] for p in img_paths]

        query_paths = img_paths[:num_query]
        query_features = features[:num_query]

        gallery_paths = img_paths[num_query:]
        gallery_features = features[num_query:]

        if self.cfg.TEST.AQE.ENABLED:
            postfix += '_aqe'
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        if self.cfg.TEST.METRIC == "cosine":
            postfix += '_cos'
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)
        else:
            postfix += '_l2'

        dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)

        if self.cfg.TEST.RERANK.ENABLED:
            postfix += '_rerank'
            use_dist = True
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)

            if self.cfg.TEST.RERANK.FAST:
                dist = fast_re_ranking(query_features, gallery_features, k1, k2, lambda_value)
            else:
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
        for i, key in enumerate(query_paths):
            submit[key] = gallery_paths[indices[i, :200]].tolist()

        if save_path is None: save_path = self.cfg.OUTPUT_DIR
        cfg_name = osp.split(args.config_file)[-1][:-4]
        with open(osp.join(save_path, f'{cfg_name}_it{self.iteration}_{postfix}.json'), 'w') as f:
            json.dump(submit, f)
        img_order = {
            'query_paths': list(query_paths),
            'gallery_paths': list(gallery_paths)
        }
        with open(osp.join(save_path, f'{cfg_name}_it{self.iteration}_{postfix}_order.json'), 'w') as f:
            json.dump(img_order, f)
        np.save(osp.join(save_path, f'{cfg_name}_it{self.iteration}_{postfix}.npy'), np.asarray(dist))

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


def get_score(results):
    score = 0
    n = 0
    for res in results.values():
        score += 0.5 * (res['Rank-1'] + res['mAP@200'])
        n += 1
    return score / n


def main(args):
    cfg = setup(args)
    if len(args.test_iter) > 0:
        cfg.defrost()
        iterations = args.test_iter.split(',')
        iterations = [int(it) if it.isdigit() else it for it in iterations]
        cfg.TEST.ITERATIONS = iterations
        cfg.freeze()

    if len(args.test_sets) > 0:
        cfg.defrost()
        cfg.DATASETS.TESTS = tuple(args.test_sets.split(','))
        cfg.freeze()
    submiter = NAICSubmiter(cfg)
    if args.test_permutation:
        best_test_score = 0
        best_hyperparameter = None
        flip_feat = 'off'
        for aqe in [False, True]:
            # for flip_feat in ['on', 'off']:
            # for k1 in range(8,41,4):
            #     for k2 in range(2,9,1):
            #         aqe = True
            metric = 'cosine'
            rerank = False
            submiter.cfg.defrost()
            submiter.cfg.TEST.AQE.ENABLED = aqe
            submiter.cfg.TEST.METRIC = metric
            submiter.cfg.TEST.RERANK.ENABLED = rerank
            submiter.cfg.TEST.FLIP_FEATS = flip_feat
            # submiter.cfg.TEST.RERANK.K1 = k1
            # submiter.cfg.TEST.RERANK.K2 = k2
            submiter.cfg.freeze()
            res = submiter.evaluation()
            score = get_score(res)
            if score > best_test_score:
                best_test_score = score
                best_hyperparameter = [aqe, metric, rerank, flip_feat]

        submiter.cfg.defrost()
        submiter.cfg.TEST.AQE.ENABLED = best_hyperparameter[0]
        submiter.cfg.TEST.METRIC = best_hyperparameter[1]
        submiter.cfg.TEST.RERANK.ENABLED = best_hyperparameter[2]
        submiter.cfg.TEST.FLIP_FEATS = best_hyperparameter[3]
        submiter.cfg.freeze()
        postfix = f'{best_test_score:.4f}'[2:]
        submiter.submit(postfix=postfix)
    else:
        res = submiter.evaluation()
        postfix = get_score(res)
        postfix = f'{postfix:.4f}'[2:]
        submiter.submit(postfix=postfix)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
