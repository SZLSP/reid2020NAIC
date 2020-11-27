#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

"""
Modified from file "tools/train_net.py"
"""

import os
import sys
import torch

# sys.path.append('..')

from fastreid.modeling.backbones import *

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if (args.resume or args.eval_only) and len(cfg.MODEL.WEIGHTS) == 0:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_best.pth')

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def load_cfg():
    args =  default_argument_parser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print("Command Line Args:", args)

    cfg = setup(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = True
    cfg.freeze()

    return cfg

def load_model():
    cfg = load_cfg()

    meta_arch = cfg.MODEL.META_ARCHITECTURE
    buid_method = cfg.MODEL.BACKBONE.NAME
    name_to_method = {
        "build_resnest_backbone": build_resnest_backbone,
        "build_resnext_backbone": build_resnext_backbone
    }
    if buid_method not in name_to_method: raise Exception("Method %s not implemented" % buid_method)
    model = name_to_method[buid_method](cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return cfg, model



# if __name__ == "__main__":
#     args = default_argument_parser().parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
#     print("Command Line Args:", args)

# launch(
#     main,
#     args.num_gpus,
#     num_machines=args.num_machines,
#     machine_rank=args.machine_rank,
#     dist_url=args.dist_url,
#     args=(args,),
# )
