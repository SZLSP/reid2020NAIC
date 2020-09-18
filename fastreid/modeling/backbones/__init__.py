# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import build_backbone, BACKBONE_REGISTRY
from .efficientnet_pytorch import build_efficientnet_backbone
from .osnet import build_osnet_backbone
from .regnet import build_regnet_backbone
from .resnest import build_resnest_backbone
from .resnest_in import build_resnest_in_backbone
from .resnet import build_resnet_backbone
from .resnext import build_resnext_backbone
