# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .activation import *
from .am_softmax import AMSoftmax
from .arc_softmax import ArcSoftmax
from .ban import BAN2d
from .batch_drop import BatchDrop
from .batch_norm import *
from .circle_softmax import CircleSoftmax
from .context_block import ContextBlock
from .cosine_loss import ArcFace, ArcCos, AdaCos, CosFace, SphereFace
from .frn import FRN, TLU
from .non_local import Non_local
from .pooling import *
from .CausalNormClassifier import Causal_Norm_Classifier
# try:
#     from .rfconv import RFConv2d
# except:
#     warnings.warn('RFConv2d import failed')
from .se_layer import SELayer
from .splat import SplAtConv2d
