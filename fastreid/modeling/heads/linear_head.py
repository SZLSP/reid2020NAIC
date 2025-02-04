# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_classifier
from .build import REID_HEADS_REGISTRY
from .classsifiers import get_classifier


@REID_HEADS_REGISTRY.register()
class LinearHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self.pool_layer = pool_layer

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':
            self.classifier = get_classifier(cfg, cls_type, in_feat, num_classes, bias=False)
        else:
            self.classifier = get_classifier(cfg, cls_type, in_feat, num_classes)

        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        global_feat = global_feat[..., 0, 0]

        # Evaluation
        if not self.training: return global_feat

        # Training
        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(global_feat)
        else:
            cls_outputs = self.classifier(global_feat, targets)


        pred_class_logits = F.linear(global_feat, self.classifier.weight)

        return cls_outputs, pred_class_logits, global_feat
