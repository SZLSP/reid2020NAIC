# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

from fastreid.layers import *
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY
from fastreid.modeling.losses import *
from fastreid.utils.weight_init import weights_init_classifier


@REID_HEADS_REGISTRY.register()
class FaceHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()

        self.output_layer = nn.Sequential(
            get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT),
            nn.Dropout2d(0.6, inplace=True),
            Flatten(),
            nn.Linear(in_feat * 8 * 8, 512),
            nn.BatchNorm1d(512),
        )

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':
            self.classifier = nn.Linear(512, num_classes, bias=False)
        elif cls_type == 'arcface':
            self.classifier = Arcface(cfg, 512, num_classes)
        elif cls_type == 'circle':
            self.classifier = Circle(cfg, 512, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcface' and 'circle'.")

        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        features = self.output_layer(features)

        # Evaluation
        if not self.training: return features

        # Training
        try:
            cls_outputs = self.classifier(features)
            pred_class_logits = cls_outputs.detach()
        except TypeError:
            cls_outputs = self.classifier(features, targets)
            pred_class_logits = F.linear(F.normalize(features.detach()), F.normalize(self.classifier.weight.detach()))

        CrossEntropyLoss.log_accuracy(pred_class_logits, targets)

        return cls_outputs, features
