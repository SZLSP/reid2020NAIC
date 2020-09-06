from fastreid.layers import *


def get_classifier(cfg, cls_type, in_feat, num_classes, **kwargs):
    if cls_type == 'linear':
        classifier = nn.Linear(in_feat, num_classes, **kwargs)
    elif cls_type == 'arcSoftmax':
        classifier = ArcSoftmax(cfg, in_feat, num_classes)
    elif cls_type == 'circleSoftmax':
        classifier = CircleSoftmax(cfg, in_feat, num_classes)
    elif cls_type == 'amSoftmax':
        classifier = AMSoftmax(cfg, in_feat, num_classes)
    elif cls_type == 'arcFace':
        classifier = ArcFace(in_feat, num_classes, **kwargs)
    elif cls_type == 'arcCos':
        classifier = ArcCos(in_feat, num_classes, **kwargs)
    elif cls_type == 'adaCos':
        classifier = AdaCos(in_feat, num_classes, **kwargs)
    elif cls_type == 'cosFace':
        classifier = CosFace(in_feat, num_classes, **kwargs)
    elif cls_type == 'sphereFace':
        classifier = SphereFace(in_feat, num_classes, **kwargs)
    else:
        raise KeyError(f"{cls_type} is invalid, please choose from "
                       f"'linear', 'arcSoftmax', 'amSoftmax','circleSoftmax',"
                       f"'arcFace','arcCos','adaCos', 'cosFace' and 'sphereFace'.")
    return classifier
