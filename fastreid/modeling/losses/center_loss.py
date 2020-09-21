from __future__ import absolute_import

import torch
from torch import nn

from fastreid.config.defaults import _C


def softmax_weights(dist, mask):
    min_v = torch.min(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - min_v

    numerator = torch.exp(diff) * mask
    numerator[mask == 0] = 0
    Z = torch.sum(numerator, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = numerator / Z
    return W


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, cfg):
        super(CenterLoss, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.feat_dim = self.cfg.HEADS.REDUCTION_DIM \
            if self.cfg.MODEL.HEADS.NAME == 'ReductionHead' else self.cfg.MODEL.HEADS.IN_FEAT
        self._scale = cfg.MODEL.LOSSES.Center.SCALE
        self.alpha = cfg.MODEL.LOSSES.Center.ALPHA
        self.beta = cfg.MODEL.LOSSES.Center.BETA
        self.hard_mining = cfg.MODEL.LOSSES.Center.HARD_MINING
        self.margin = cfg.MODEL.LOSSES.Center.MARGIN
        # self.use_gpu = torch.cuda.is_available()
        self.register_buffer('centers', torch.randn(self.num_classes, self.feat_dim))

        # if self.use_gpu:
        #
        #         nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        # else:
        #     self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, features, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = features.size(0)
        labels = labels.long()
        centers = self.centers[labels].detach()

        delta1 = (centers - features)
        center_diff = centers.unsqueeze(1) - centers.unsqueeze(0)
        mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
        center_dist = torch.norm(center_diff, dim=2)
        mask = mask * (center_dist <= self.margin).float()
        if mask.sum() < 1:
            delta2 = torch.zeros_like(centers)
        else:
            if self.hard_mining:
                min_dist = torch.min(center_dist + (1 - mask) * 1e10)
                mask = center_dist.isclose(min_dist).float() * mask
            weight = softmax_weights(-center_dist, mask)
            delta2 = (center_diff * weight.unsqueeze(2)).sum(dim=1)

        dist = torch.square(centers - features)
        if torch.any(torch.isnan(delta2)) or torch.any(torch.isnan(delta1)):
            print()
        self.centers[labels] = centers - self.alpha * delta1 - self.beta * delta2  # update center

        # dist = torch.pow(x, 2).sum(dim=1) + \
        #        torch.pow(centers, 2).sum(dim=1) - \
        #        2 * (x * centers).sum(dim=1)

        loss = dist.clamp(min=1e-12, max=1e+12).mean()
        if torch.isnan(loss):
            print()

        return loss * self._scale


class CenterLoss_old(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, cfg):
        super(CenterLoss_old, self).__init__()
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.feat_dim = cfg.MODEL.HEADS.IN_FEAT
        self._scale = cfg.MODEL.LOSSES.Center.SCALE
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.centers.data = self.centers.data / torch.norm(self.centers.data, p=2, dim=-1, keepdim=True)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss * self._scale


if __name__ == '__main__':
    _C.defrost()
    _C.MODEL.HEADS.NUM_CLASSES = 6
    use_gpu = False
    center_loss = CenterLoss(_C)
    center_loss2 = CenterLoss_old(_C)
    center_loss2.centers = nn.Parameter(center_loss.centers.data.clone())

    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).cuda()
    import copy

    loss = center_loss(copy.deepcopy(features), copy.deepcopy(targets))
    loss2 = center_loss2(copy.deepcopy(features), copy.deepcopy(targets))
    print(torch.allclose(loss, loss2))
