# encoding: utf-8


from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY
from .classsifiers import get_classifier
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output

@REID_HEADS_REGISTRY.register()
class BNneckHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        self.pool_layer = pool_layer
        self.cfg = cfg
        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)
        hdim = 2048
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        # identity classification layer
        # cls_type = cfg.MODEL.HEADS.CLS_LAYER
        # if cls_type == 'linear':
        #     self.classifier = get_classifier(cfg, cls_type, in_feat, num_classes, bias=False)
        # else:
        #     self.classifier = get_classifier(cfg, cls_type, in_feat, num_classes)
        #
        # self.classifier.apply(weights_init_classifier)


    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]
        bn_feat = self.slf_attn(bn_feat)
        # Evaluation
        if not self.training:

            return bn_feat

        # Training

        # emb_dim = bn_feat.size(-1)
        # class_feat = bn_feat.view(self.cfg.SOLVER.IMS_PER_BATCH/self.cfg.DATALOADER.NUM_INSTANCE,self.cfg.DATALOADER.NUM_INSTANCE,-1)
        # fake_gallery = class_feat[:,:self.cfg.DATALOADER.NUM_INSTANCE//2,:]
        # fake_query = class_feat[:,self.cfg.DATALOADER.NUM_INSTANCE//2:,:]
        #
        #
        # num_query = np.prod(fake_query.shape[-2:])
        # fake_gallery = self.slf_attn(fake_gallery,fake_gallery,fake_gallery)
        # fake_query = fake_query.view(-1,emb_dim).unsqueeze(1)


        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
        else:
            cls_outputs = self.classifier(bn_feat, targets)

        pred_class_logits = F.linear(bn_feat, self.classifier.weight)

        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")
        #fake_label_aux = torch.arange(4, dtype=torch.int8).repeat(10).view(10,4).transpose(0,1).contiguous().view(-1)

        return cls_outputs, pred_class_logits, feat
