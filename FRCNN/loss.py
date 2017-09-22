import torch
import torch.nn as nn
import numpy as np

from utils_.utils import to_var

def rpn_loss(rpn_cls_prob, rpn_bbox_pred, labels, bbox_targets, rpn_bbox_inside_weights, logits):

    if isinstance(rpn_cls_prob, np.ndarray):
        rpn_cls_prob = torch.from_numpy(rpn_cls_prob)
    if isinstance(rpn_bbox_pred, np.ndarray):
        rpn_bbox_pred = torch.from_numpy(rpn_bbox_pred)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    if isinstance(bbox_targets, np.ndarray):
        bbox_targets = torch.from_numpy(bbox_targets)

    # rpn_cls_prob : torch.Size([1, 18, H/16, W/16])
    # rpn_bbox_pred : torch.Size([1, 36, H/16, 2/16])
    # labels : numpy.ndarray  (H/16 * W/16,)
    # bbox_targets : numpy.ndarray  (H/16 * W/16, 4)

    height, width = rpn_cls_prob.size()[-2:]  # 14, 14

    # (1, 18, H/16, W/16) => (9, 2, H/16, W/16) => (9, H/16, W/16, 2) => (9 * H/16 * W/16, 2)
    rpn_cls_prob = rpn_cls_prob.view(-1, 2, height, width).permute(0, 2, 3, 1).contiguous().view(-1, 2)

    labels = labels.long()  # convert properly
    # (H/16 * W/16 * 9) => (1, H/16, W/16, 9) => (1, 9, H/16, W/16) => (9 * H/16 * W/16, )
    labels = labels.view(1, height, width, -1).permute(0, 3, 1, 2).contiguous()
    labels = labels.view(-1)

    # index where not -1
    idx = labels.ge(0).nonzero()[:, 0]
    rpn_cls_prob = rpn_cls_prob.index_select(0, to_var(idx))
    labels = labels.index_select(0, idx)
    logits = logits.squeeze().index_select(0, to_var(idx))


    # (H/16 * W/16, 4) => (1, H/16, W/16, 36) => (1, 36, H/16, W/16)
    rpn_bbox_targets = bbox_targets
    rpn_bbox_targets = rpn_bbox_targets.view(1, height, width, -1).permute(0, 3, 1, 2)
    rpn_bbox_targets = to_var(rpn_bbox_targets)

    po_cnt = torch.sum(labels.eq(1))
    ne_cnt = torch.sum(labels.eq(0))

    # for debug
    maxv, predict = rpn_cls_prob.data.max(1)
    

    po_idx = labels.eq(1).nonzero()[:, 0].cuda() if torch.cuda.is_available() else labels.eq(1).nonzero()[:, 0]
    ne_idx = labels.eq(0).nonzero()[:, 0].cuda() if torch.cuda.is_available() else labels.eq(0).nonzero()[:, 0]
    labels = labels.cuda() if torch.cuda.is_available() else labels

    tp = torch.sum(predict.index_select(0, po_idx).eq(labels.index_select(0, po_idx))) if po_cnt > 0 else 0
    tn = torch.sum(predict.index_select(0, ne_idx).eq(labels.index_select(0, ne_idx)))

    labels = to_var(labels)
    #cls_crit = nn.NLLLoss()
    #cls_loss = cls_crit(rpn_cls_prob, labels)

    cls_crit = nn.CrossEntropyLoss()
    cls_loss = cls_crit(logits, labels)

    # (H/16 * W/16, 4) => (1, H/16, W/16, 36) => (1, 36, H/16, W/16)
    rpn_bbox_inside_weights = torch.from_numpy(rpn_bbox_inside_weights)
    rpn_bbox_inside_weights = rpn_bbox_inside_weights.view(1, height, width, -1).permute(0, 3, 1, 2)
    rpn_bbox_inside_weights = rpn_bbox_inside_weights.cuda() if torch.cuda.is_available() else rpn_bbox_inside_weights     


    rpn_bbox_pred = to_var(torch.mul(rpn_bbox_pred.data, rpn_bbox_inside_weights))
    rpn_bbox_targets = to_var(torch.mul(rpn_bbox_targets.data, rpn_bbox_inside_weights))


    reg_crit = nn.SmoothL1Loss(size_average=False)
    reg_loss = reg_crit(rpn_bbox_pred, rpn_bbox_targets) / (po_cnt + 1e-4)

    log = (po_cnt, ne_cnt, tp, tn)
    print(log)
    return cls_loss, reg_loss * 10, log


def frcnn_loss(scores, bbox_pred, labels, bbox_targets, frcnn_bbox_inside_weights, logits):

    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    if isinstance(bbox_pred, np.ndarray):
        bbox_pred = torch.from_numpy(bbox_pred)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    if isinstance(bbox_targets, np.ndarray):
        bbox_targets = torch.from_numpy(bbox_targets)

    labels = to_var(labels)
    labels = labels.long()
    bbox_targets = to_var(bbox_targets)

    fg_cnt = torch.sum(labels.data.ne(0))
    bg_cnt = labels.data.numel() - fg_cnt
    #print(fg_cnt, bg_cnt)

    maxv, predict = scores.data.max(1)
    tp = torch.sum(predict[:fg_cnt].eq(labels.data[:fg_cnt])) if fg_cnt > 0 else 0
    tn = torch.sum(predict[fg_cnt:].eq(labels.data[fg_cnt:]))


    ce_weights = torch.ones(scores.size()[1])
    ce_weights[0] = float(fg_cnt) / bg_cnt

    if torch.cuda.is_available():
        ce_weights = ce_weights.cuda()

    #cls_crit = nn.NLLLoss(weight=ce_weights)
    #cls_loss = cls_crit(scores, labels)

    logits = logits.squeeze()
    cls_crit = nn.CrossEntropyLoss(weight=ce_weights)
    cls_loss = cls_crit(logits, labels)

    frcnn_bbox_inside_weights = torch.from_numpy(frcnn_bbox_inside_weights).cuda() if torch.cuda.is_available() else torch.from_numpy(frcnn_bbox_inside_weights)
   

    bbox_pred = to_var(torch.mul(bbox_pred.data, frcnn_bbox_inside_weights))
    bbox_targets = to_var(torch.mul(bbox_targets.data, frcnn_bbox_inside_weights))

    reg_crit = nn.SmoothL1Loss(size_average=False)
    reg_loss = reg_crit(bbox_pred, bbox_targets)  / (fg_cnt + 1e-4)

    log = (fg_cnt, bg_cnt, tp, tn)

    return cls_loss, reg_loss * 10, log

