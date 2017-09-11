import torch
import torch.nn as nn
import numpy as np

from utils_.utils import to_var

def rpn_loss(rpn_cls_prob, rpn_bbox_pred, labels, bbox_targets):

    if isinstance(rpn_cls_prob, np.ndarray):
        rpn_cls_prob = torch.from_numpy(rpn_cls_prob)
    if isinstance(rpn_bbox_pred, np.ndarray):
        rpn_bbox_pred = torch.from_numpy(rpn_bbox_pred)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    if isinstance(bbox_targets, np.ndarray):
        bbox_targets = torch.from_numpy(bbox_targets)

    # rpn_cls_prob : torch.Size([1, 18, 14, 14])
    # rpn_bbox_pred : torch.Size([1, 36, 14, 14])
    # labels : numpy.ndarray  (1764,)
    # bbox_targets : numpy.ndarray  (1764, 4)

    height, width = rpn_cls_prob.size()[-2:]  # 14, 14

    # (1, 18, 14, 14) => (9, 2, 14, 14) => (9, 14, 14, 2) => (9 * 14 * 14, 2)
    rpn_cls_prob = rpn_cls_prob.view(-1, 2, height, width).permute(0, 2, 3, 1).contiguous().view(-1, 2)

    labels = labels.long()  # convert properly
    # (1764,) => (1, 14 , 14 , 9) => (1, 9, 14, 14) => (1764, )
    labels = labels.view(1, height, width, -1).permute(0, 3, 1, 2).contiguous()
    labels = labels.view(-1)

    # index where not -1
    idx = labels.ge(0).nonzero()[:, 0]
    rpn_cls_prob = rpn_cls_prob.index_select(0, to_var(idx, requires_grad=False))
    labels = labels.index_select(0, idx)
    labels = to_var(labels, requires_grad=False)

    # (1764, 4) => (1, 14, 14, 36) => (1, 36, 14, 14)
    rpn_bbox_targets = bbox_targets
    rpn_bbox_targets = rpn_bbox_targets.view(1, height, width, -1).permute(0, 3, 1, 2)
    rpn_bbox_targets = to_var(rpn_bbox_targets, requires_grad=False)

    # TODO 로스이렇게 쓰는게 맞는지...확인해라!!

    cls_crit = nn.NLLLoss()
    reg_crit = nn.SmoothL1Loss()
    log_rpn_cls_prob = torch.log(rpn_cls_prob)
    cls_loss = cls_crit(log_rpn_cls_prob, labels)
    # verify normalization and sigma
    reg_loss = reg_crit(rpn_bbox_pred, rpn_bbox_targets)


    return cls_loss, reg_loss


def frcnn_loss(scores, bbox_pred, labels, bbox_targets):

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

    # print(bbox_targets)

    cls_crit = nn.NLLLoss()
    log_scores = torch.log(scores)
    cls_loss = cls_crit(log_scores, labels)

    # print(log_scores.size(), labels.size())

    reg_crit = nn.SmoothL1Loss()
    reg_loss = reg_crit(bbox_pred, bbox_targets)
    # print(log_scores.size(), labels.size(), bbox_pred.size(), bbox_targets.size())

    return cls_loss, reg_loss

