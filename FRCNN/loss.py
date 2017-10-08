import torch
import torch.nn as nn
import numpy as np

from utils_.utils import to_var, to_tensor

def rpn_loss(rpn_cls_prob, rpn_logits, rpn_bbox_pred, rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights):

    """
    Arguments:
        rpn_cls_prob (Tensor): (1, 2*9, H/16, W/16)
        rpn_logits (Tensor): (H/16 * W/16 , 2) object or non-object rpn_logits
        rpn_bbox_pred (Tensor): (1, 4*9, H/16, W/16) predicted boxes
        rpn_labels (Ndarray) : (H/16 * W/16 * 9 ,)
        rpn_bbox_targets (Ndarray) : (H/16 * W/16 * 9, 4)
        rpn_bbox_inside_weights (Ndarray) : (H/16 * W/16 * 9, 4) masking for only positive box loss

    Return:
        cls_loss (Scalar) : classfication loss
        reg_loss * 10 (Scalar) : regression loss
        log (Tuple) : for logging
    """

    height, width = rpn_cls_prob.size()[-2:]  # (H/16, W/16)

    rpn_cls_prob = rpn_cls_prob.squeeze(0).permute(1, 2, 0).contiguous()  # (1, 18, H/16, W/16) => (H/16 ,W/16, 18)
    rpn_cls_prob = rpn_cls_prob.view(-1, 2)  # (H/16 ,W/16, 18) => (H/16 * W/16 * 9, 2)

    rpn_labels = to_tensor(rpn_labels).long() # convert properly # (H/16 * W/16 * 9)


    # index where not -1
    idx = rpn_labels.ge(0).nonzero()[:, 0]
    rpn_cls_prob = rpn_cls_prob.index_select(0, to_var(idx))
    rpn_labels = rpn_labels.index_select(0, idx)
    rpn_logits = rpn_logits.squeeze().index_select(0, to_var(idx))


    po_cnt = torch.sum(rpn_labels.eq(1))
    ne_cnt = torch.sum(rpn_labels.eq(0))

    # for debug
    maxv, predict = rpn_cls_prob.data.max(1)
    
    #print(predict)
    po_idx = predict.eq(1).nonzero()
    ne_idx = predict.eq(0).nonzero()

    po_idx = po_idx.view(-1) if po_idx.dim() > 0 else None
    ne_idx = ne_idx.view(-1) if ne_idx.dim() > 0 else None

    try:
        tp = torch.sum(predict.index_select(0, po_idx).eq(rpn_labels.index_select(0, po_idx))) if po_cnt > 0 and po_idx is not None else 0
        tn = torch.sum(predict.index_select(0, ne_idx).eq(rpn_labels.index_select(0, ne_idx))) if ne_cnt > 0 and ne_idx is not None else 0
    except Exception as e:
        print(e)
        tp = 0
        tn = 0

    rpn_labels = to_var(rpn_labels)

    cls_crit = nn.NLLLoss()
    log_rpn_cls_prob = torch.log(rpn_cls_prob)
    cls_loss = cls_crit(log_rpn_cls_prob, rpn_labels)

    #cls_crit = nn.CrossEntropyLoss()
    #cls_loss = cls_crit(rpn_logits, rpn_labels)



    rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets)
    rpn_bbox_targets = rpn_bbox_targets.view(height, width, 36)  # (H/16 * W/16 * 9, 4)  => (H/16 ,W/16, 36)
    rpn_bbox_targets = rpn_bbox_targets.permute(2, 0, 1).contiguous().unsqueeze(0) # (H/16 ,W/16, 36) => (1, 36, H/16, W/16)
    rpn_bbox_targets = to_var(rpn_bbox_targets)


    rpn_bbox_inside_weights = torch.from_numpy(rpn_bbox_inside_weights)
    rpn_bbox_inside_weights = rpn_bbox_inside_weights.view(height, width, 36)  # (H/16 * W/16 * 9, 4)  => (H/16 ,W/16, 36)
    rpn_bbox_inside_weights = rpn_bbox_inside_weights.permute(2, 0, 1).contiguous().unsqueeze(0) # (H/16 ,W/16, 36) => (1, 36, H/16, W/16)

    rpn_bbox_inside_weights = rpn_bbox_inside_weights.cuda() if torch.cuda.is_available() else rpn_bbox_inside_weights     


    rpn_bbox_pred = to_var(torch.mul(rpn_bbox_pred.data, rpn_bbox_inside_weights))
    rpn_bbox_targets = to_var(torch.mul(rpn_bbox_targets.data, rpn_bbox_inside_weights))

    reg_crit = nn.SmoothL1Loss(size_average=False)
    reg_loss = reg_crit(rpn_bbox_pred, rpn_bbox_targets) / rpn_labels.size(0)

    #reg_crit = nn.SmoothL1Loss(size_average=False)
    #reg_loss = reg_crit(rpn_bbox_pred, rpn_bbox_targets) / (rpn_bbox_pred.size(0) / 9)

    #reg_crit = nn.SmoothL1Loss(size_average=False)
    #reg_loss = reg_crit(rpn_bbox_pred, rpn_bbox_targets) / (po_cnt * 4 + 1e-4)

    # for avoid print error
    if po_cnt == 0:
        po_cnt = 1
    if ne_cnt == 0:
        ne_cnt = 1

    log = (po_cnt, ne_cnt, tp, tn)
    #print("rpn log", log, "loss" ,cls_loss.data[0], reg_loss.data[0] * 10)

    return cls_loss, reg_loss, log


def frcnn_loss(frcnn_cls_prob, frcnn_logits, frcnn_bbox_pred, frcnn_labels, frcnn_bbox_targets, frcnn_bbox_inside_weights):
    
    """
    Arguments:
        frcnn_cls_prob (Tensor): (256, 21) 21 class prob
        frcnn_logits (Tensor): (256 , 21) 21 class logtis
        frcnn_bbox_pred (Tensor): (256, 84) predicted boxes for 21 class
        frcnn_labels (Ndarray) : (256,)
        frcnn_bbox_targets (Ndarray) : (256, 84)
        frcnn_bbox_inside_weights (Ndarray) : (256, 84) masking for only foreground box loss

    Return:
        cls_loss (Scalar) : classfication loss
        reg_loss * 10 (Scalar) : regression loss
        log (Tuple) : for logging
    """


    frcnn_labels = to_tensor(frcnn_labels).long()
    fg_cnt = torch.sum(frcnn_labels.ne(0))
    bg_cnt = frcnn_labels.numel() - fg_cnt
    #print(fg_cnt, bg_cnt)

    try:
        maxv, predict = frcnn_cls_prob.data.max(1)
        tp = torch.sum(predict[:fg_cnt].eq(frcnn_labels[:fg_cnt])) if fg_cnt > 0 else 0
        tn = torch.sum(predict[fg_cnt:].eq(frcnn_labels[fg_cnt:])) if bg_cnt > 0 else 0
    except Exception as e:
        print(e)
        print(fg_cnt, frcnn_labels)
        tp = 0
        tn = 0

    frcnn_labels = to_var(frcnn_labels)



    ce_weights = torch.ones(frcnn_cls_prob.size()[1])
    ce_weights[0] = float(fg_cnt) / bg_cnt if bg_cnt != 0 else 1

    if torch.cuda.is_available():
        ce_weights = ce_weights.cuda()


    cls_crit = nn.NLLLoss(weight=ce_weights)
    log_frcnn_cls_prob = torch.log(frcnn_cls_prob)
    cls_loss = cls_crit(log_frcnn_cls_prob, frcnn_labels)


    #cls_crit = nn.CrossEntropyLoss(weight=ce_weights)
    #cls_crit = nn.CrossEntropyLoss()
    #cls_loss = cls_crit(frcnn_logits, frcnn_labels)

    frcnn_bbox_inside_weights = to_tensor(frcnn_bbox_inside_weights)
    frcnn_bbox_targets = to_tensor(frcnn_bbox_targets)

    frcnn_bbox_pred = to_var(torch.mul(frcnn_bbox_pred.data, frcnn_bbox_inside_weights))
    frcnn_bbox_targets = to_var(torch.mul(frcnn_bbox_targets, frcnn_bbox_inside_weights))

    #reg_crit = nn.SmoothL1Loss(size_average=False)
    reg_crit = nn.SmoothL1Loss(size_average=True)
    reg_loss = reg_crit(frcnn_bbox_pred, frcnn_bbox_targets) # / (fg_cnt * 4 + 1e-4)

    # for avoid print error
    if fg_cnt == 0:
        fg_cnt = 1
    if bg_cnt == 0:
        bg_cnt = 1

    log = (fg_cnt, bg_cnt, tp, tn)
    #print("frcnn log", log)

    return cls_loss, reg_loss, log

