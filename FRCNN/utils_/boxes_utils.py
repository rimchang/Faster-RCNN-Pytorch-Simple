import torch
import numpy as np

# torch tensors
def bbox_overlaps(boxes, gt_boxes):
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = torch.from_numpy(gt_boxes)

    # boxes : some props boxes
    # gt_boxes : ground truth box

    oo = []

    for b in gt_boxes:
        # 각각 0,1,2,3 col을 복사.
        x1 = boxes.select(1, 0).clone()
        y1 = boxes.select(1, 1).clone()
        x2 = boxes.select(1, 2).clone()
        y2 = boxes.select(1, 3).clone()

        # gt와의 intersection 계산
        x1[x1.lt(b[0])] = b[0]
        y1[y1.lt(b[1])] = b[1]
        x2[x2.gt(b[2])] = b[2]
        y2[y2.gt(b[3])] = b[3]

        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # intersection area
        inter = torch.mul(w, h).float()

        # total area
        aarea = torch.mul((boxes.select(1, 2) - boxes.select(1, 0) + 1), (boxes.select(1, 3) - boxes.select(1, 1) + 1)).float()
        barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

        # intersection over union overlap
        o = torch.div(inter, (aarea + barea - inter))

        # set invalid entries to 0 overlap
        o[w.lt(0)] = 0
        o[h.lt(0)] = 0

        oo += [o]

    return torch.cat([o.view(-1, 1) for o in oo], 1)


def _unmap(data, count, inds, fill=0):
    """ Unmap boxes subset of item (data) back to the original set of items (of
    size count) """

    # make label data
    # map up to original set of anchors
    # labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    # bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret




def bbox_transform(boxes, gt_boxes):
    # boxes is anchor or pred_rois

    # rois : (x, y, x`, y`)
    # x` - x + 1 = width
    # y` - y  +1 = height
    # x + 0.5 * width = ctr_x
    # y + 0.5 * height = ctr_y

    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    # x : predicted box, x_a : anchor box, x* : ground truth box

    # faster R-Cnn paper
    # t_x = (x - x_a)/w_a
    # t_y = (y - y_a)/h_a
    # t_w = log(w/w_a)
    # t_h = log(h/h_a)

    # t_x* = (x* - x_a)/w_a
    # t_y* = (y* - y_a)/h_a
    # t_w* = log(w*/w_a)
    # t_h* = log(h*/h_a)


    # print("gt_widths / ex_widths", gt_widths / ex_widths)
    # print("gt_heights / ex_heights", gt_heights / ex_heights)
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)


    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    # boxes is anchor or pred_box
    # if image is (224 * 224) then boxes are (1764, 4)
    # deltas are also (1764, 4)

    #print(boxes.shape, deltas.shape)

    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)


    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    col_0 = (slice(0, None, 4))
    col_1 = (slice(1, None, 4))
    col_2 = (slice(2, None, 4))
    col_3 = (slice(3, None, 4))

    # (1764, 1)
    dx = deltas[:, col_0]
    dy = deltas[:, col_1]
    dw = deltas[:, col_2]
    dh = deltas[:, col_3]


    # x : predicted box, x_a : anchor box, x* : ground truth box

    # faster R-Cnn paper
    # t_x = (x - x_a)/w_a
    # t_y = (y - y_a)/h_a
    # t_w = log(w/w_a)
    # t_h = log(h/h_a)

    # t_x* = (x* - x_a)/w_a
    # t_y* = (y* - y_a)/h_a
    # t_w* = log(w*/w_a)
    # t_h* = log(h*/h_a)

    # (1764, 1 ) * (1764, 1) + (1764, 1) = > (1764, 1)
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, col_0] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, col_1] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, col_2] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, col_3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    im_shape : (H, W)
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1](W)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0](H)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    #np.set_printoptions(threshold=np.nan)
    #print("box: {}, ws : {}, hs : {}".format(boxes, ws, hs))
    #np.set_printoptions(threshold=100)
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    #print('box_num : {}, filter_boxe : {}'.format(boxes.shape[0], keep.shape[0]))
    return keep

def py_cpu_nms(proposals_boxes_c, thresh):
    """Pure Python NMS baseline."""
    # proposals_boxes_c (? , 5)[class, x, y, x`, y`]
    scores = proposals_boxes_c[:, 0]
    x1 = proposals_boxes_c[:, 1]
    y1 = proposals_boxes_c[:, 2]
    x2 = proposals_boxes_c[:, 3]
    y2 = proposals_boxes_c[:, 4]
    

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # width * height
    order = scores.argsort()[::-1] # score가 큰것부터.

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0] # keep 해야 하는 index들이다.
        order = order[inds + 1] # inds가 0번째 인덱스를 제외하고 계산했으므로.. 원래 index랑 맞춰줄려면 1을 더해야함.

    return keep

