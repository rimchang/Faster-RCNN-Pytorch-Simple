import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from data.voc_data import COLORS, VOC_CLASSES
from utils_.boxes_utils import clip_boxes, bbox_transform_inv, py_cpu_nms


def to_var(x, *args, **kwargs):

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if torch.cuda.is_available():
        x = Variable(x, *args, **kwargs).cuda()
    else:
        x = Variable(x, *args, **kwargs)

    return x

def to_tensor(x, *args, **kwargs):

    if torch.cuda.is_available():
        x = torch.from_numpy(x).cuda()
    else:
        x = torch.from_numpy(x)
    return x


def make_name_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + str(i) + "=" + str(hyparam_dict[i]) + "_"
    return str_result[:-1]


def read_pickle(path, model, solver):

    try:
        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print(recent_iter, path)

        with open(path + "/model_" + recent_iter + ".pkl", "rb") as f:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(f))
            else:
                model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

        with open(path + "/solver_" + recent_iter + ".pkl", "rb") as f:
            if torch.cuda.is_available():
                solver.load_state_dict(torch.load(f))
            else:
                solver.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))


    except Exception as e:

        print("fail try read_pickle", e)


def save_pickle(path, epoch, model, solver):

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/model_" + str(epoch) + ".pkl", "wb") as f:
        torch.save(model.state_dict(), f)
    with open(path + "/solver_" + str(epoch) + ".pkl", "wb") as f:
        torch.save(solver.state_dict(), f)

def save_image(path, iteration, img):

    if not os.path.exists(path):
        os.makedirs(path)

    img.save(path + "/img_" + str(iteration) + ".png")


# ============= visualization utils =============#


def label_img_get(img_np, boxes_np=None, labels_np=None, score=" ", show=True):

    # make PIL object
    img = Image.fromarray(img_np.astype('uint8'))
    draw = ImageDraw.Draw(img)

    if boxes_np is not None:

        for i in range(len(boxes_np)):

            color = COLORS[0]
            if labels_np is not None:
                color = COLORS[labels_np[i] % len(COLORS)]
                name = VOC_CLASSES[labels_np[i]] + " " + str(score[i:i+1][0:1])[1:7]
                font = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 20)
                draw.text(boxes_np[i][:2], name, fill=color, font=font)

            coordinate = [(boxes_np[i][0],boxes_np[i][1]), (boxes_np[i][2], boxes_np[i][3])]
            draw.rectangle(coordinate, outline=color)

    if show:
        img.show()
    return img


def score_img_get(img_np, boxes_np, score=" ", show=True):

    # make PIL object
    img = Image.fromarray(img_np.astype('uint8'))
    draw = ImageDraw.Draw(img)

    sorted_indices = np.argsort(score, axis=0) # ascent order

    if boxes_np is not None:

        for i, idx in enumerate(sorted_indices):

            intensity = int(i / len(sorted_indices) * 255)

            # 명도가 높을 수록... 더 score가 높은 box RPN 결과 시각화를 위해!
            color = (intensity, intensity, intensity, 128)

            draw.rectangle(boxes_np[idx], outline=color)

    if show:
        img.show()
    return img

def obj_img_get(image_np, cls_score_np, bbox_pred_np, roi_boxes_np, args, show=True):

    gt_assignment = np.argmax(cls_score_np, axis=1)
    max_score = np.max(cls_score_np, axis=1)

    # bbox_pred_np (?, 84) make to bbox_deltas (?, 4)
    bbox_deltas = []
    for index in range(len(gt_assignment)):
        cls = int(gt_assignment[index])
        start = 4 * cls
        end = start + 4
        bbox_deltas.append(bbox_pred_np[index, start:end])

    bbox_deltas = np.vstack(bbox_deltas)

    if args.target_normalization:

        bbox_deltas = bbox_deltas * np.array((0.1, 0.1, 0.2, 0.2)) - np.array((0.0, 0.0, 0.0, 0.0))


    keep = np.where((gt_assignment > 0) & (max_score >= 0.05))[0]

    gt_assignment = gt_assignment[keep]
    max_score = max_score[keep]
    bbox_deltas = bbox_deltas[keep]
    roi_boxes_np = roi_boxes_np[keep]

    # bbox_deltas to boxes
    boxes = bbox_transform_inv(roi_boxes_np, bbox_deltas)

    # clip boxes
    boxes = clip_boxes(boxes, image_np.shape[:2])
    boxes_c = np.hstack((boxes, max_score[:, np.newaxis]))

    # apply nms for last object detection and find unsurely object , background object
    nms_keep = py_cpu_nms(boxes_c, args.frcnn_nms)

    gt_assignment = gt_assignment[nms_keep]
    max_score = max_score[nms_keep]
    boxes = boxes[nms_keep]


    # filtering for don't print too many object
    sort_keep = np.argsort(max_score)[::-1]
    sort_keep = sort_keep[:args.num_printobj]

    img = label_img_get(image_np, boxes[sort_keep], gt_assignment[sort_keep], max_score[sort_keep], show=show)

    return img

def obj_notreg_img_get(image_np, cls_score_np, roi_boxes_np, show=True):

    gt_assignment = np.argmax(cls_score_np, axis=1)
    max_score = np.max(cls_score_np, axis=1)

    # clip boxes
    cliped_boxes = clip_boxes(roi_boxes_np, image_np.shape[:2])

    fg_keep_not = np.where(gt_assignment == 0)

    # keep nms_keep, and remove th_keep_not, fg_keep_not
    mask = np.ones(gt_assignment.shape[0], dtype=bool)
    mask[fg_keep_not] = False


    img = label_img_get(image_np, cliped_boxes[mask], gt_assignment[mask], max_score[mask], show=show)

    return img