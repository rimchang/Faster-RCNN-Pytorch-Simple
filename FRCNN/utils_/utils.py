import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from data.voc_data import COLORS, VOC_CLASSES
from utils_.boxes_utils import clip_boxes, bbox_transform_inv, py_cpu_nms


def to_var(x, *args, **kwargs):
    if torch.cuda.is_available():
        x = Variable(x, *args, **kwargs).cuda()
    else:
        x = Variable(x, *args, **kwargs)

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


def img_get(img_np, boxes_np=None, labels_np=None, score=" ", show=True):
    img_np = img_np.squeeze().transpose((1, 2, 0))


    # denormalize
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([ 0.485,  0.456,  0.406])
    img_np = (std * img_np + mean) * 255

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

            draw.rectangle(boxes_np[i], outline=color)

    if show:
        img.show()
    return img


def proposal_img_get(img_np, boxes_np=None, labels_np=None, score=" ", show=True):
    img_np = img_np.squeeze().transpose((1, 2, 0))


    # denormalize
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([ 0.485,  0.456,  0.406])
    img_np = (std * img_np + mean) * 255

    # make PIL object
    img = Image.fromarray(img_np.astype('uint8'))
    draw = ImageDraw.Draw(img)



    sorted_indices = np.argsort(score, axis=0)

    if boxes_np is not None:

        for i in range(len(boxes_np)):

            where_argmax = np.where(sorted_indices == i)[0][0]
            intensity = int(where_argmax / len(sorted_indices) * 255)

            # 명도가 높을 수록... 더 score가 높은 box RPN 결과 시각화를 위해!
            color = (intensity, intensity, intensity, 128)

            draw.rectangle(boxes_np[i], outline=color)

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


    # bbox_deltas to boxes
    boxes = bbox_transform_inv(roi_boxes_np, bbox_deltas)
    boxes_c = np.hstack((boxes, max_score[:, np.newaxis]))

    # clip boxes
    cliped_boxes = clip_boxes(boxes, image_np.shape[-2:])

    # apply nms for last object detection and find unsurely object , background object
    nms_keep = py_cpu_nms(boxes_c, args.frcnn_nms)
    th_keep_not = np.where(max_score < args.test_ob_thresh)
    fg_keep_not = np.where(gt_assignment == 0)


    # keep nms_keep, and remove th_keep_not, fg_keep_not
    mask = np.zeros(gt_assignment.shape[0], dtype=bool)
    mask[nms_keep] = True
    mask[th_keep_not] = False
    mask[fg_keep_not] = False

    boxes_filterd = cliped_boxes[mask]
    labels_filterd = gt_assignment[mask]
    score_filterd = max_score[mask]


    # filtering for don't print too many object
    sort_keep = np.argsort(score_filterd)[::-1]
    sort_keep = sort_keep[:args.num_printobj]

    img = img_get(image_np, boxes_filterd[sort_keep], labels_filterd[sort_keep], score_filterd[sort_keep], show=show)

    return img