from collections import OrderedDict
import numpy as np
import os

from data.voc_data import VOC_CLASSES
from data.voc_eval import voc_eval
from utils_.utils import make_name_string



def eval(args):

    hyparam_list = [("model", args.model_name),
                    ("train", (args.pre_nms_topn, args.nms_thresh, args.post_nms_topn)),
                    ("test", (args.test_pre_nms_topn, args.test_nms_thresh, args.test_post_nms_topn)),
                    ("pos_th", args.pos_threshold),
                    ("bg_th", args.bg_threshold),
                    ("init_gau", args.init_gaussian),
                    ("last_nms", args.frcnn_nms),
                    ("init_gau", args.init_gaussian),
                    ("include_gt", args.include_gt),
                    ("ft_conv3", args.ft_conv3),
                    ("lr", args.lr)]


    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    name_param = "/" + make_name_string(hyparam_dict)
    print(name_param)

    # =========== evaluation ============#


    if args.make_val_boxes:
        path = args.output_dir + args.result_dir + name_param
    else:
        path =  "." + args.result_dir + name_param


    _devkit_path = args.input_dir + "/VOCdevkit"
    annopath = os.path.join(
        _devkit_path,
        'VOC2007',
        'Annotations',
        '{:s}.xml')
    imagesetfile = os.path.join(
        _devkit_path,
        'VOC2007',
        'ImageSets',
        'Main',
        'val.txt')

    cachedir = os.path.join(path, 'annotations_cache')
    aps = []

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)




    filename = path + "/det_test_{:s}.txt"

    for i, cls in enumerate(VOC_CLASSES):
        if cls == 'background':
            continue
        cls_filename = filename.format(cls)
        rec, prec, ap = voc_eval(
            cls_filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=True)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))


    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

