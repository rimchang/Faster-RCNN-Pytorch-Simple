from collections import OrderedDict
from time import perf_counter as pc

import matplotlib
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torchvision import transforms
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf

from data.voc_data import VOCDetection, detection_collate, AnnotationTransform
from utils_.utils import make_name_string
from utils_.anchors import get_anchors, anchor

from model import *
from utils_.utils import *
from proposal import ProposalLayer
from target import rpn_targets, frcnn_targets
from loss import rpn_loss, frcnn_loss
from utils_.boxes_utils import RandomHorizontalFlip, Maxsizescale
from vgg import VGG16

def train(args):

    hyparam_list = [("model", args.model_name),
                    ("bk", args.backbone),
                    ("train", (args.pre_nms_topn, args.nms_thresh, args.post_nms_topn)),
                    ("test", (args.test_pre_nms_topn, args.test_nms_thresh, args.test_post_nms_topn)),
                    ("pos_th", args.pos_threshold),
                    ("bg_th", args.bg_threshold),
                    ("last_nms", args.frcnn_nms),
                    ("include_gt", args.include_gt),
                    ("lr", args.lr)]


    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    name_param = "/" + make_name_string(hyparam_dict)
    print(name_param)


    # for using tensorboard
    if args.use_tensorboard:
        import tensorflow as tf

        summary_writer = tf.summary.FileWriter(args.output_dir + args.log_dir + name_param)

        def inject_summary(summary_writer, tag, value, step):
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                summary_writer.add_summary(summary, global_step=step)

        inject_summary = inject_summary

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = VOCDetection(root=args.input_dir + "/VOCdevkit2007", image_set="trainval",
                            transform=transform, target_transform=AnnotationTransform())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True,
    num_workers = 1, collate_fn = detection_collate)

    # model define
    t0 = pc()

    class Model(nn.Module):
        """
        this Model class is used for simple model saving and loading
        """
        def __init__(self, args):
            super(Model, self).__init__()

            print("using backbone", args.backbone)
            if args.backbone == "vgg16_torch":
                self.feature_extractor = CNN()

            elif args.backbone == "vgg16_longcw":
                self.feature_extractor = VGG16()
                self.feature_extractor.load_from_npy_file('../input/pretrained_model/VGG_imagenet.npy')

            self.rpn = RPN()
            self.fasterrcnn = FasterRcnn()
            self.proplayer = ProposalLayer(args=args)
            self.roipool = ROIpooling()

    model = Model(args)

    feature_extractor = model.feature_extractor
    rpn = model.rpn
    fasterrcnn = model.fasterrcnn
    proplayer = model.proplayer
    roipool = model.roipool

    print("model loading time : {:.2f}".format(pc() - t0))


    # init gaussian N(0, 0.01) for new layers
    for module in [rpn, fasterrcnn, roipool]:

        for key, value in dict(module.named_parameters()).items():
            if 'bias' in key:
                value.data.zero_()
            else:
                value.data.normal_(0, 0.01)


    params = []
    for module in [rpn, fasterrcnn, roipool]:
        for key, value in dict(module.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': args.lr * (args.bias_double_lr + 1),
                                'weight_decay': args.bias_weight_decay and args.weight_decay or 0}]
                else:
                    params += [{'params': [value], 'lr': args.lr, 'weight_decay': args.weight_decay}]



    solver = optim.SGD(params, momentum=args.momentum,weight_decay=args.weight_decay)

    solver.param_groups[0]['epoch'] = 0
    solver.param_groups[0]['iter'] = 0


    path =  "." + args.pickle_dir + name_param

    if os.path.isdir(path) ==False:
        path = args.output_dir + args.pickle_dir + name_param

    read_pickle(path, model, solver)

    def adjust_learning_rate(optimizer, epoch, ft_step=args.ft_step, step=args.lr_step):

        if int(epoch) == int(ft_step):
            optimizer.param_groups[0]['lr'] = args.lr
            print("CNN network's learning rate increase to {}".format(args.lr))
        if step is not None and int(epoch) in step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print("ALL network's learning rate decrease to {}".format(param_group['lr']))

    if torch.cuda.is_available():
        print("using cuda")
        feature_extractor.cuda()
        rpn.cuda()
        roipool.cuda()
        fasterrcnn.cuda()

    feature_extractor.train()
    rpn.train()
    roipool.train()
    fasterrcnn.train()

    if args.backbone == "vgg16_torch":
        pixel_mean = (0.485, 0.456, 0.406)
        pixel_std = (0.229, 0.224, 0.225)

    elif args.backbone == "vgg16_longcw":
        pixel_mean = (122.7717 / 255, 115.9465 / 255, 102.9801 / 255)
        pixel_std = (1, 1, 1)

    for epoch in range(args.n_epochs):

        solver.param_groups[0]['epoch'] += 1
        epoch = solver.param_groups[0]['epoch']

        for image_, gt_boxes_c in trainloader:


            solver.param_groups[0]['iter'] += 1
            iteration = solver.param_groups[0]['iter']

            # boxes_c : [x, y, x`, y`, class]
            # boxes : [x, y, x`, y`]


            image = image_
            old_image_info = (image.size()[2], image.size()[3], image.size()[1]) # (H, W, C)

            """
            im_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Scale(600),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
            """

            random_number = random.random()

            if random_number < 0.5:

                # flip x, x1
                x1 = gt_boxes_c[:, 0].clone()
                x2 = gt_boxes_c[:, 2].clone()

                gt_boxes_c[:, 0] = (old_image_info[1] - 1) - x2
                gt_boxes_c[:, 2] = (old_image_info[1] - 1) - x1


            im_transform = transforms.Compose([
                transforms.ToPILImage(),
                Maxsizescale(600, maxsize=1000),
                RandomHorizontalFlip(random_number),
                transforms.ToTensor(),
                transforms.Normalize(pixel_mean,
                                     pixel_std),
            ])

            image = im_transform(image.squeeze())
            image = image.unsqueeze(0)


            scale = (image.size()[2]/old_image_info[0], image.size()[3]/old_image_info[1]) # new/old (H, W)
            image_info = (image.size()[2], image.size()[3], image.size()[1], scale)  # (H, W, C, S)

            # gt_boxes_c : [x, y, x`, y`, class]

            # x*scale[1](W) , x`*scale[1](W)
            gt_boxes_c[:, 0] *= scale[1]
            gt_boxes_c[:, 2] *= scale[1]

            # y*scale[0](H) , y`*scale[0](H)
            gt_boxes_c[:, 1] *= scale[0]
            gt_boxes_c[:, 3] *= scale[0]

            gt_boxes_c = gt_boxes_c.numpy()


            image = to_var(image)

            t0 = pc()
            features = feature_extractor.forward(image)
            rpn_bbox_pred, rpn_cls_prob, rpn_logits = rpn(features)


            # ============= region proposal =============#

            all_anchors_boxes = get_anchors(features, anchor)
            proposals_boxes, scores = proplayer.proposal(rpn_bbox_pred, rpn_cls_prob, all_anchors_boxes, image_info, test=False, args=args)

            # ============= Get Targets =================#

            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights = rpn_targets(all_anchors_boxes, gt_boxes_c, image_info, args)
            frcnn_labels, roi_boxes, frcnn_bbox_targets, frcnn_bbox_inside_weights = frcnn_targets(proposals_boxes, gt_boxes_c, args)

            if roi_boxes.shape[0] == 0:
                continue

            # ============= frcnn ========================#

            rois_features = roipool(features, roi_boxes)
            frcnn_bbox_pred, frcnn_cls_prob, frcnn_logits = fasterrcnn(rois_features)

            # ============= Compute loss =================#

            rpn_cls_loss, rpn_reg_loss, rpn_log = rpn_loss(rpn_cls_prob, rpn_logits, rpn_bbox_pred, rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights)
            frcnn_cls_loss, frcnn_reg_loss, frcnn_log = frcnn_loss(frcnn_cls_prob, frcnn_logits, frcnn_bbox_pred, frcnn_labels, frcnn_bbox_targets, frcnn_bbox_inside_weights)

            rpnloss = rpn_cls_loss + rpn_reg_loss
            frcnnloss = frcnn_cls_loss + frcnn_reg_loss
            total_loss =  rpnloss + frcnnloss


            # ============= Update =======================#

            solver.zero_grad()
            total_loss.backward()

            tot_norm = clip_grad_norm(model.parameters(), 10.)
            solver.step()


            time = float(pc() - t0)


            # =============== logging each iteration ===============#

            if (iteration) % args.log_step == 0:

                if args.use_tensorboard:
                    log_save_path = args.output_dir + args.log_dir + name_param
                    if not os.path.exists(log_save_path):
                        os.makedirs(log_save_path)

                    info = {
                        'loss/rpn_cls_loss': rpn_cls_loss.data[0],
                        'loss/rpn_reg_loss': rpn_reg_loss.data[0],
                        'loss/rpn_loss': rpnloss.data[0],
                        'loss/frcnn_cls_loss': frcnn_cls_loss.data[0],
                        'loss/frcnn_reg_loss': frcnn_reg_loss.data[0],
                        'loss/frcnn_loss': frcnnloss.data[0],
                        'loss/total_loss': total_loss.data[0],
                        'etc/tot_norm': tot_norm,
                        'etc/rpn_proposal_size': proposals_boxes.shape[0],
                        'etc/rpn_fg_boxes_size': rpn_log[0],
                        'etc/rpn_bg_boxes_size': rpn_log[1],
                        'etc/rpn_TP': rpn_log[2]/rpn_log[0]*100,
                        'etc/rpn_TN': rpn_log[3]/rpn_log[1]*100,
                        'etc/frccn_fg_boxes_size': frcnn_log[0],
                        'etc/frccn_bg_boxes_size': frcnn_log[1],
                        'etc/frccn_TP': frcnn_log[2]/frcnn_log[0]*100,
                        'etc/frccn_TN': frcnn_log[3]/frcnn_log[1]*100,

                    }

                    for tag, value in info.items():
                        inject_summary(summary_writer, tag, value, iteration)

                    summary_writer.flush()



                print('Epoch : {}, Iter-{} , rpn_loss : {:.4f}, frcnn_loss : {:.4f}, total_loss : {:.4f}, lr : {:.4f}, time : {:.4f}'
                    .format(
                    epoch,
                    iteration,
                    rpnloss.data[0],
                    frcnnloss.data[0],
                    total_loss.data[0],
                    solver.state_dict()['param_groups'][0]["lr"],
                    time)
                    , end=" ,"
                )

                print("RPN : {} {} {} {:.2f} {:.2f}, FRCNN : {} {} {} {:.2f} {:.2f}, gt {}"
                    .format(
                    proposals_boxes.shape[0],
                    rpn_log[0],
                    rpn_log[1],
                    rpn_log[2] / rpn_log[0] * 100,
                    rpn_log[3] / rpn_log[1] * 100,
                    frcnn_log[0] + frcnn_log[1],
                    frcnn_log[0],
                    frcnn_log[1],
                    frcnn_log[2] / frcnn_log[0] * 100,
                    frcnn_log[3] / frcnn_log[1] * 100,
                    gt_boxes_c.shape[0])

                )
            # =========== visualization img with object ============#


            if (iteration) % args.image_save_step == 0:

                # all proposals_boxes visualization


                image_np = image.data.cpu().numpy()
                image_np = image_np.squeeze().transpose((1, 2, 0))

                image_np = (pixel_std * image_np + pixel_mean) * 255 # (H, W, C)


                score_np = scores
                cls_score_np = frcnn_cls_prob.data.cpu().numpy()
                roi_boxes_np = np.array(roi_boxes)
                bbox_pred_np = frcnn_bbox_pred.data.cpu().numpy()

                #all_anchors_img = proposal_img_get(image_np, all_anchors_boxes, score=score_np, show=False)
                proposal_img = score_img_get(image_np, proposals_boxes, score=score_np, show=False)
                nreg_img = obj_notreg_img_get(image_np, cls_score_np, roi_boxes_np, show=False)
                obj_img = obj_img_get(image_np, cls_score_np, bbox_pred_np, roi_boxes_np, args, show=False)
                gt_img = label_img_get(image_np, gt_boxes_c[:, :-1], gt_boxes_c[:, -1].astype('int'), show=False)

                fig = plt.figure(figsize=(15, 30)) # width 2700 height 900
                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.02)

                for i, img in enumerate([proposal_img, nreg_img, obj_img, gt_img]):
                    axes = fig.add_subplot(4, 1, 1+i)
                    axes.axis('off')
                    axes.imshow(np.asarray(img, dtype='uint8'), aspect='auto')


                path = args.output_dir + args.image_dir + name_param

                if not os.path.exists(path):
                    os.makedirs(path)

                plt.savefig(path + "/" + str(epoch) + "_" + str(iteration) + '.png')
                plt.close("all")
                print("save image")


        # =============== each epoch save model or save image ===============#

        adjust_learning_rate(solver, epoch)

        if (epoch) % args.pickle_step == 0:
            path = args.output_dir + args.pickle_dir + name_param
            save_pickle(path, epoch, model, solver)
            print("save model, optim")




