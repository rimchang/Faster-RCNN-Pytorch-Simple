import torch
import torch.optim as optim
from torchvision import transforms
from collections import OrderedDict
from time import perf_counter as pc
import os

from data import VOCDetection, detection_collate, AnnotationTransform
from utils_.utils import to_var, make_name_string
from utils_.anchors import get_anchors, anchor

from model import *
from proposal import ProposalLayer
from target import rpn_targets, frcnn_targets
from loss import rpn_loss, frcnn_loss



def train(args):

    hyparam_list = [("model", args.model_name),
                    ("momentum", args.momentum),
                    ("weight_decay", args.weight_decay),
                    ("lr", args.lr)]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    name_param = make_name_string(hyparam_dict)
    print(name_param)


    # for using tensorboard
    if args.use_tensorboard:
        import tensorflow as tf

        summary_writer = tf.summary.FileWriter(args.output_dir + args.log_dir + name_param)

        def inject_summary(summary_writer, tag, value, step):
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                summary_writer.add_summary(summary, global_step=step)

        inject_summary = inject_summary

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = VOCDetection(root="../input/VOCdevkit", image_set="train",
                            transform=transform, target_transform=AnnotationTransform())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True,
    num_workers = 1, collate_fn = detection_collate)

    # model define
    t0 = pc()
    feature_extractor = CNN()
    rpn = RPN()
    fasterrcnn = FasterRcnn()


    proplayer = ProposalLayer(args=args)
    roipool = ROIpooling()
    print("model loading time : {:.2f}".format(pc() - t0))

    solver = optim.SGD([
                            {'params': feature_extractor.parameters(), 'lr': args.ft_lr},
                            {'params': rpn.parameters()},
                            {'params': fasterrcnn.parameters()}
                        ], lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

    solver.param_groups[0]['step'] = 0
    solver.param_groups[0]['iter'] = 0

    def adjust_learning_rate(optimizer, epoch, ft_step=args.ft_step, step=args.lr_step):

        if epoch == ft_step:
            optimizer.param_groups[0]['lr'] = args.lr
            print("CNN network's learning rate increase to {}".format(args.lr))
        if step is not None and epoch in step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print("ALL network's learning rate decrease to {}".format(param_group['lr']))

    if torch.cuda.is_available():
        print("using cuda")
        feature_extractor.cuda()
        rpn.cuda()
        roipool.cuda()
        fasterrcnn.cuda()

    for epoch in range(args.n_epochs):
        for image, gt_boxes_c in trainloader:

            # boxes_c : [class, x, y, x`, y`]
            # boxes : [x, y, x`, y`]
            
            # TODO 이미지 Info만 가져오고.. 이미지 scale 하고 annotation 전처리해야한다.
            scale = 1
            image_info = (image.size()[2], image.size()[3], image.size()[1], scale) # (H, W, C, S)
            gt_boxes_c = gt_boxes_c.numpy()
            image = to_var(image)

            t0 = pc()
            features = feature_extractor.forward(image)
            rpn_bbox_pred, rpn_cls_prob = rpn(features)


            # ============= region proposal =============#

            all_anchors_boxes = get_anchors(features, anchor)
            proposals_boxes, scores = proplayer.proposal(rpn_bbox_pred, rpn_cls_prob, all_anchors_boxes, im_info=image_info)

            # ============= Get Targets =================#

            rpn_labels, rpn_bbox_targets = rpn_targets(all_anchors_boxes, image, gt_boxes_c, args)
            frcnn_labels, roi_boxes, frcnn_bbox_targets = frcnn_targets(proposals_boxes, gt_boxes_c, args)

            # ============= frcnn ========================#

            rois_features = roipool(features, roi_boxes)
            bbox_pred, cls_score = fasterrcnn(rois_features)

            # ============= Compute loss =================#

            rpnloss = rpn_loss(rpn_cls_prob, rpn_bbox_pred, rpn_labels, rpn_bbox_targets)
            frcnnloss = frcnn_loss(cls_score, bbox_pred, frcnn_labels, frcnn_bbox_targets)
            total_loss = rpnloss + frcnnloss

            print(total_loss)

            # ============= Update =======================#

            solver.zero_grad()
            total_loss.backward()

            solver.step()
            print("one batch training time : {:.2f}".format(pc() - t0))

            # =============== logging each iteration ===============#

            each_mini_batchsize = frcnn_labels.shape[0]
            solver.param_groups[0]['iter'] += 1
            solver.param_groups[0]['step'] += each_mini_batchsize
            iteration = solver.param_groups[0]['iter']


            if args.use_tensorboard:
                log_save_path = args.output_dir + args.log_dir + name_param
                if not os.path.exists(log_save_path):
                    os.makedirs(log_save_path)

                info = {
                    'loss/rpn_loss': rpnloss.data[0],
                    'loss/frcnn_loss': frcnnloss.data[0],
                    'loss/total_loss': total_loss.data[0],

                    'etc/mini_btsize': each_mini_batchsize,

                }

                for tag, value in info.items():
                    inject_summary(summary_writer, tag, value, iteration)

                summary_writer.flush()

            print('Iter : {}, Step-{} , rpn_loss : {:.4}, frcnn_loss : {:.4}, total_loss : {:.4}, mn_btsize : {} ,lr : {:.4}'
                .format(
                    solver.param_groups[0]['iter'],
                    solver.param_groups[0]['step'],
                    rpnloss.data[0],
                    frcnnloss.data[0],
                    total_loss.data[0],
                    each_mini_batchsize,
                    solver.state_dict()['param_groups'][0]["lr"])
                )

            # =============== each epoch save model or save image ===============#

            adjust_learning_rate(solver, epoch)
            if (epoch + 1) % args.pickle_step == 0:
                pickle_save_path = args.output_dir + args.pickle_dir + name_param
                path = pickle_save_path

                if not os.path.exists(path):
                    os.makedirs(path)


                #with open(path + "/optim.pkl", "wb") as f:
                #    torch.save(solver.state_dict(), f)

                #print("save optim")
