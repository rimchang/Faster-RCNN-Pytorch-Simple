from torchvision import transforms

from data.voc_data import VOCDetection, detection_collate, AnnotationTransform
from loss import rpn_loss, frcnn_loss
from model import *
from proposal import ProposalLayer
from target import rpn_targets, frcnn_targets
from utils_.anchors import get_anchors, anchor

def debug(args):

    # data test

    transform = transforms.Compose([transforms.ToTensor()])


    trainset = VOCDetection(root="../input/VOCdevkit", image_set="train",
                            transform=transform, target_transform=AnnotationTransform())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True,
    num_workers = 1, collate_fn = detection_collate)

    it = iter(trainloader)
    image, gt_boxes = next(it)

    # TODO 이미지 Info만 가져오고.. 이미지 scale 하고 annotation 전처리해야한다.
    scale = 1
    image_info = (image.size()[2], image.size()[3], image.size()[1], scale)
    gt_boxes = gt_boxes.numpy()


    image = to_var(image)
    print("image : {}, gt_boxes : {}".format(image.size(), gt_boxes))

    # CNN test

    feature_extractor = CNN()
    features = feature_extractor.forward(image)  # torch.Size([1, 512, 62, 37])
    print("features : {} ".format(features.size()))

    # RPN test

    rpn = RPN()
    rpn_bbox_pred, rpn_cls_prob = rpn(features)
    print("rpn_bbox_pred : {}, rpn_cls_prob : {}".format(rpn_bbox_pred.size(), rpn_cls_prob.size())) # torch.Size([1, 36, 62, 37]) torch.Size([1, 18, 62, 37])

    # get_achors test

    all_anchors = get_anchors(features, anchor)
    print("all_anchors : {}".format(all_anchors.shape))

    # proposal layer test

    proplayer = ProposalLayer(rpn_bbox_pred, rpn_cls_prob, all_anchors, im_info=image_info, args=args)
    proposals, scores = proplayer.proposal()
    print("proposals : {}, scores : {}".format(proposals.shape, scores.shape))
    print(proposals.astype("int"))

    # rpn_target test
    rpn_labels, rpn_bbox_targets = rpn_targets(all_anchors, image, gt_boxes, args)
    print("rpn_labels : {}, bbox_target : {}".format(rpn_labels.shape, rpn_bbox_targets.shape)) # (20646,) (20646, 4)

    # gt_boxes도 추가해줘야 해서 targets을 먼저 구한다.
    # frcnn_targets test
    frcnn_labels, rois, frcnn_bbox_targets = frcnn_targets(proposals, gt_boxes, args)
    print("frcnn_labels : {}, rois : {}, frcnn_bbox_targets : {}".format(frcnn_labels.shape, rois.shape, frcnn_bbox_targets.shape))

    # ROIpooling test
    roipool = ROIpooling()
    rois_features = roipool(features, rois)
    print("rois_features : {}".format(rois_features.size()))

    # FasterRcnn test
    fasterrcnn = FasterRcnn()
    bbox_pred, cls_score = fasterrcnn(rois_features)
    print("bbox_pred : {}, cls_scores : {}".format(bbox_pred.size(), cls_score.size()))

    # rpn_loss test
    rpnloss = rpn_loss(rpn_cls_prob, rpn_bbox_pred, rpn_labels, rpn_bbox_targets)
    print("rpnloss : {}".format(rpnloss))

    # frcnn_loss test
    print([type(j) for j in [cls_score, bbox_pred, frcnn_labels, frcnn_bbox_targets]])
    frcnnloss = frcnn_loss(cls_score, bbox_pred, frcnn_labels, frcnn_bbox_targets)
    print("frcnnloss : {} ".format(frcnnloss))

    total_loss = rpnloss + frcnnloss
    print("total_loss : {}".format(total_loss))

    total_loss.backward()