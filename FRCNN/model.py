import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from utils_.utils import to_var


class CNN(nn.Module):
    def __init__(self):
        """Load the pretrained vgg11 and delete fc layer."""
        super(CNN, self).__init__()
        vggnet = models.vgg16(pretrained=True)
        modules = list(vggnet.children())[:-1]  # delete the last fc layer.
        modules = list(modules[0])[:-1]  # delete the last pooling layer

        self.vggnet = nn.Sequential(*modules)

    def forward(self, images):
        """Extract the image feature vectors."""

        # return features in relu5_3
        features = self.vggnet(images)
        return features



class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        self.conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU()

        # 9 anchor * 2 classfier (object or non-object)
        self.conv1 = nn.Conv2d(512, 2 * 9, kernel_size=1, stride=1)

        # 9 anchor * 4 coordinate regressor
        self.conv2 = nn.Conv2d(512, 4 * 9, kernel_size=1, stride=1)
        self.softmax2d = nn.Softmax2d()

    def forward(self, features):
        features = self.conv(features)
        features = self.relu(features)
        logits, rpn_bbox_pred = self.conv1(features), self.conv2(features)
        # torch.Size([1, 18, 14, 14]) torch.Size([1, 36, 14, 14])
        # print(logits.size(), rpn_bbox_pred.size())

        logits = logits.view(
            (-1, 2, 9 * features.size()[2] * features.size()[3]))  # (1, 18, 14, 14) => (1, 2, 9  * 14 * 14)
        logits = logits.permute(0, 2, 1)  # (1, 2, 9 * 14 * 14) => (1, 9 * 14 * 14 , 2)
        logits = logits.unsqueeze(3)  # (1, 9 * 14 * 14 , 2) => (1, 9 * 14 * 14 , 2, 1) for softmax2d
        rpn_cls_prob = self.softmax2d(logits)

        # verify normalization by each class
        # 1, 1
        # print(torch.sum(rpn_cls_prob[:,:,0,:]), torch.sum(rpn_cls_prob[:,:,1,:]))

        rpn_cls_prob = rpn_cls_prob.squeeze(3)  # (1, 9 * 14 * 14 , 2, 1)  => (1, 9 * 14 * 14 , 2)
        rpn_cls_prob = rpn_cls_prob.permute(0, 2, 1).contiguous()  # (1, 9 * 14 * 14 , 2) => (1, 2, 9 * 14 * 14)
        rpn_cls_prob = rpn_cls_prob.view(
            (-1, 18, features.size()[2], features.size()[3]))  # (1, 2, 9 * 14 * 14) => (1, 18, 14, 14)

        return rpn_bbox_pred, rpn_cls_prob


class ROIpooling(nn.Module):
    def __init__(self, size=(7, 7), spatial_scale=1.0 / 16.0):
        super(ROIpooling, self).__init__()
        self.adapmax2d = nn.AdaptiveMaxPool2d(size)
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):

        # features : torch.Size([1, 512, 14, 14])
        # rois : torch.Size([1764, 5])

        if type(rois) == np.ndarray:
            rois = to_var(torch.from_numpy(rois))

        rois = rois.data.float()
        #rois[:, 1:].mul_(self.spatial_scale)
        rois.mul_(self.spatial_scale)
        rois = rois.long()

        output = []
        # print(features.size(), rois.size())
        print("feature : {}".format(features.size()))
        for i in range(rois.size(0)):
            roi = rois[i]
            # im_idx = roi[0]

            # print(roi)
            # roi : (class, x, y, x`, y`)
            #roi_feature = features[:, :, roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
            try:

                roi_feature = features[:, :, roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
            except Exception as e:
                print(e, features, roi)

            pool_feature = self.adapmax2d(roi_feature)
            output.append(pool_feature)

        return torch.cat(output, 0)


class FasterRcnn(nn.Module):
    def __init__(self):
        super(FasterRcnn, self).__init__()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.classfier = nn.Linear(4096, 21)
        self.regressor = nn.Linear(4096, 21 * 4)

    def forward(self, features):
        features = features.view(-1, 512 * 7 * 7)
        features = self.fc1(features)
        features = self.fc2(features)
        return self.regressor(features), self.classfier(features)


if __name__ == '__main__':

    # CNN test

    test1 = to_var(torch.ones((1, 3, 1000, 600)))
    feature_extractor = CNN()
    features = feature_extractor.forward(test1)  # torch.Size([1, 512, 62, 37])
    print(features.size())

    # RPN test

    rpn = SimpleRpn()
    rpn_feature = rpn.forawrd(features)
    print(rpn_feature.size()) # torch.Size([1, 512, 62, 37])

    # RPN clasfier test

    rpn_classfier = RPNClassifier(512)
    rpn_bbox_pred, rpn_cls_prob = rpn_classfier(rpn_feature)
    print(rpn_bbox_pred.size(), rpn_cls_prob.size()) # torch.Size([1, 36, 62, 37]) torch.Size([1, 18, 62, 37])


    import torch
    from utils_.anchors import get_anchors, anchor
    from utils_.utils import to_var
    from proposal import ProposalLayer

    all_anchors = get_anchors(features, anchor)

    proplayer = ProposalLayer(rpn_bbox_pred, rpn_cls_prob, all_anchors, im_info=(600, 1000, 3, 10))
    proposals, scores = proplayer.proposal()
    print(len(proposals), len(scores))
    print(proposals.astype("int"))

    # ROIpooling test

    roipool = ROIpooling()
    proposals1 = proposals.copy()
    proposals1 = np.append(proposals[:, 0:1], proposals1, axis=1)
    rois_features = roipool(features, proposals1)
    print(rois_features.size())

    # FasterRCNN test

    fasterrcnn = FasterRcnn()
    bbox_pred, cls_score = fasterrcnn(rois_features)
    print(bbox_pred.size(), cls_score.size())
