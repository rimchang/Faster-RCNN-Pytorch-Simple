import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from utils_.utils import to_var


class CNN(nn.Module):

    def __init__(self):
        """Load the pretrained vgg11 and delete fc layer."""
        super(CNN, self).__init__()

        vggnet = models.vgg16(pretrained=True)
        modules = list(vggnet.children())[:-1]  # delete the last fc layer.
        modules = list(modules[0])[:-1]  # delete the last pooling layer

        self.vggnet = nn.Sequential(*modules)

        for module in list(self.vggnet.children())[:10]:
            print("fix weight", module)
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, images):
        """Extract the image feature vectors."""

        # return features in relu5_3
        features = self.vggnet(images)
        return features



class RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
                                            nn.ReLU())

        # 9 anchor * 2 classfier (object or non-object) each grid
        self.conv1 = nn.Conv2d(512, 2 * 9, kernel_size=1, stride=1)

        # 9 anchor * 4 coordinate regressor each grids
        self.conv2 = nn.Conv2d(512, 4 * 9, kernel_size=1, stride=1)
        self.softmax = nn.Softmax()

    def forward(self, features):

        features = self.conv(features)

        logits, rpn_bbox_pred = self.conv1(features), self.conv2(features)

        height, width = features.size()[-2:]
        logits = logits.squeeze(0).permute(1, 2, 0).contiguous()  # (1, 18, H/16, W/16) => (H/16 ,W/16, 18)
        logits = logits.view(-1, 2)  # (H/16 ,W/16, 18) => (H/16 * W/16 * 9, 2)

        rpn_cls_prob = self.softmax(logits)
        rpn_cls_prob = rpn_cls_prob.view(height, width, 18)  # (H/16 * W/16 * 9, 2)  => (H/16 ,W/16, 18)
        rpn_cls_prob = rpn_cls_prob.permute(2, 0, 1).contiguous().unsqueeze(0) # (H/16 ,W/16, 18) => (1, 18, H/16, W/16)

        return rpn_bbox_pred, rpn_cls_prob, logits


class ROIpooling(nn.Module):

    def __init__(self, size=(7, 7), spatial_scale=1.0 / 16.0):
        super(ROIpooling, self).__init__()
        self.adapmax2d = nn.AdaptiveMaxPool2d(size)
        self.spatial_scale = spatial_scale

    def forward(self, features, rois_boxes):

        # rois_boxes : [x, y, x`, y`]

        if type(rois_boxes) == np.ndarray:
            rois_boxes = to_var(torch.from_numpy(rois_boxes))

        rois_boxes = rois_boxes.data.float().clone()
        rois_boxes.mul_(self.spatial_scale)
        rois_boxes = rois_boxes.long()

        output = []

        for i in range(rois_boxes.size(0)):
            roi = rois_boxes[i]

            try:

                roi_feature = features[:, :, roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
            except Exception as e:
                print(e, roi)


            pool_feature = self.adapmax2d(roi_feature)
            output.append(pool_feature)

        return torch.cat(output, 0)


class FasterRcnn(nn.Module):

    def __init__(self):
        super(FasterRcnn, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                 nn.ReLU(),
                                 nn.Dropout())

        self.fc2 = nn.Sequential(nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout())

        # 20 class + 1 backround classfier each roi
        self.classfier = nn.Linear(4096, 21)
        self.softmax = nn.Softmax()

        # 21 class * 4 coordinate regressor each roi
        self.regressor = nn.Linear(4096, 21 * 4)

    def forward(self, features):

        features = features.view(-1, 512 * 7 * 7)
        features = self.fc1(features)
        features = self.fc2(features)

        try:
            logits = self.classfier(features)
            scores = self.softmax(logits)
            bbox_delta = self.regressor(features)

        except Exception as e:
            print(e, logits)

        return bbox_delta, scores, logits


