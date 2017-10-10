
import torch
import torch.nn as nn
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class VGG16(nn.Module):
    def __init__(self, bn=False):
        super(VGG16, self).__init__()

        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
                                   Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))


        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.conv2.parameters():
            param.requires_grad = False

        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn))

    def forward(self, im_data):
        # im_data, im_scales = get_blobs(image)
        # im_info = np.array(
        #     [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        #     dtype=np.float32)
        # data = Variable(torch.from_numpy(im_data)).cuda()
        # x = data.permute(0, 3, 1, 2)

        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        own_dict = self.state_dict()
        for name, val in own_dict.items():
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}/{}:0'.format(i, j, ptype)
            param = torch.from_numpy(params[key])
            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)
            val.copy_(param)

    def load_from_npy_file(self, fname):
        print("load pre-trained model from",fname)
        own_dict = self.state_dict()
        params = np.load(fname, encoding = 'latin1').item()
        for name, val in own_dict.items():
            # # print name
            # # print val.size()
            # # print param.size()
            # if name.find('bn.') >= 0:
            #     continue

            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}'.format(i, j)
            param = torch.from_numpy(params[key][ptype])

            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)

            val.copy_(param)


if __name__ == '__main__':
    vgg = VGG16()
    vgg.load_from_npy_file('../input/pretrained_model/VGG_imagenet.npy')
    print(vgg)