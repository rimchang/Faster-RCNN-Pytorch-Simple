import numpy as np

# for simplified using constant anchor reference
# base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)
anchor = np.array([[-83., -39., 100., 56.],
                   [-175., -87., 192., 104.],
                   [-359., -183., 376., 200.],
                   [-55., -55., 72., 72.],
                   [-119., -119., 136., 136.],
                   [-247., -247., 264., 264.],
                   [-35., -79., 52., 96.],
                   [-79., -167., 96., 184.],
                   [-167., -343., 184., 360.]])


def get_anchors(feature, anchor, feat_stride=16):
    # feature.size() maybe (1, 512, 14, 14)
    height, width = feature.size()[-2:]  # 14, 14

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride

    # np.meshgrid : Return coordinate matrices from coordinate vectors.
    # vector 2개를 통해 grid를 만들어주는 함수
    # [  0  44  88 132 176] [  0  44  88 132 176] 2개의 vector를 통해 grid를 만들어준다.
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # (14, 4) 의 grid point들 (이미지 사이즈에 맞게  extend 되어져있다)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors

    A = anchor.shape[0]  # maybe 9
    K = shifts.shape[0]  # maybe 196 why? 14*14 features

    # (1, 9, 4) + (196, 1, 4) => (196, 9, 4) 브로드캐스트되어서 더해진다.
    # 즉 기준점이되는 reference anchor에서부터 shifted anchor를 더해주는것.
    all_anchors = (anchor.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))

    # 196 * 9 개의 anchor (1794, 4)
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors

if __name__ == '__main__':

    import torch

    # features = feature_extractor.forward(test1)  # torch.Size([1, 512, 62, 37])

    # this features are fake features
    features = torch.randn((1, 512, 62, 37))
    all_anchors = get_anchors(features, anchor)
    print(all_anchors.shape)