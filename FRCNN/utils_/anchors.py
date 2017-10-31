import numpy as np

# for simplified using constant anchor reference
# base_size=16, ratios=[0.5, 1, 2], scales=np.array(8, 16, 32)
anchor = np.array([
 [ -84,  -40,   99,   55],
 [-176,  -88,  191,  103],
 [-360, -184,  375,  199],
 [ -56,  -56,   71,   71],
 [-120, -120,  135,  135],
 [-248, -248,  263,  263],
 [ -36,  -80,   51,   95],
 [ -80, -168,   95,  183],
 [-168, -344,  183,  359],
],dtype=np.float32)

def get_anchors(feature, anchor, feat_stride=16):

    height, width = feature.size()[-2:]  # H/16, W/16

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride

    # np.meshgrid : Return coordinate matrices from coordinate vectors.
    # vector 2개를 통해 grid를 만들어주는 함수
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # (H/16, W/16) 의 grid point들 (이미지 사이즈에 맞게 scale 되어져있다)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors

    A = anchor.shape[0]  # maybe 9
    K = shifts.shape[0]  # maybe H/16 * W/16

    # (1, 9, 4) + (H/16 * W/16, 1, 4) => (H/16 * W/16, 9, 4) 브로드캐스트되어서 더해진다.
    # 즉 기준점이되는 reference anchor에서부터 shifted anchor를 더해주는것.
    all_anchors = (anchor.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))

    # row order : H/16, W/16, A
    # 총 H/16 * W/16 * 9 개의 anchor
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors
