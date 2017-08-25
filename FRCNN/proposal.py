import numpy as np

from utils_.boxes_utils import bbox_transform_inv, py_cpu_nms, clip_boxes, filter_boxes

class ProposalLayer:
    def __init__(self, args):

        # rpn_bbox_pred : torch.Size([1, 36, 14, 14])
        # rpn_cls_prob : torch.Size([1, 18, 14, 14])
        # all_anchors_boxes : numpy.ndarray (1764, 4)
        # im_info : (H, W, C, scale)
        self.args = args


    def _get_pos_score(self, rpn_cls_prob):

        pos_scores = rpn_cls_prob[:, :9]  # (1, 9, 14, 14)
        pos_scores = pos_scores.permute(0, 2, 3, 1).contiguous()  # (1, 9, 14, 14) => (1, 14, 14, 9)
        pos_scores = pos_scores.view((-1, 1))  # (1, 14, 14, 9) => (1764, 1)

        return pos_scores

    def _get_bbox_deltas(self, rpn_bbox_pred):

        print("rpn_bbox_pred", rpn_bbox_pred.size())
        bbox_deltas = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # (1, 36, 14, 14) => (1, 14, 14, 36)
        bbox_deltas = bbox_deltas.view((-1, 4))  # (1, 14, 14, 36) => (1764, 4)
        print("bbox_deltas", bbox_deltas.size())
        return bbox_deltas

    def proposal(self, rpn_bbox_pred, rpn_cls_prob, all_anchors_boxes, im_info):
        """
        proposal operation in cpu
        """
        print(rpn_bbox_pred.size(), rpn_cls_prob.size())
        bbox_deltas = self._get_bbox_deltas(rpn_bbox_pred).data.numpy()

        # 1. Convert anchors into proposal via bbox transformation
        print(all_anchors_boxes.shape, bbox_deltas.shape)
        proposals_boxes = bbox_transform_inv(all_anchors_boxes, bbox_deltas)  # (1764, 4) all proposal boxes

        # 2. clip proposal boxes to image
        proposals_boxes = clip_boxes(proposals_boxes, im_info[0:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[3])
        filter_indices = filter_boxes(proposals_boxes, self.args.min_size * im_info[3])

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        pos_score = self._get_pos_score(rpn_cls_prob).data.numpy()

        # mask filter_indices False so other assign 0
        mask = np.ones(proposals_boxes.shape[0], dtype=bool)  # np.ones_like(a,dtype=bool)
        mask[filter_indices] = False
        pos_score[mask] = 0.0

        indices = np.argsort(pos_score.squeeze())

        # 5. take topn score proposal
        topn_indices = indices[:self.args.pre_nms_topn]
        print(indices, filter_indices, len(indices), len(filter_indices))

        # 6. apply nms (e.g. threshold = 0.7)
        proposals_boxes_c = np.hstack((pos_score[topn_indices], proposals_boxes[topn_indices]))  # (1000, 5)
        keep = py_cpu_nms(proposals_boxes_c, self.args.nms_thresh)

        # 7. take after_nms_topn (e.g. 300)
        if self.args.post_nms_topn > 0:
            keep = keep[:self.args.post_nms_topn]

        # 8. return the top proposals (-> RoIs top)
        proposals_boxes = proposals_boxes[keep, :]
        scores = pos_score[keep]

        return proposals_boxes, scores

if __name__ == '__main__':

    import torch
    from utils_.anchors import get_anchors, anchor
    from utils_ import to_var

    # torch.Size([1, 36, 62, 37]) torch.Size([1, 18, 62, 37])
    # rpn_bbox_pred, rpn_cls_prob = rpn_classfier(rpn_feature)
    rpn_bbox_pred = to_var(torch.randn((1, 36, 62, 37)))
    rpn_cls_prob = to_var(torch.randn((1, 18, 62, 37)))

    features = to_var(torch.randn((1, 512, 62, 37)))
    all_anchors_boxes = get_anchors(features, anchor)

    proplayer = ProposalLayer(rpn_bbox_pred, rpn_cls_prob, all_anchors_boxes, im_info=(600, 1000, 3, 10))
    proposals, scores = proplayer.proposal()
    print(len(proposals), len(scores))
    print(proposals.astype("int"))