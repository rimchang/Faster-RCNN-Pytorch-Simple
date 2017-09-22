import numpy as np

from utils_.boxes_utils import bbox_transform_inv, py_cpu_nms, clip_boxes, filter_boxes

class ProposalLayer:
    def __init__(self, args):

        self.args = args


    def _get_pos_score(self, rpn_cls_prob):

        pos_scores = rpn_cls_prob[:, :9]  # (1, 9, H/16, W/16)
        pos_scores = pos_scores.permute(0, 2, 3, 1).contiguous()  # (1, 9, H/16, W/16) => (1, H/16, W/16, 9)
        pos_scores = pos_scores.view((-1, 1))  # (1, H/16, W/16, 9) => (H/16 * W/16, 1)

        return pos_scores

    def _get_bbox_deltas(self, rpn_bbox_pred):


        bbox_deltas = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # (1, 36, H/16, W/16) => (1, H/16, W/16, 36)
        bbox_deltas = bbox_deltas.view((-1, 4))  # (1, H/16, W/16, 36) => (H/16 * W/16, 4)

        return bbox_deltas

    def proposal(self, rpn_bbox_pred, rpn_cls_prob, all_anchors_boxes, im_info, test, args):
        """
        Arguments:
            rpn_bbox_pred (Tensor) : (1, 4*9, H/16, W/16)
            rpn_cls_prob (Tensor) : (1, 2*9, H/16, W/16)
            all_anchors_boxes (Ndarray) : (9 * H/16 * W/16, 4) predicted boxes
            im_info (Tuple) : (Height, Width, Channel, Scale)
            test (Bool) : True or False
            args (argparse.Namespace) : global arguments

        Return:

            # in each minibatch number of proposal boxes is variable
            proposals_boxes (Ndarray) : ( # proposal boxes, 4)
            scores (Ndarray) :  ( # proposal boxes, )
        """

        # if test == False, using training args else using testing args
        pre_nms_topn = args.pre_nms_topn if test == False else args.test_pre_nms_topn
        nms_thresh = args.nms_thresh if test == False else args.test_nms_thresh
        post_nms_topn = args.post_nms_topn if test == False else args.test_post_nms_topn


        bbox_deltas = self._get_bbox_deltas(rpn_bbox_pred).data.cpu().numpy()

        # 1. Convert anchors into proposal via bbox transformation
        proposals_boxes = bbox_transform_inv(all_anchors_boxes, bbox_deltas)  # (H/16 * W/16, 4) all proposal boxes
        pos_score = self._get_pos_score(rpn_cls_prob).data.cpu().numpy()


        height, width = im_info[0:2]


        # if test==True, keep anchors inside the image
        # if test==False, delete anchors inside the image
        if test == False:
            _allowed_border = 0
            inds_inside = np.where(
                (all_anchors_boxes[:, 0] >= -_allowed_border) &
                (all_anchors_boxes[:, 1] >= -_allowed_border) &
                (all_anchors_boxes[:, 2] < width + _allowed_border) &  # width
                (all_anchors_boxes[:, 3] < height + _allowed_border)  # height
            )[0]

            mask = np.zeros(proposals_boxes.shape[0], dtype=bool)
            mask[inds_inside] = True

            proposals_boxes = proposals_boxes[mask]
            pos_score = pos_score[mask]

        # 2. clip proposal boxes to image
        proposals_boxes = clip_boxes(proposals_boxes, im_info[0:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[3])
        filter_indices = filter_boxes(proposals_boxes, self.args.min_size * max(im_info[3]))

        # delete filter_indices
        mask = np.zeros(proposals_boxes.shape[0], dtype=bool)
        mask[filter_indices] = True

        proposals_boxes = proposals_boxes[mask]
        pos_score = pos_score[mask]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        indices = np.argsort(pos_score.squeeze())[::-1] # descent order


        # 5. take topn score proposal
        topn_indices = indices[:pre_nms_topn]


        # 6. apply nms (e.g. threshold = 0.7)
        # proposals_boxes_c : [x, y, x`, y`, class]
        proposals_boxes_c = np.hstack((proposals_boxes[topn_indices], pos_score[topn_indices]))
        keep = py_cpu_nms(proposals_boxes_c, nms_thresh)

        # 7. take after_nms_topn (e.g. 300)
        if post_nms_topn > 0:
            keep = keep[:post_nms_topn]


        # 8. return the top proposals (-> RoIs top)
        proposals_boxes = proposals_boxes_c[keep, :-1]
        scores = proposals_boxes_c[keep, -1]

        return proposals_boxes, scores
