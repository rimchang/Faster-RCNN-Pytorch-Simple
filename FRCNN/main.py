import argparse

from run.train import train
from run.make_val_boxes import make_val_boxes
from run.eval import evaluation


def main(args):


    if args.train:
        train(args)

    if args.make_val_boxes:
        make_val_boxes(args)

    if args.test:
        evaluation(args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # other parameters
    parser.add_argument('--model_name', type=str, default="C3",
                        help='this model_name is used for naming directory name')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='train')
    parser.add_argument('--make_val_boxes', type=str2bool, default=True,
                        help='if this True, excute make_val_boxes after training ')
    parser.add_argument('--test', type=str2bool, default=True,
                        help='if this True, excute test after training ')
    parser.add_argument('--use_tensorboard', type=str2bool, default=True,
                        help='using tensorboard logging')

    parser.add_argument('--test_max_per_image', type=int, default=10,
                        help='max per image for test time')
    parser.add_argument('--test_ob_thresh', type=int, default=0.05,
                        help='class threshhold for test')
    parser.add_argument('--frcnn_nms', type=int, default=0.3,
                        help='frcnn nms thresh for last boxes')


    # proposal layer args
    proposal_layer = parser.add_argument_group('proposal_layer')
    proposal_layer.add_argument('--min_size', type=int, default=10,
                        help='minimum proposal region size')
    proposal_layer.add_argument('--pre_nms_topn', type=float, default=12000,
                        help='proposal region topn filter before nms')
    proposal_layer.add_argument('--post_nms_topn', type=float, default=300,
                        help='proposal region topn filter after nms')
    proposal_layer.add_argument('--nms_thresh', type=float, default=0.7,
                        help='IOU nms thresholds')

    proposal_layer.add_argument('--test_pre_nms_topn', type=float, default=6000,
                        help='proposal region topn filter before nms')
    proposal_layer.add_argument('--test_post_nms_topn', type=float, default=300,
                        help='proposal region topn filter after nms')
    proposal_layer.add_argument('--test_nms_thresh', type=float, default=0.7,
                        help='IOU nms thresholds')

    # rpn_targets args
    rpn_targets = parser.add_argument_group('rpn_targets')
    rpn_targets.add_argument('--neg_threshold', type=float, default=0.3,
                        help='negative sample thresholds')
    rpn_targets.add_argument('--pos_threshold', type=float, default=0.7,
                        help='positive sample thresholds')
    rpn_targets.add_argument('--rpn_batch_size', type=int, default=256,
                        help='mini batch size for rpn')

    # frcnn_targets args
    frcnn_targets = parser.add_argument_group('frcnn_targets')
    frcnn_targets.add_argument('--fg_fraction', type=float, default=0.25,
                        help='foreground fraction')
    frcnn_targets.add_argument('--fg_threshold', type=float, default=0.5,
                        help='foreground object thresholds')
    frcnn_targets.add_argument('--bg_threshold', type=tuple, default=(0.1, 0.5),
                        help='background object thresholds')
    frcnn_targets.add_argument('--include_gt', type=str2bool, default=True,
                        help='include ground truth box in frcnn_targets')
    frcnn_targets.add_argument('--frcnn_batch_size', type=int, default=256,
                        help='mini batch size for frcnn')


    training = parser.add_argument_group('training')
    training.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    training.add_argument('--ft_lr', type=float, default=0.001,
                        help='little low lr for fine tuning')
    training.add_argument('--ft_step', type=int, default=2,
                        help='at ft_step epoch small lr increase in finetuning CNN')
    training.add_argument('--lr_step', type=tuple, default=None,
                        help='at each lr_step epoch lr decrease 0.1')
    training.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    training.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay')
    training.add_argument('--num_printobj', type=int, default=10,
                        help='object print number')
    training.add_argument('--init_gaussian', type=str2bool, default=True,
                        help='initialize weight with gaussian N(0,0.01)')
    training.add_argument('--ft_conv3', type=str2bool, default=True,
                        help='fine tuning after conv3 in vggnet')

    # Model Parmeters
    training.add_argument('--n_epochs', type=float, default=4,
                        help='max epochs')



    # dir parameters
    other = parser.add_argument_group('other')
    other.add_argument('--output_dir', type=str, default="../output",
                        help='output path')
    other.add_argument('--input_dir', type=str, default='../input',
                        help='input path')
    other.add_argument('--pickle_dir', type=str, default='/pickle',
                        help='input path')
    other.add_argument('--result_dir', type=str, default='/result',
                        help='input path')
    other.add_argument('--log_dir', type=str, default='/log',
                        help='for tensorboard log path save in output_dir + log_dir')
    other.add_argument('--image_dir', type=str, default='/image',
                        help='for output image path save in output_dir + image_dir')

    # step parameter
    other.add_argument('--pickle_step', type=int, default=4,
                        help='pickle save at pickle_step epoch')
    other.add_argument('--log_step', type=int, default=20,
                        help='tensorboard log save and print log at log_step epoch')
    other.add_argument('--image_save_step', type=int, default=100,
                        help='output image save at image_save_step iteration')



    args = parser.parse_args()


    main(args)