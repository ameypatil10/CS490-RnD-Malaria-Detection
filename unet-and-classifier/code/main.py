import argparse
from test import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=500, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_classes', type=int, default=1, help='number of classes for dataset')
parser.add_argument('--model_interval', type=int, default=1, help='model save interval')
parser.add_argument('--print_interval', type=int, default=400, help='print interval')
parser.add_argument('--snap_interval', type=int, default=2000, help='batch train log interval')
parser.add_argument('--unet_thresh', type=float, default=0.5, help='Threshold for predictions')
parser.add_argument('--classifier_thresh', type=float, default=0.9, help='Threshold for classifier')
parser.add_argument('--save_pred', type=bool, default=True, help='Save predictions')
parser.add_argument('--gpu', type=int, default=0, help='GPU number')

args = parser.parse_args()

# test(args)

classifier_loop(args)
