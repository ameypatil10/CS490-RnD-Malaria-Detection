from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt
import pandas as pd
import scipy.misc
import json
from PIL import Image
from PIL import ImageFilter
import csv
from .util import read_image
import ast

def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=1200, max_size=1600):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=1200, max_size=1600):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale

def make_csv(json_file, csv_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    cnt = 0
    root_dir = '../../malaria/images'
    Data = []
    for img in data:
        img_name = img['image']['pathname'][7:]
        img_path = root_dir+img_name
        # print(img_path)
        bounding_boxes, labels = [], []
        for obj in img['objects']:
            bb = obj['bounding_box']
            bbox = (bb['minimum']['r'], bb['minimum']['c'], bb['maximum']['r'], bb['maximum']['c'])
            bounding_boxes.append(bbox)
            labels.append(int(obj['category'] != 'red blood cell'))
        Data.append([img_path, bounding_boxes, labels])
    df = pd.DataFrame(Data, columns=['img', 'bbox', 'label'])
    df.to_csv(csv_file)
    print('data written in csv')
        


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        # self.db = VOCBboxDataset(opt.voc_data_dir)
        make_csv('../../malaria/test.json', 'data/test.csv')
        self.data_frame = pd.read_csv('data/test.csv', header='infer')
        # print(self.data_frame)
        self.label_names = ['rbc', 'infected']
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label = self.data_frame.iloc[idx, 1:4] #self.db.get_example(idx)
        ori_img = read_image(ori_img, color=True)
        bbox = np.stack(ast.literal_eval(bbox)).astype(np.float32)
        label = np.stack(ast.literal_eval(label)).astype(np.int32)
        # print(ori_img, bbox, label)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.data_frame)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=False):
        self.opt = opt
        make_csv('../../malaria/test.json', 'data/test.csv')
        self.data_frame = pd.read_csv('data/test.csv', header='infer')
        # print(self.data_frame)
        self.label_names = ['rbc', 'infected']
        # self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        # ori_img, bbox, label, difficult = self.db.get_example(idx)
        ori_img, bbox, label = self.data_frame.iloc[idx, 1:4]
        ori_img = read_image(ori_img, color=True)
        bbox = np.stack(ast.literal_eval(bbox)).astype(np.float32)
        label = np.stack(ast.literal_eval(label)).astype(np.int32)
        img = preprocess(ori_img)
        difficult = np.array([False]*label.shape[0], dtype=np.bool).astype(np.uint8)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.data_frame)
