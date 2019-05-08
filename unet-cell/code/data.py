from __future__ import print_function, division
import os
import json
import csv
import torch
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from PIL import ImageFilter
import scipy.misc
from sklearn.utils import shuffle

classes = ['red blood cell', 'trophozoite', 'schizont', 'difficult', 'ring', 'leukocyte', 'gametocyte']
class_dict = {'red blood cell': 0, 'trophozoite': 1, 'schizont': 2, 'difficult': 3, 'ring': 4, 'leukocyte': 5, 'gametocyte': 6, 0: 0, 1: 1, '0': 0, '1': 1}

image_shape = (128, 128)
channels = 3

os.makedirs('../../../malaria/', exist_ok=True)
training_json = '../../../malaria/training.json'
test_json = '../../../malaria/training.json'
data_json = '../../../malaria/data.json'

images_dir = '../../../malaria/images'
cell_label_dir = '../../../malaria/cell-labels'
infected_cell_label_dir = '../../../malaria/infected-cell-labels'
multi_label_dir = '../../../malaria/multi-cell-labels'
os.makedirs(images_dir, exist_ok=True)
os.makedirs(cell_label_dir, exist_ok=True)
os.makedirs(infected_cell_label_dir, exist_ok=True)
os.makedirs(multi_label_dir, exist_ok=True)

os.makedirs('../data/', exist_ok=True)
train_file = '../data/training.json'
test_file = '../data/test.json'
data_file = '../data/data.json'

train_csv = '../data/train.csv'
test_csv = '../data/test.csv'
data_csv = '../data/data.csv'

classes = ['red blood cell', 'trophozoite', 'schizont', 'difficult', 'ring', 'leukocyte', 'gametocyte']
class_dict = {'red blood cell': 0, 'trophozoite': 1, 'schizont': 2, 'difficult': 3, 'ring': 4, 'leukocyte': 5, 'gametocyte': 6}

def make_labels(root_dir, target_dir, inp):
    with open(inp, 'r') as f:
        data = json.load(f)
    cnt = 0
    print('Saving labels..')
    for img in data:
        img_name = img['image']['pathname'][7:]
        img_path = root_dir+img_name
        # print(img_path)
        im = Image.open(img_path)
        (x,y) = im.size
        label = np.zeros((y,x))
        # print(pixels.shape, np.max(pixels), np.min(pixels))
        for obj in img['objects']:
            bb = obj['bounding_box']
            (x1,y1,x2,y2) = (bb['minimum']['r'], bb['minimum']['c'], bb['maximum']['r'], bb['maximum']['c'])
            label[x1:x2, y1:y2] = 1
        scipy.misc.imsave(target_dir+img_name, label)
        cnt += 1
        # print(target_dir+img_name)
    print('Done saving '+str(cnt)+' all-cell-labels')

# make_labels(images_dir, cell_label_dir, data_file)

def make_only_infected_labels(root_dir, target_dir, inp):
    with open(inp, 'r') as f:
        data = json.load(f)
    cnt = 0
    print('Saving labels..')
    for img in data:
        img_name = img['image']['pathname'][7:]
        img_path = root_dir+img_name
        # print(img_path)
        im = Image.open(img_path)
        (x,y) = im.size
        label = np.zeros((y,x))
        # print(pixels.shape, np.max(pixels), np.min(pixels))
        for obj in img['objects']:
            bb = obj['bounding_box']
            (x1,y1,x2,y2) = (bb['minimum']['r'], bb['minimum']['c'], bb['maximum']['r'], bb['maximum']['c'])
            if obj['category'] != "red blood cell":
                label[x1:x2, y1:y2] = 1
        scipy.misc.imsave(target_dir+img_name, label)
        cnt += 1
        # print(target_dir+img_name)
    print('Done saving '+str(cnt)+' only-infected-cell-labels')

# make_only_infected_labels(images_dir, infected_cell_label_dir, data_file)

def make_multi_labels(root_dir, target_dir, inp):
    with open(inp, 'r') as f:
        data = json.load(f)
    cnt = 0
    print('Saving labels..')
    for img in data:
        img_name = img['image']['pathname'][7:]
        img_path = root_dir+img_name
        # print(img_path)
        im = Image.open(img_path)
        (x,y) = im.size
        label = np.zeros((y,x))
        # print(pixels.shape, np.max(pixels), np.min(pixels))
        for obj in img['objects']:
            bb = obj['bounding_box']
            (x1,y1,x2,y2) = (bb['minimum']['r'], bb['minimum']['c'], bb['maximum']['r'], bb['maximum']['c'])
            if obj['category'] != "red blood cell":
                label[x1:x2, y1:y2] = 1
            else:
                label[x1:x2, y1:y2] = 0.5
        scipy.misc.imsave(target_dir+img_name, label)
        cnt += 1
        # print(target_dir+img_name)
    print('Done saving '+str(cnt)+' multi-cell-labels')

# make_multi_labels(images_dir, multi_label_dir, data_file)

class MalariaDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, csv_file, img_dir, label_dir, img_size=None, img_transform=None, label_transform=None, header='infer', dtype='train', split=0.9):
        'Initialization'
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.data_frame = pd.read_csv(csv_file, header=header)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.dtype = dtype
        if self.dtype not in ['train', 'valid', 'test']:
            raise Exception('dtype must be \'train\' or \'valid\' or \'test\'')
        self.split = split
        self.validation_index = int(split*len(self.data_frame))

  def __len__(self):
        'Denotes the total number of samples'
        if self.dtype == 'test':
            return len(self.data_frame)
        elif self.dtype == 'train':
            return self.validation_index
        elif self.dtype == 'valid':
            return len(self.data_frame) - self.validation_index

  def __getitem__(self, index):
        'Generates one sample of data'
        if self.dtype == 'valid':
            index += self.validation_index
        img_name = self.data_frame.iloc[index, 0]
        img_path = self.img_dir + img_name
        image = Image.open(img_path)
        label_path = self.label_dir + img_name
        label = Image.open(label_path)
        if self.img_size:
            xl,yl = image.size
            (x,y) = random.randint(0,xl-self.img_size[0]),random.randint(0,yl-self.img_size[1])
            image = image.crop((x,y,x+self.img_size[0],y+self.img_size[1]))
            label = label.crop((x,y,x+self.img_size[0],y+self.img_size[1]))
        if self.img_transform:
            image = self.img_transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return (image, label, img_name)
