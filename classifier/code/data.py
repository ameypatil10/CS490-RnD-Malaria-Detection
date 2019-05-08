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

classes = ['red blood cell', 'trophozoite', 'schizont', 'difficult', 'ring', 'leukocyte', 'gametocyte']
class_dict = {'red blood cell': 0, 'trophozoite': 1, 'schizont': 2, 'difficult': 3, 'ring': 4, 'leukocyte': 5, 'gametocyte': 6, 0: 0, 1: 1, '0': 0, '1': 1}

image_shape = (128, 128)
channels = 3

os.makedirs('../../../malaria/', exist_ok=True)
training_json = '../../../malaria/training.json'
test_json = '../../../malaria/training.json'
data_json = '../../../malaria/data.json'
images_dir = '../../../malaria/images/'

os.makedirs('../data/', exist_ok=True)
training_csv = '../data/training.csv'
test_csv = '../data/test.csv'

classes = ['red blood cell', 'trophozoite', 'schizont', 'difficult', 'ring', 'leukocyte', 'gametocyte']
class_dict = {'red blood cell': 0, 'trophozoite': 1, 'schizont': 2, 'difficult': 3, 'ring': 4, 'leukocyte': 5, 'gametocyte': 6}

def merge_json(training_json, test_json, data_json):
    with open(training_json) as f1:
        data1 = json.load(f1)
    with open(test_json) as f2:
        data2 = json.load(f2)
    

def data_csv(training_json, test_json, training_csv, test_csv, split=0.8):
    with open(training_json) as f1:
        data1 = json.load(f1)
    with open(test_json) as f2:
        data2 = json.load(f2)
    Data = []
    for item in data1:
        imagepath = item['image']['pathname'][8:]
        img_shape = item['image']['shape'] #(1200, 1600, 3)
        for cell in item['objects']:
            bb = cell['bounding_box']
            label = cell['category']
            bounding_box = (bb['minimum']['c'], bb['minimum']['r'], bb['maximum']['c'],  bb['maximum']['r'])
            line = [imagepath, bounding_box, class_dict[label]]
            Data.append(line)
    for item in data2:
        imagepath = item['image']['pathname'][8:]
        img_shape = item['image']['shape'] #(1200, 1600, 3)
        for cell in item['objects']:
            bb = cell['bounding_box']
            label = cell['category']
            bounding_box = (bb['minimum']['c'], bb['minimum']['r'], bb['maximum']['c'],  bb['maximum']['r'])
            line = [imagepath, bounding_box, class_dict[label]]
            Data.append(line)    
    headers = ['image path', '(minimum r, minimum c, maximum r, maximum c)', 'label']
    random.shuffle(Data)

    with open(training_csv, 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(headers)
        wr.writerows(Data[0:int(len(Data)*split)])
    print('Training dataset written into '+training_csv)
    with open(test_csv, 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(headers)
        wr.writerows(Data[int(len(Data)*split):])
    print('Test dataset written into '+training_csv)

# data_csv(training_json, test_json, training_csv, test_csv)

def label_weights(data, df=False):
    if not df:
        data = pd.read_csv(data)
    counts = list(data['label'].value_counts())
    weights = list(map(lambda x:x/sum(counts), counts))
    return counts, weights

def img_shape(csv_file):
    data = pd.read_csv(csv_file)
    shapes = list(data.iloc[:,1])
    xlens = np.array(list(map(lambda x: eval(x)[2]-eval(x)[0], shapes)))
    ylens = np.array(list(map(lambda x: eval(x)[3]-eval(x)[1], shapes)))
    print(np.mean(xlens), np.mean(ylens), np.median(xlens), np.median(ylens))
    
class MalariaDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, csv_file, root_dir, transform=None, header='infer', binary=True, ds_type=None, split=0.90, skip_rate=None, shift=False):
        'Initialization'
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file, header=header)
        self.transform = transform
        self.ds_type = ds_type
        self.split = split
        self.binary = binary
        self.skip_rate = skip_rate
        self.shift = shift
        if ds_type and ds_type == 'train':
            rows = np.arange(0,int(split*len(self.data_frame)))
            self.data_frame = self.data_frame.ix[rows]
        if ds_type and ds_type == 'valid':
            rows = np.arange(int(split*len(self.data_frame)), len(self.data_frame))
            self.data_frame = self.data_frame.ix[rows]
        self.count, self.label_weights = label_weights(self.data_frame, df=True)
        if binary:
            self.label_weights = [self.label_weights[0], sum(self.label_weights[1:])]
            self.count = [self.count[0], sum(self.count[1:])]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_frame)

  def __getitem__(self, index):
        'Generates one sample of data'
        if self.skip_rate and self.ds_type and self.ds_type == 'train':
            while(self.data_frame.iloc[index, -1] == 0 and random.random() < self.skip_rate):
                index = (index+1)%len(self.data_frame)
        label = self.data_frame.iloc[index, -1]
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[index, 0])
        image = Image.open(img_name)
        if self.binary:
            label = int(label != 0)
        bounding_box = eval(self.data_frame.iloc[index, 1])
        # print(image.size)
        if self.shift:
            # print('before - ', bounding_box)
            shiftx, shifty = random.randint(-image_shape[0]//3,image_shape[0]//3), random.randint(-image_shape[1]//3,image_shape[1]//3)
            bounding_box = list(bounding_box)
            bounding_box[0] = max(0, bounding_box[0]+shiftx)
            bounding_box[1] = max(0, bounding_box[1]+shifty)
            bounding_box[2] = min(image.size[0], bounding_box[2]-shiftx)
            bounding_box[3] = min(image.size[1], bounding_box[3]-shifty)
            bounding_box = tuple(bounding_box)
            # print('after - ', bounding_box)
        image = image.crop(bounding_box)
        image = image.resize(image_shape, Image.ANTIALIAS)
        if self.transform:
            image = self.transform(image)
        return (image, label, self.data_frame.iloc[index, 0])