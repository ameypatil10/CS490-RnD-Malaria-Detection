import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class ClassifierModel(nn.Module):
    def __init__(self, img_size, n_classes):
        super(ClassifierModel, self).__init__()

        self.img_size = img_size
        self.n_classes = n_classes

        def model_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.35)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *model_block(3, 16, bn=False),
            torch.nn.MaxPool2d(2),
            *model_block(16, 32),
            torch.nn.MaxPool2d(2),
            *model_block(32, 64),
        )

        # The height and width of downsampled image
        ds_size = self.img_size // 2**5
        self.adv_layer = nn.Sequential( nn.Linear(64*ds_size**2, self.n_classes),
                                        nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity