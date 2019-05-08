import time
import code
import os, torch
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import optim
from data import *
from model import *
from train import *

os.makedirs('../data', exist_ok=True)
os.makedirs('../model', exist_ok=True)
os.makedirs('../results/', exist_ok=True)
os.makedirs('../results/predictions', exist_ok=True)
os.makedirs('../results/predictions/TP', exist_ok=True)
os.makedirs('../results/predictions/FP', exist_ok=True)
os.makedirs('../results/predictions/TN', exist_ok=True)
os.makedirs('../results/predictions/FN', exist_ok=True)
os.makedirs('../../../malaria/images', exist_ok=True)

def test(args, n_classes=1):
    print(args)
    if cuda:
        model = ClassifierModel(image_shape[0], n_classes).cuda()
        checkpoint = torch.load('../model/model1')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
    else:
        model = ClassifierModel(image_shape[0], n_classes)
        checkpoint = torch.load('../model/model1')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

    test_dataset = MalariaDataset(csv_file=test_csv,
                        root_dir=images_dir,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]),
                        binary=True, ds_type='test')
    # print('loaded dataset')
    test_loader = DataLoader(test_dataset, batch_size=500,
                            shuffle=False, num_workers=2)
    # print('loaded dataloader')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    print('Starting Testing.')
    Precision, Recall, F1, acc, Mcc, n = 0.0, 0.0, 0.0, 0.0, 0.0, len(test_loader)
    tp, fp, tn, fn = 0,0,0,0
    TP, FP, TN, FN = 0,0,0,0
    with torch.no_grad():
        for i, (img, labels, img_names) in enumerate(test_loader):
            img = Variable(img.type(Tensor), requires_grad=False)
            labels = Variable(labels.float(), requires_grad=False)
            if cuda:
                img = img.cuda()
                labels = labels.cuda()
            pred = model(img).view(-1)
            pred_labels = (pred >= args.thresh).long()
            if cuda:
                pred_labels = pred_labels.cuda()
            if args.save_pred:
                if min([tp,fp,tn,fn]) < args.num_pred:
                    if labels[0] == 1 and pred_labels[0] == 1 and tp < args.num_pred:
                        save_image(img.data[0], '../results/predictions/TP/'+img_names[0])
                        tp+=1
                        print('True Positive saved')
                    if labels[0] == 1 and pred_labels[0] == 0 and fn < args.num_pred:
                        save_image(img.data[0], '../results/predictions/FN/'+img_names[0])
                        fn+=1
                        print('False Negative saved')
                    if labels[0] == 0 and pred_labels[0] == 1 and fp < args.num_pred:
                        save_image(img.data[0], '../results/predictions/FP/'+img_names[0])
                        fp+=1
                        print('False Positive saved')
                    if labels[0] == 0 and pred_labels[0] == 0 and tn < args.num_pred:
                        save_image(img.data[0], '../results/predictions/TN/'+img_names[0])
                        tn+=1
                        print('True Negative saved')
                else:
                    print('Predictions saved.')
                    return None
            if not args.save_pred:
                precision, recall, f1, mcc, ac, (a, b, c, d) = accuracy(pred_labels, labels.long(), give_stat=True)
                Precision += precision
                Recall += recall
                F1 += f1
                Mcc += mcc
                acc += ac
                TP += a
                FP += b
                TN += c
                FN += d
                # print(i+1, len(test_loader))
    Precision, Recall, F1, Mcc, acc = Precision/n, Recall/n, F1/n, Mcc/n, acc/n
    print(args.thresh, '[test_accuracy - {0:.3f}, precision - {1:.6f}, recall - {2:.6f}, F1-score - {3:.6f}, MCC - {4:.6f}, TP - {5:.1f}, FP - {6:.1f}, TN - {7:.1f}, FN - {8:.1f},]'.format(acc, Precision, Recall, F1, Mcc, TP, FP, TN, FN))
    return Precision, Recall, F1, Mcc, acc

def validation(args, n_classes=1, split=0.8):
    print(args)
    if cuda:
        model = ClassifierModel(image_shape[0], n_classes).cuda()
        checkpoint = torch.load('../model/model1')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
    else:
        model = ClassifierModel(image_shape[0], n_classes)
        checkpoint = torch.load('../model/model1')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

    validation_dataset = MalariaDataset(csv_file=training_csv,
                        root_dir=images_dir,
                        split=split,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]),
                        binary=True, ds_type='valid')
    # print('loaded dataset')
    validation_loader = DataLoader(validation_dataset, batch_size=1000,
                            shuffle=False, num_workers=8)
    # print('loaded dataloader')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    print('Starting Validating.')
    Precision, Recall, F1, acc, Mcc, n = 0.0, 0.0, 0.0, 0.0, 0.0, len(validation_loader)
    tp, fp, tn, fn = 0,0,0,0
    TP, FP, TN, FN = 0,0,0,0
    with torch.no_grad():
        for i, (img, labels, img_names) in enumerate(validation_loader):
            img = Variable(img.type(Tensor), requires_grad=False)
            labels = Variable(labels.float(), requires_grad=False)
            if cuda:
                img = img.cuda()
                labels = labels.cuda()
            pred = model(img).view(-1)
            pred_labels = (pred >= args.thresh).long()
            if cuda:
                pred_labels = pred_labels.cuda()
            if args.save_pred:
                if min([tp,fp,tn,fn]) < args.num_pred:
                    if labels[0] == 1 and pred_labels[0] == 1 and tp < args.num_pred:
                        save_image(img.data[0], '../results/predictions/TP/'+img_names[0])
                        tp+=1
                        print('True Positive saved')
                    if labels[0] == 1 and pred_labels[0] == 0 and fn < args.num_pred:
                        save_image(img.data[0], '../results/predictions/FN/'+img_names[0])
                        fn+=1
                        print('False Negative saved')
                    if labels[0] == 0 and pred_labels[0] == 1 and fp < args.num_pred:
                        save_image(img.data[0], '../results/predictions/FP/'+img_names[0])
                        fp+=1
                        print('False Positive saved')
                    if labels[0] == 0 and pred_labels[0] == 0 and tn < args.num_pred:
                        save_image(img.data[0], '../results/predictions/TN/'+img_names[0])
                        tn+=1
                        print('True Negative saved')
                else:
                    print('Predictions saved.')
                    return None
            if not args.save_pred:
                precision, recall, f1, mcc, ac, (a, b, c, d) = accuracy(pred_labels, labels.long(), give_stat=True)
                Precision += precision
                Recall += recall
                F1 += f1
                Mcc += mcc
                acc += ac
                TP += a
                FP += b
                TN += c
                FN += d
                # print(i+1, len(test_loader))
    Precision, Recall, F1, Mcc, acc = Precision/n, Recall/n, F1/n, Mcc/n, acc/n
    print(args.thresh, '[validation_accuracy - {0:.3f}, precision - {1:.6f}, recall - {2:.6f}, F1-score - {3:.6f}, MCC - {4:.6f}, TP - {5:.1f}, FP - {6:.1f}, TN - {7:.1f}, FN - {8:.1f},]'.format(acc, Precision, Recall, F1, Mcc, TP, FP, TN, FN))
    return Precision, Recall, F1, Mcc, acc

def choose_hyperparam(args):
    args.thresh = 0.5
    while(args.thresh < 1):
        validation(args)
        args.thresh += 0.05
