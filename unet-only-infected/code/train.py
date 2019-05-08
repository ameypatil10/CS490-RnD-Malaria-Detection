import time
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

os.makedirs('../data', exist_ok=True)
os.makedirs('../model', exist_ok=True)
os.makedirs('../results', exist_ok=True)

cuda = True if torch.cuda.is_available() else False

def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

def train(args):
    print(args)
    train_dataset = MalariaDataset(csv_file=train_csv,
                        img_dir=images_dir,
                        label_dir=infected_cell_label_dir,
                        img_size=(512,512),
                        dtype='train',
                        split=0.85,
                        img_transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]),
                        label_transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))
    validation_dataset = MalariaDataset(csv_file=train_csv,
                        img_dir=images_dir,
                        label_dir=infected_cell_label_dir,
                        img_size=(512,512),
                        dtype='valid',
                        split=0.85,
                        img_transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]),
                        label_transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))
    print('loaded dataset')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
    # print('loaded dataloader')
    if cuda:
        # criterion = torch.nn.BCELoss().cuda(args.gpu)
        model = UNet(3, 1).cuda(args.gpu)
    else:
        # criterion = torch.nn.BCELoss()
        model = UNet(3, 1)
        
    params_count = 0
    for param in model.parameters():
        params_count += np.prod(param.size())
    print('number of parameters = ', params_count)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.99,
                          weight_decay=0.0005)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    def validation(model):
        iou, acc = 0, 0
        for i, (img, label, img_name) in enumerate(validation_loader):
            img = Variable(img.type(Tensor), requires_grad=False)
            label = Variable(label.type(Tensor), requires_grad=False)
            if cuda:
                img.cuda(args.gpu)
                label.cuda(args.gpu)
            with torch.no_grad(): 
                pred = model(img)
                pred_th = (pred >= args.unet_thresh).float()
                stat = accuracy(pred_th, label)
                iou += stat[0]
                acc += stat[1]
        return (iou/len(validation_loader), acc/len(validation_loader))

    # loss, iou, accuracy
    train_log = np.zeros((3,1+int(args.n_epochs*len(train_loader)/args.batch_size)))
    # train_loss, train_iou, train_accuracy, test_iou, test_accuracy
    snap_log = np.zeros((5, 1+int(args.n_epochs*len(train_loader)/args.snap_interval)))

    print('Starting training.')
    batch, snap, batch_to_snap = 0, 0, 0
    start_time = time.time()
    last_print = 0
    for epoch in range(args.n_epochs):
        for i, (img, label, img_names) in enumerate(train_loader):
            # print(img.shape, label.shape)
            img = Variable(img.type(Tensor), requires_grad=False)
            label = Variable(label.type(Tensor), requires_grad=False)
            if cuda:
                img.cuda(args.gpu)
                label.cuda(args.gpu)
            pred = model(img)
            pos_weight = max(torch.mean(label).item(), 0.1)
            pos_weight = 8#0.5/pos_weight
            loss = weighted_binary_cross_entropy(pred, label, pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred_th = (pred >= args.unet_thresh).float()
            acc = accuracy(pred_th, label)
            train_log[0, batch] = loss.item()
            train_log[1, batch] = acc[0]
            train_log[2, batch] = acc[1]
            snap_log[0,snap] += loss.item()
            snap_log[1,snap] += acc[0]
            snap_log[2,snap] += acc[1]
            batch_to_snap += 1

            if batch % args.print_interval == 0 and batch > 0:
                print('[Epoch - {0:.1f}, batch - {1:.3f}, loss - {2:.6f}, train_iou - {3:.6f}, train_accuracy - {4:.6f}]'.format(1.0*epoch, i/len(train_loader), loss.item(), np.average(train_log[1, last_print:batch]), np.average(train_log[2, last_print:batch])))
                last_print = batch
            if batch % args.snap_interval == 0 and batch > 0:
                val = validation(model)
                snap_log[3, snap] = val[0]
                snap_log[4, snap] = val[1]
                snap_log[0:3,snap] /= batch_to_snap
                batch_to_snap = 0
                print('SNAP === [Epoch - {0:.1f}, Batch No - {1:.1f}, Snap No. - {2:.1f}, train_iou - {3:.6f}, train_accuracy - {4:.6f}, validation_iou - {5:.6f}, validation_accuracy - {6:.6f}] ==='\
                    .format(1.0*epoch, 1.0*batch, 1.0*snap, snap_log[1, snap], snap_log[2, snap], snap_log[3, snap], snap_log[4, snap]))
                snap += 1
            batch += 1
        np.savetxt('../results/train_log1.csv', train_log, delimiter=",")
        np.savetxt('../results/snap_log1.csv', snap_log, delimiter=",")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '../model/model3')

def accuracy(label, pred):
    with torch.no_grad():
        pred, label = pred.detach().cpu().numpy(), label.detach().cpu().numpy()
        intersection = np.logical_and(label, pred)
        union = np.logical_or(label, pred)
        if np.sum(union) == 0:
            iou_score = 1
        else:
            iou_score = np.sum(intersection) / np.sum(union)
        pixel_acc = np.average(label==pred)
        return (iou_score, pixel_acc)

def validation(args):
    print(args)
    if cuda:
        model = UNet(3, 1).cuda(args.gpu)
        checkpoint = torch.load('../model/model3')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
    else:
        model = UNet(3, 1)
        checkpoint = torch.load('../model/model3')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

    train_dataset = MalariaDataset(csv_file=train_csv,
                        img_dir=images_dir,
                        label_dir=infected_cell_label_dir,
                        dtype='valid',
                        split=0.85,
                        img_transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]),
                        label_transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))
    # print('loaded dataset')
    test_loader = DataLoader(train_dataset, batch_size=1,
                            shuffle=True, num_workers=8)
    # print('loaded dataloader')

    max_iou = 0
    opt_thresh = 0.5
    print('Starting validating.')
    for thresh in list(np.arange(0, 1, 0.1)):
        args.unet_thresh = thresh
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        iou_score_avg, pixel_acc_avg = 0, 0
        cnt = 0
        for i, (img, label, img_name) in enumerate(test_loader):
            img = Variable(img.type(Tensor), requires_grad=False)
            label = Variable(label.type(Tensor), requires_grad=False)
            if cuda:
                img.cuda(args.gpu)
                label.cuda(args.gpu)
            with torch.no_grad(): 
                pred = model(img)
                pred_th = (pred >= args.unet_thresh).float()
                stat = accuracy(pred_th, label)
                iou_score_avg += stat[0]
                pixel_acc_avg += stat[1]
                cnt += 1
        stat = iou_score_avg/cnt, pixel_acc_avg/cnt
        if stat[0] > max_iou:
           max_iou = stat[0]
           opt_thresh = thresh
        print(args.unet_thresh, '({0:.4f}, {1:.4f})'.format(stat[0], stat[1]))
    print('opt iou = ', max_iou, 'opt thresh = ' ,opt_thresh)
    return opt_thresh


def test(args):
    print(args)
    if cuda:
        model = UNet(3, 1).cuda(args.gpu)
        checkpoint = torch.load('../model/model3')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
    else:
        model = UNet(3, 1)
        checkpoint = torch.load('../model/model3')
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

    train_dataset = MalariaDataset(csv_file=test_csv,
                        img_dir=images_dir,
                        label_dir=infected_cell_label_dir,
                        dtype='test',
                        img_transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]),
                        label_transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))
    # print('loaded dataset')
    test_loader = DataLoader(train_dataset, batch_size=1,
                            shuffle=True, num_workers=8)
    # print('loaded dataloader')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    iou_score_avg, pixel_acc_avg = 0, 0
    cnt = 0
    print('Starting Testing.')
    for i, (img, label, img_name) in enumerate(test_loader):
        img = Variable(img.type(Tensor), requires_grad=False)
        label = Variable(label.type(Tensor), requires_grad=False)
        if cuda:
            img.cuda(args.gpu)
            label.cuda(args.gpu)
        with torch.no_grad(): 
            pred = model(img)
            pred_th = (pred >= args.unet_thresh).float()
            stat = accuracy(pred_th, label)
            iou_score_avg += stat[0]
            pixel_acc_avg += stat[1]
            if args.save_pred:
                os.makedirs('../results'+str(img_name[0][:-4]), exist_ok=True)
                os.system('cp '+images_dir+img_name[0]+' ../results'+str(img_name[0][:-4])+'/img.png')
                # save_image(img, '../results'+str(img_name[0][:-4])+'/img.png')
                save_image(label, '../results'+str(img_name[0][:-4])+'/label.png')
                save_image(pred, '../results'+str(img_name[0][:-4])+'/pred.png')
                save_image(pred_th, '../results'+str(img_name[0][:-4])+'/thpred.png')
                print('{0:.4f}, iou_score - {1:.4f}, accuracy - {2:.4f}'.format(i * args.batch_size / len(test_loader), stat[0], stat[1]))
            cnt += 1
            # if i > 20:
            #     break
    stat = iou_score_avg/cnt, pixel_acc_avg/cnt
    print('({0:.4f}, {1:.4f})'.format(stat[0], stat[1]))
