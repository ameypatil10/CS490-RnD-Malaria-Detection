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

os.makedirs('../data', exist_ok=True)
os.makedirs('../model', exist_ok=True)
os.makedirs('../results', exist_ok=True)
os.makedirs('../../../malaria/images', exist_ok=True)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def train(args, resume=False, n_classes=1):
    print(args)
    train_dataset = MalariaDataset(csv_file=training_csv,
                        root_dir=images_dir,
                        split=0.90,
                        shift=True,
                        transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]),
                        binary=True, ds_type='train', skip_rate=args.lambda2)
    validation_dataset = MalariaDataset(csv_file=training_csv,
                        root_dir=images_dir,
                        split=0.90,
                        shift=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]),
                        binary=True, ds_type='valid')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2)
    print('loaded train data - '+str(len(train_loader)))
    cnts,wts = label_weights(training_csv)
    if n_classes == 1:
        wts = [wts[0], sum(wts[1:])]
        cnts = [cnts[0], sum(cnts[1:])]
    print(wts)
    wts1 = list(map(lambda x: 1/x, wts))
    wts1 = list(map(lambda x: x/sum(wts1), wts1))
    class_weights = torch.FloatTensor(wts1)
    print(class_weights)
    if cuda:
        # criterion = torch.nn.BCEWithLogitsLoss().cuda(args.gpu)
        criterion = torch.nn.BCELoss().cuda(args.gpu)
        model = ClassifierModel(image_shape[0], n_classes).cuda(args.gpu)
        class_weights = class_weights.cuda(args.gpu)
    else:
        # criterion = torch.nn.BCEWithLogitsLoss()
        criterion = torch.nn.BCELoss()
        model = ClassifierModel(image_shape[0], n_classes)
        
    params_count = 0
    for param in model.parameters():
        params_count += np.prod(param.size())
    print('number of parameters = ', params_count)
    
    model.apply(weights_init_normal)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def validation(model):
        Precision, Recall, F1, acc, Mcc, n = 0.0, 0.0, 0.0, 0.0, 0.0, len(validation_loader)
        print('Validating (of '+str(len(validation_loader))+") .. ")
        with torch.no_grad():
            for i, (img, labels, img_names) in enumerate(validation_loader):
                img = Variable(img.type(Tensor), requires_grad=False)
                labels = Variable(labels.float(), requires_grad=False)
                if cuda:
                    img = img.cuda(args.gpu)
                    labels = labels.cuda(args.gpu)
                pred = model(img).view(-1)
                pred_labels = (pred >= args.thresh).long()
                if cuda:
                    pred_labels = pred_labels.cuda(args.gpu)
                precision, recall, f1, mcc, ac = accuracy(pred_labels, labels.long(), gpu=args.gpu)
                Precision += precision
                Recall += recall
                F1 += f1
                Mcc += mcc
                acc += ac
        return Precision/n, Recall/n, F1/n, Mcc/n, acc/n

    # loss, precision, recall, f1, mcc, accuracy
    train_log = np.zeros((6, 1+int(args.n_epochs*len(train_loader))))
    # train_loss, precision, recall, f1, mcc, train_accuracy, precision, recall, f1, mcc, validation_accuracy
    snap_log = np.zeros((11, 1+int(args.n_epochs*len(train_loader)/args.snap_interval)))

    print('Starting training..')
    batch, snap, batch_to_snap = 0, 0, 0
    n_one = 0
    start_time = time.time()
    for epoch in range(args.n_epochs):
        for i, (img, labels, img_names) in enumerate(train_loader):
            img = Variable(img.type(Tensor), requires_grad=False)
            labels = Variable(labels.float(), requires_grad=False)
            if cuda:
                img = img.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
            pred = model(img).view(-1)
            loss = criterion(pred, labels)
            # loss = weighted_binary_cross_entropy(pred, labels, [args.lambda1, 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_labels = (pred >= args.thresh).long()
            if cuda:
                pred_labels = pred_labels.cuda(args.gpu)
            precision, recall, f1, mcc, acc = accuracy(pred_labels, labels.long(), gpu=args.gpu)
            train_log[0, batch] = loss.item()
            train_log[1, batch] = precision
            train_log[2, batch] = recall
            train_log[3, batch] = f1
            train_log[4, batch] = mcc
            train_log[5, batch] = acc
            snap_log[0,snap] += loss.item()
            snap_log[1,snap] += precision
            snap_log[2,snap] += recall
            snap_log[3,snap] += f1
            snap_log[4,snap] += mcc
            snap_log[5,snap] += acc
            batch_to_snap += 1
            n_one += torch.sum(labels).item()

            if batch % args.print_interval == 0 and batch > 0:
                print('[Epoch - {0:.1f}, batch - {1:.3f}, loss - {2:.6f}, train_accuracy - {3:.3f}, precision - {4:.6f}, recall - {5:.6f}, F1-score - {6:.6f}, MCC - {7:.6f}, %+ve labels- {8:.2f}]'.format(1.0*epoch, 100.0*i/len(train_loader), loss.item(), train_log[4, batch], train_log[1, batch], train_log[2, batch], train_log[3, batch], train_log[4, batch], n_one/(args.batch_size*args.print_interval)))
                n_one = 0
            if batch % args.snap_interval == 0 and batch > 0:
                Precision, Recall, F1, Mcc, val = validation(model)
                snap_log[6, snap] = Precision
                snap_log[7, snap] = Recall
                snap_log[8, snap] = F1
                snap_log[9, snap] = Mcc
                snap_log[10, snap] = val
                snap_log[0:6,snap] /= batch_to_snap
                batch_to_snap = 0
                print('SNAP -- {0:.3f} === [Epoch - {1:.1f}, Batch No - {2:.1f}, Snap No. - {3:.1f}, train_accuracy - {4:.3f}, precision - {5:.6f}, recall - {6:.6f}, F1-score - {7:.6f}, MCC - {8:.6f},\n validation_accuracy - {9:.3f}, val_precision - {10:.6f}, val_recall - {11:.6f}, val_F1-score - {12:.6f}, MCC - {13:.6f},] ==='\
                    .format(time.time()-start_time, 1.0*epoch, 1.0*batch, 1.0*snap, snap_log[5, snap], snap_log[1, snap], snap_log[2, snap], snap_log[3, snap], snap_log[4, snap], snap_log[10, snap], snap_log[6, snap], snap_log[7, snap], snap_log[8, snap], snap_log[9, snap]))
                snap += 1
                start_time = time.time()
            batch += 1
        np.savetxt('../results/train_log1.csv', train_log, delimiter=",")
        np.savetxt('../results/snap_log1.csv', snap_log, delimiter=",")
        # torch.save(model, '../model-multi/model')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '../model/model1')

def accuracy(pred, labels, class_weights=None, give_stat=False, gpu=0):
    def one_hot(y):
        if cuda:
            y_onehot = y.cpu().numpy()
            y_onehot = (np.arange(2) == y_onehot[:,None]).astype(np.float32)
            y_onehot = torch.from_numpy(y_onehot).cuda(gpu)
        else:
            y_onehot = y.numpy()
            y_onehot = (np.arange(2) == y_onehot[:,None]).astype(np.float32)
            y_onehot = torch.from_numpy(y_onehot)
        return y_onehot
    with torch.no_grad():
        # print(pred, labels)
        temp = (pred == labels)
        TP = torch.sum((labels == 1) * (pred == 1)).item()
        FP = torch.sum((labels == 0) * (pred == 1)).item()
        TN = torch.sum((labels == 0) * (pred == 0)).item()
        FN = torch.sum((labels == 1) * (pred == 0)).item()
        precision = 0
        if TP+FP > 0:
            precision = 1.0*TP/(TP+FP)
        recall = 0
        if TP+FN > 0:
            recall = 1.0*TP/(TP+FN)
        f1 = 0
        if precision+recall > 0:
            f1 = 2.0*precision*recall/(precision+recall)
        mcc, den = 0, (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
        if den > 0:
            mcc = (TP*TN - FP*FN)/np.sqrt(den)
        # print (TP, FP, TN, FN, precision, recall, f1)
        if class_weights is not None:
            class_weights_rep = class_weights.repeat(1,pred.shape[0])
            class_weights_rep = class_weights_rep.view(-1,class_weights.shape[0])
            # code.interact(local=locals())
            return  torch.sum(class_weights_rep*one_hot(temp).float()) / torch.sum(class_weights_rep*one_hot(labels).float())
        if give_stat:
            return precision, recall, f1, mcc, 100.0 * torch.sum(temp).long() / labels.shape[0], (TP, FP, TN, FN)
        return precision, recall, f1, mcc, 100.0 * torch.sum(temp).long() / labels.shape[0]
