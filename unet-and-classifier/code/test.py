import time
import os, torch
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.autograd import Variable
from data import *
from torch import optim
from model import *
from classifier_model import *
import torch.nn.functional as F

os.makedirs('../../data', exist_ok=True)
os.makedirs('../model', exist_ok=True)
os.makedirs('../results', exist_ok=True)
os.makedirs('../results/unet-predictions', exist_ok=True)
os.makedirs('../../../malaria/images', exist_ok=True)
os.makedirs('../../../malaria/cell-labels', exist_ok=True)
os.makedirs('../results/classifier/', exist_ok=True)

cuda = True if torch.cuda.is_available() else False

def segmentation_accuracy(label, pred):
    with torch.no_grad():
        pred, label = pred.detach().cpu().numpy(), label.detach().cpu().numpy()
        intersection = np.logical_and(label, pred)
        union = np.logical_or(label, pred)
        iou_score = np.sum(intersection) / np.sum(union)
        pixel_acc = np.average(label==pred)
        return (iou_score, pixel_acc)

def test(args):
    if cuda:
        unet = UNet(3, 1).cuda()
        unet = torch.load('../model/unet')
        # print('unet loaded')
        # classifier = ClassifierModel(image_shape[0]).cuda()
        # checkpoint = torch.load('../model/classifier')
        # classifier.load_state_dict(checkpoint['model_state_dict'])
        # print('classifier loaded')
    else:
        unet = UNet(3, 1)
        unet = torch.load('../model/unet', map_location='cpu')
        # classifier = ClassifierModel(image_shape[0])
        # checkpoint = torch.load('../model/classifier')
        # classifier.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = MalariaDataset(csv_file=data_csv,
                        img_dir=images_dir,
                        label_dir=cell_label_dir,
                        img_transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]),
                        label_transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))
    # print('loaded dataset')
    test_loader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False, num_workers=4)
    # print('loaded dataloader')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    iou_score_avg, pixel_acc_avg = 0, 0
    cnt = 0
    saved = []
    print('Starting Testing.')
    for i, (img, label, img_name) in enumerate(test_loader):
        print(i, img.shape, label.shape, img_name)
        img = Variable(img.type(Tensor), requires_grad=False)
        label = Variable(label.type(Tensor), requires_grad=False)
        if cuda:
            img.cuda()
            label.cuda()
        with torch.no_grad(): 
            pred = unet(img)
            pred_th = (pred >= args.unet_thresh).float()
            stat = segmentation_accuracy(pred_th, label)
            iou_score_avg += stat[0]
            pixel_acc_avg += stat[1]
            # print(img.shape, label.shape, pred.shape)
            os.makedirs('../results/', exist_ok=True)
            #save_image(img, '../results/img.png')
            #save_image(label, '../results/label.png')
            #save_image(pred, '../results/pred.png')
            #save_image(pred_th, '../results/thpred.png')
            torch.save(img, '../results/unet-predictions/img-'+img_name[0][1:][:-4])
            torch.save(label, '../results/unet-predictions/label-'+img_name[0][1:][:-4])
            torch.save(pred, '../results/unet-predictions/pred-'+img_name[0][1:][:-4])
            torch.save(pred_th, '../results/unet-predictions/pred_th-'+img_name[0][1:][:-4])
            print('Saved Unet Mask.', stat, torch.sum(pred_th).item())
            saved.append(img_name[0][1:][:-4])
            
            cnt += 1
            #if i > 50:
            #    break
    #with open("saved.csv", "wb") as f:
    #    writer = csv.writer(f)
    #    writer.writerows(saved)
    stat = iou_score_avg/cnt, pixel_acc_avg/cnt
    print('({0:.4f}, {1:.4f})'.format(stat[0], stat[1]))

def classifier_loop(args):
    if cuda:
        classifier = ClassifierModel(image_shape[0]).cuda()
        checkpoint = torch.load('../model/classifier')
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier = ClassifierModel(image_shape[0])
        checkpoint = torch.load('../model/classifier')
        classifier.load_state_dict(checkpoint['model_state_dict'])
    print('model loaded.')
    
    import glob
    toks = glob.glob("../results/unet-predictions/img-*")  
    toks = list(map(lambda x: x[32:], toks))

    for indx, tok in enumerate(toks):
        img = torch.load('../results/unet-predictions/img-'+tok)
        label = torch.load('../results/unet-predictions/label-'+tok)
        # pred = torch.load('../results/unet-predictions/pred-'+tok)
        pred_th = torch.load('../results/unet-predictions/pred_th-'+tok)
        
        # print('images loaded.')

        prob_map = classify(img, pred_th, classifier, args)

        torch.save(prob_map, '../results/unet-predictions/prob_map-'+tok)

        prob_map_th = (prob_map >= args.classifier_thresh)
        #torch.save(prob_map_th, '../results/unet-predictions/prob_map_th-'+tok)
        os.makedirs('../results/classifier/'+tok, exist_ok=True)
        os.system('cp ../../../malaria/infected-cell-labels/'+tok+'.png ../results/classifier/'+tok+'/infected.png')
        save_image(img, '../results/classifier/'+tok+'/img.png')
        save_image(label, '../results/classifier/'+tok+'/labl.png')
        save_image(pred_th, '../results/classifier/'+tok+'/pred_th.png')
        save_image(prob_map, '../results/classifier/'+tok+'/prob_map.png')
        save_image(prob_map_th, '../results/classifier/'+tok+'/prob_map_th.png')
        print(indx, tok)


def classify(img, unet_map, classifier, args):
    with torch.no_grad():
        l, r = image_shape[0]//2, image_shape[1]//2
        prob_map = torch.zeros(unet_map.shape)
        im = torch.zeros(args.batch_size, 3, image_shape[0], image_shape[1])
        # print(img.shape, unet_map.shape)
        img = F.pad(input=img, pad=(l, l, r, r), mode='reflect')

        if cuda:
            img = img.cuda()
            im = im.cuda()
            prob_map = prob_map.cuda()
        # print(img.shape, prob_map.shape)

        cnt = 0
        lind = []
        for i in range(prob_map.shape[2]):
            for j in range(prob_map.shape[3]):
                if unet_map[0,0,i,j] > 0:
                    if cnt == args.batch_size:
                        pr = classifier(im)
                        for ind in range(pr.shape[0]):
                            prob_map[0,0,lind[ind][0],lind[ind][1]] = pr[ind]
                        cnt = 0
                        lind = []
                        # print((i,j))
                    else:
                        im[cnt, :,:,:] = img[:, :, i:i+2*l, j:j+2*r]
                        lind.append((i,j))
                        cnt+=1
        if cnt > 0:
            im = im[0:cnt, :, :, :]
            pr = classifier(im)
            for ind in range(pr.shape[0]):
                prob_map[0,0,lind[ind][0],lind[ind][1]] = pr[ind]
    # print('prob_map created')
    return prob_map
