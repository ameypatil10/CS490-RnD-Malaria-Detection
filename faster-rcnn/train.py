from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import os

# import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

import numpy as np
import torch

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

os.makedirs('results/', exist_ok=True)


def eval(dataloader, faster_rcnn, test_num=500, save_pred=False, trainer=None, opt=None, train_dataloader=None):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        # torch.save(imgs[0], 'results/img-'+str(ii))
        # torch.save(pred_bboxes_, 'results/pred_bboxes_-'+str(ii))
        # torch.save(pred_labels_, 'results/pred_labels_-'+str(ii))
        # torch.save(pred_scores_, 'results/pred_scores_-'+str(ii))
        # torch.save(gt_bboxes, 'results/gt_bboxes-'+str(ii))
        # torch.save(gt_labels, 'results/gt_labels-'+str(ii))
        # print('saved into results')
        gt_bboxes_ = list(gt_bboxes_.numpy())
        gt_labels_ = list(gt_labels_.numpy())
        gt_difficults_ = list(gt_difficults_.numpy())
        if save_pred:
            tiny_result = eval_detection_voc(pred_bboxes_, pred_labels_, pred_scores_, gt_bboxes_, gt_labels_, gt_difficults_, use_07_metric=True)
            save_pred_image(trainer, train_dataloader, opt, idx=ii, map=tiny_result['map'])
            print(tiny_result)
        gt_bboxes += gt_bboxes_
        gt_labels += gt_labels_
        gt_difficults += gt_difficults_
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            result = eval_detection_voc(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults,
                use_07_metric=True)
            return result
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)
    
    dataset = Dataset(opt)
    testset = TestDataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                #   pin_memory=True,
                                  num_workers=opt.num_workers)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    # trainer.vis.text(dataset.label_names, win='labels')
    # test_and_save(trainer, test_dataloader, opt)
    eval_result = eval(test_dataloader, trainer.faster_rcnn, test_num=opt.test_num, save_pred=True, trainer=trainer, opt=opt, train_dataloader=dataloader)
    print('test_accuracy = ', eval_result['map'])
    # save_pred(trainer, dataloader, opt)
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                # if os.path.exists(opt.debug_file):
                #     ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
            if (ii + 1) % opt.test_every == 0:
                eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
                trainer.vis.plot('test_map', eval_result['map'])
                lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
                log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                        str(eval_result['map']),
                                                        str(trainer.get_meter_data()))
                trainer.vis.log(log_info)

                if eval_result['map'] > best_map:
                    best_map = eval_result['map']
                    best_path = trainer.save(best_map=best_map)
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        print('epoch: ', epoch, log_info)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                str(eval_result['map']),
                                                str(trainer.get_meter_data()))
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break

def save_pred_image(trainer, dataloader, opt, idx=None, map=0):
    for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
        if idx and idx == ii:
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            ori_img_ = inverse_normalize(at.tonumpy(img[0]))
            gt_img = visdom_bbox(ori_img_,
                                    at.tonumpy(bbox_[0]),
                                    at.tonumpy(label_[0]))
            trainer.vis.img('gt_img-'+str(ii)+' - map='+str(map), gt_img)
            _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
            pred_img = visdom_bbox(ori_img_,
                                    at.tonumpy(_bboxes[0]),
                                    at.tonumpy(_labels[0]).reshape(-1),
                                    at.tonumpy(_scores[0]))
            trainer.vis.img('pred_img-'+str(ii)+' - map='+str(map), pred_img)
            print('images to visdom.')


if __name__ == '__main__':
    import fire

    fire.Fire()
