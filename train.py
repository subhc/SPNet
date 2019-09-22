#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  _list-11-01
# Modified by: Subhabrata Choudhury

from __future__ import absolute_import, division, print_function

import pickle
import os
import sys
import time
import inspect
import shutil
import os.path as osp
import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
import torchvision.models.resnet
from tqdm import tqdm
from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils.loss import CrossEntropyLoss2d
import json

def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0] or "vgg" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias



class RandomImageSampler(torch.utils.data.Sampler):
    r"""Samples classes randomly, then returns images corresponding to those classes.
    """

    def __init__(self, seenset, novelset):
        self.data_index = []
        for v in seenset:
            self.data_index.append([v, 0])
        for v,i in novelset:
            self.data_index.append([v, i+1])

    def __iter__(self):
        return iter([ self.data_index[i] for i in np.random.permutation(len(self.data_index))])

    def __len__(self):
        return len(self.data_index)

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    optimizer.param_groups[1]["lr"] = 10 * new_lr
    optimizer.param_groups[2]["lr"] = 20 * new_lr


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(new_target).long()



@click.command()
@click.option("-c", "--config", type=str, required=True)
@click.option("--cuda/--no-cuda", default=True)
@click.option("--excludeval/--no-excludeval", default=False)
@click.option("--embedding", default='fastnvec')
@click.option("--continue-from", type=int)
@click.option("--nolog", is_flag=True)
@click.option("--inputmix", type=str, default='seen')
@click.option("--imagedataset", default='cocostuff')
@click.option("--experimentid", type=str)
@click.option("--nshot", type=int)
@click.option("--ishot", type=int, default=0)
def main(config, cuda, excludeval,  embedding, continue_from, nolog, inputmix, imagedataset, experimentid, nshot, ishot ):
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    #print(values)

    #in case you want to save to the location of script you're running
    datadir = os.path.join('data/datasets', imagedataset)
    if not nolog:
        #name the savedir, might add logs/ before the datetime for clarity
        if experimentid is None:
            savedir = time.strftime('%Y%m%d%H%M%S')
        else:
            savedir = experimentid
        #the full savepath is then:
        savepath = os.path.join('logs', imagedataset, savedir)
        #in case the folder has not been created yet / except already exists error:
        try:
            os.makedirs(savepath)
            print("Log dir:", savepath)
        except:
            pass
        if continue_from is None:
            #now join the path in save_screenshot:
            shutil.copytree('./libs/', savepath+'/libs')
            shutil.copy2(osp.abspath(inspect.stack()[0][1]), savepath)
            shutil.copy2(config, savepath)
            args_dict = {}
            for a in args:
                args_dict[a] = values[a]
            with open(savepath+'/args.json', 'w') as fp:
                json.dump(args_dict, fp)

    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(config), Loader=yaml.FullLoader))
    visibility_mask = {}
    if excludeval:
        seen_classes = np.load(datadir+'/split/seen_cls.npy')
    else:
        seen_classes = np.asarray(np.concatenate([np.load(datadir+'/split/seen_cls.npy'), np.load(datadir+'/split/val_cls.npy')]),dtype=int)

    novel_classes = np.load(datadir+'/split/novel_cls.npy')
    seen_novel_classes = np.concatenate([seen_classes, novel_classes])

    seen_map = np.array([-1]*256)
    for i,n in enumerate(list(seen_classes)):
        seen_map[n] = i

    visibility_mask[0] = seen_map.copy()
    for i, n in enumerate(list(novel_classes)):
        visibility_mask[i+1] = seen_map.copy()
        visibility_mask[i+1][n] = seen_classes.shape[0]+i
    if excludeval:
        train = np.load(datadir+'/split/train_list.npy')[:-CONFIG.VAL_SIZE]
    else:
        train = np.load(datadir+'/split/train_list.npy')

    novelset = []
    seenset = []

    if inputmix == 'novel' or inputmix == 'both':
        inverse_dict = pickle.load(open(datadir+'/split/inverse_dict_train.pkl', 'rb'))
        for icls, key in enumerate(novel_classes):
            if(inverse_dict[key].size >0):
                for v in inverse_dict[key][ishot*20: ishot*20+nshot]:
                    novelset.append((v, icls))
                    #print((v, icls))

    if inputmix == 'both':
        seenset = []
        inverse_dict = pickle.load(open(datadir+'/split/inverse_dict_train.pkl', 'rb'))
        for icls, key in enumerate(seen_classes):
            if(inverse_dict[key].size >0):
                for v in inverse_dict[key][ishot*20: ishot*20+nshot]:
                    seenset.append(v)

    if inputmix == 'seen':
        seenset = range(train.shape[0])

    sampler = RandomImageSampler(seenset, novelset)

    if inputmix == 'novel':
        visible_classes = seen_novel_classes
        if nshot is not None:
            nshot = str(nshot)+'n'
    elif inputmix == 'seen':
        visible_classes = seen_classes
        if nshot is not None:
            nshot = str(nshot)+'s'
    elif inputmix == 'both':
        visible_classes = seen_novel_classes
        if nshot is not None:
            nshot = str(nshot)+'b'

    
    print("Visible classes:", visible_classes.size, " \nClasses are: ", visible_classes, "\nTrain Images:", train.shape[0])

    #a Dataset 10k or 164k
    dataset = get_dataset(CONFIG.DATASET)(train=train, test=None,
            root=CONFIG.ROOT,
            split=CONFIG.SPLIT.TRAIN,
            base_size=513,
            crop_size=CONFIG.IMAGE.SIZE.TRAIN,
            mean=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
            warp=CONFIG.WARP_IMAGE,
            scale=(0.5, 1.5),
            flip=True,
            visibility_mask=visibility_mask
        )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.NUM_WORKERS,
        sampler = sampler
    )
    
    if  embedding == 'word2vec':
        class_emb = pickle.load(open(datadir+'/word_vectors/word2vec.pkl', "rb"))
    elif embedding == 'fasttext':
        class_emb = pickle.load(open(datadir+'/word_vectors/fasttext.pkl', "rb"))
    elif embedding == 'fastnvec':
        class_emb = np.concatenate([pickle.load(open(datadir+'/word_vectors/fasttext.pkl', "rb")), pickle.load(open(datadir+'/word_vectors/word2vec.pkl', "rb"))], axis = 1)
    else:
        print("invalid emb ", embedding)
        sys.exit() 

    print((class_emb.shape))
    class_emb = F.normalize(torch.tensor(class_emb), p=2, dim=1).cuda()

    loader_iter = iter(loader)
    DeepLab = DeepLabV2_ResNet101_MSC
    #import ipdb; ipdb.set_trace()
    state_dict = torch.load(CONFIG.INIT_MODEL)

    # Model load
    model = DeepLab(class_emb.shape[1], class_emb[visible_classes]) 
    if continue_from is not None and continue_from > 0:
        print("Loading checkpoint: {}".format(continue_from))
        #import ipdb; ipdb.set_trace()
        model = nn.DataParallel(model)
        state_file = osp.join(savepath, "checkpoint_{}.pth".format(continue_from))
        if osp.isfile(state_file+'.tar') :
            state_dict = torch.load(state_file+'.tar')
            model.load_state_dict(state_dict['state_dict'], strict=True)
        elif osp.isfile(state_file) :
            state_dict = torch.load(state_file)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("Checkpoint {} not found".format(continue_from))
            sys.exit()

    else:
        model.load_state_dict(state_dict, strict=False)  # make strict=True to debug if checkpoint is loaded correctly or not if performance is low
        model = nn.DataParallel(model)
    model.to(device)
    # Optimizer

    optimizer = {
        "sgd": torch.optim.SGD(
            # cf lr_mult and decay_mult in train.prototxt
            params=[
                {
                    "params": get_params(model.module, key="1x"),
                    "lr": CONFIG.LR,
                    "weight_decay": CONFIG.WEIGHT_DECAY,
                },
                {
                    "params": get_params(model.module, key="10x"),
                    "lr": 10 * CONFIG.LR,
                    "weight_decay": CONFIG.WEIGHT_DECAY,
                },
                {
                    "params": get_params(model.module, key="20x"),
                    "lr": 20 * CONFIG.LR,
                    "weight_decay": 0.0,
                }
            ],
            momentum=CONFIG.MOMENTUM,
        ),
        "adam": torch.optim.Adam(
            # cf lr_mult and decay_mult in train.prototxt
            params=[
                {
                    "params": get_params(model.module, key="1x"),
                    "lr": CONFIG.LR,
                    "weight_decay": CONFIG.WEIGHT_DECAY,
                },
                {
                    "params": get_params(model.module, key="10x"),
                    "lr": 10 * CONFIG.LR,
                    "weight_decay": CONFIG.WEIGHT_DECAY,
                },
                {
                    "params": get_params(model.module, key="20x"),
                    "lr": 20 * CONFIG.LR,
                    "weight_decay": 0.0,
                }
            ]
        )
        # Add any other optimizer
    }.get(CONFIG.OPTIMIZER)

    if 'optimizer' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
    print("Learning rate:",  CONFIG.LR )
    # Loss definition
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    criterion.to(device)

    if not nolog:
        # TensorBoard Logger
        if continue_from is not None:
            writer = SummaryWriter(savepath+'/runs/fs_{}_{}_{}'.format(continue_from, nshot, ishot))
        else:
            writer = SummaryWriter(savepath+'/runs')
        loss_meter = MovingAverageValueMeter(20)

    model.train()
    model.module.scale.freeze_bn()

    pbar =  tqdm(
        range(1, CONFIG.ITER_MAX + 1),
        total=CONFIG.ITER_MAX,
        leave=False,
        dynamic_ncols=True,
    )
    for iteration in pbar:

        # Set a learning rate
        poly_lr_scheduler(
            optimizer=optimizer,
            init_lr=CONFIG.LR,
            iter=iteration - 1,
            lr_decay_iter=CONFIG.LR_DECAY,
            max_iter=CONFIG.ITER_MAX,
            power=CONFIG.POLY_POWER,
        )

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        iter_loss = 0
        for i in range(1, CONFIG.ITER_SIZE + 1):
            try:
                data, target = next(loader_iter)
            except:
                loader_iter = iter(loader)
                data, target = next(loader_iter)

            # Image
            data = data.to(device)

            # Propagate forward
            outputs = model(data)
            # Loss
            loss = 0
            for output in outputs:
                # Resize target for {100%, 75%, 50%, Max} outputs
                target_ = resize_target(target, output.size(2))
                target_ = torch.tensor(target_).to(device)
                loss += criterion.forward(output, target_)

            # Backpropagate (just compute gradients wrt the loss)
            #print(loss)
            loss /= float(CONFIG.ITER_SIZE)
            loss.backward()

            iter_loss += float(loss)

        #print(iter_loss)
        pbar.set_postfix(loss = "%.3f" % iter_loss)

        # Update weights with accumulated gradients
        optimizer.step()
        if not nolog:
            loss_meter.add(iter_loss)
            # TensorBoard
            if iteration % CONFIG.ITER_TB == 0:
                writer.add_scalar("train_loss", loss_meter.value()[0], iteration)
                for i, o in enumerate(optimizer.param_groups):
                    writer.add_scalar("train_lr_group{}".format(i), o["lr"], iteration)
                if False:  # This produces a large log file
                    for name, param in model.named_parameters():
                        name = name.replace(".", "/")
                        writer.add_histogram(name, param, iteration, bins="auto")
                        if param.requires_grad:
                            writer.add_histogram(
                                name + "/grad", param.grad, iteration, bins="auto"
                            )

            # Save a model
            if continue_from is not None:
                if iteration in CONFIG.ITER_SAVE:
                    torch.save(
                        {
                            'iteration': iteration,
                            'state_dict': model.state_dict(),
                        },
                        osp.join(savepath, "checkpoint_{}_{}_{}_{}.pth.tar".format(continue_from, nshot, ishot, iteration)),
                    )

                # Save a model (short term) [unnecessary for fewshot]
                if False and iteration % 100 == 0:
                    torch.save(
                        {
                            'iteration': iteration,
                            'state_dict': model.state_dict(),
                        },
                        osp.join(savepath, "checkpoint_{}_{}_{}_current.pth.tar".format(continue_from, nshot, ishot)),
                    )
                    print(osp.join(savepath, "checkpoint_{}_{}_{}_current.pth.tar".format(continue_from, nshot, ishot)))
            else:
                if iteration % CONFIG.ITER_SAVE == 0:
                    torch.save(
                        {
                            'iteration': iteration,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        },
                        osp.join(savepath, "checkpoint_{}.pth.tar".format(iteration)),
                    )

                # Save a model (short term)
                if iteration % 100 == 0:
                    torch.save(
                        {
                            'iteration': iteration,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        },
                        osp.join(savepath, "checkpoint_current.pth.tar"),
                    )

    if not nolog:
        if continue_from is not None:
            torch.save(
                {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                },
                osp.join(savepath, "checkpoint_{}_{}_{}_{}.pth.tar".format(continue_from, nshot, ishot, CONFIG.ITER_MAX))
            )
        else:
            torch.save(
                {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                },
                osp.join(savepath, "checkpoint_{}.pth.tar".format(CONFIG.ITER_MAX))
            )

if __name__ == "__main__":
    main()

