#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-03
# Modified by: Subhabrata Choudhury


from __future__ import absolute_import, division, print_function
import sys
import json
import os.path as osp
#import ipdb
import os
import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from PIL import Image
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import dense_crf, scores, scores_gzsl
import pickle
import re
import timeit


@click.command()
@click.option("-c", "--config", type=str, required=True)
@click.option("--excludeval/--no-excludeval", default=False)
@click.option("--embedding", default='fastnvec')
@click.option("-m", "--model-path", type=str, required=True)
@click.option("-r", "--run", type=str, required=True)
@click.option("--cuda/--no-cuda", default=True)
@click.option("--crf", is_flag=True)
@click.option("--redo", is_flag=True)
@click.option("--threshold", type=float)
@click.option("--imagedataset", default='cocostuff')
def main(config, excludeval, embedding,  model_path, run, cuda, crf, redo, imagedataset, threshold):
    pth_extn = '.pth.tar'
    if osp.isfile(model_path.replace(pth_extn,  "_" + run + ".json")) and not threshold and not redo:
        print("Already Done!")
        with open(model_path.replace(pth_extn,  "_" + run + ".json")) as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    if not key == "Class IoU":
                        print(key, value)
        sys.exit()

    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(config)))

    
    datadir = os.path.join('data/datasets', imagedataset)
    print("Split dir: ", datadir)
    savedir = osp.dirname(model_path)
    epoch = re.findall("checkpoint_(.*)\."+pth_extn[1:], osp.basename(model_path))[-1]
    val = None
    visible_classes = None


    if run == 'zlss' or run == 'flss':
        val = np.load(datadir + '/split/test_list.npy')
        visible_classes = np.load(datadir + '/split/novel_cls.npy')
    elif run == 'gzlss' or run == 'gflss':
        val = np.load(datadir + '/split/test_list.npy')
        if excludeval:
            vals_cls = np.asarray(np.load(datadir+'/split/seen_cls.npy'), dtype=int)
        else:
            vals_cls = np.asarray(np.concatenate([np.load(datadir+'/split/seen_cls.npy'), np.load(datadir+'/split/val_cls.npy')]), dtype=int)
        valu_cls = np.load(datadir + '/split/novel_cls.npy')
        visible_classes = np.concatenate([vals_cls, valu_cls])
    else:
        print("invalid run ", run)
        sys.exit()

    if threshold is not None and run != 'gzlss':
        print("invalid run for threshold", run)
        sys.exit()
    
    
    cls_map = np.array([255]*256)
    for i,n in enumerate(visible_classes):
        cls_map[n] = i
    
    

    if threshold is not None:
        savedir = osp.join(savedir, str(threshold))
        
    if crf is not None:
        savedir = savedir+'-crf'

    if run == 'gzlss' or run == 'gflss':

        novel_cls_map = np.array([255]*256)
        for i,n in enumerate(list(valu_cls)):
            novel_cls_map[cls_map[n]] = i

        seen_cls_map = np.array([255]*256)
        for i,n in enumerate(list(vals_cls)):
            seen_cls_map[cls_map[n]] = i

        if threshold is not None:

            thresholdv = np.asarray(np.zeros((visible_classes.shape[0],1)), dtype=np.float)
            thresholdv[np.in1d(visible_classes, vals_cls),0] = threshold
            thresholdv = torch.tensor(thresholdv).float().cuda()

    visible_classesp = np.concatenate([visible_classes, [255]])

    all_labels  = np.genfromtxt(datadir+'/labels_2.txt', delimiter='\t', usecols=1, dtype='str')

    print("Visible Classes: ", visible_classes)
    
    
    # Dataset 
    dataset = get_dataset(CONFIG.DATASET)(train=None, test=val,
        root=CONFIG.ROOT,
        split=CONFIG.SPLIT.TEST,
        base_size=CONFIG.IMAGE.SIZE.TEST,
        mean=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        warp=CONFIG.WARP_IMAGE,
        scale=None,
        flip=False,
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

    class_emb = class_emb[visible_classes]
    class_emb = F.normalize(torch.tensor(class_emb), p=2, dim=1).cuda()


    print("Embedding dim: ", class_emb.shape[1])
    print("# Visible Classes: ", class_emb.shape[0])


    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE.TEST,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=False,
    )

    torch.set_grad_enabled(False)


    # Model 
    model = DeepLabV2_ResNet101_MSC(class_emb.shape[1], class_emb) 

    sdir = osp.join(savedir, model_path.replace(pth_extn,  ""), str(epoch), run)

    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    model.to(device)
    imgfeat = []
    targets, outputs = [], []
    for data, target, img_id in tqdm(
        loader, total=len(loader), leave=False, dynamic_ncols=True
    ):
        # Image
        data = data.to(device)
        # Forward propagation
        output = model(data)
        output = F.interpolate(output, size=data.shape[2:], mode="bilinear", align_corners = False)
        
        output = F.softmax(output, dim=1)
        if threshold is not None:
            output = output - thresholdv.view(1,-1,1,1)

        target = cls_map[target.numpy()]

        # Postprocessing
        if crf:
            output = output.data.cpu().numpy()
            crf_output = np.zeros(output.shape)
            images = data.data.cpu().numpy().astype(np.uint8)
            for i, (image, prob_map) in enumerate(zip(images, output)):
                image = image.transpose(1, 2, 0)
                crf_output[i] = dense_crf(image, prob_map)
            output = crf_output
            output = np.argmax(output, axis=1)
        else:
            output = torch.argmax(output, dim=1).cpu().numpy()

        for o, t in zip(output, target):
            outputs.append(o)
            targets.append(t)

    if run == 'gzlss' or  run == 'gflss' :
        score, class_iou = scores_gzsl(targets, outputs, n_class=len(visible_classes), seen_cls=cls_map[vals_cls], unseen_cls=cls_map[valu_cls])
    else:
        score, class_iou = scores(targets, outputs, n_class=len(visible_classes))

    for k, v in score.items():
        print(k, v)

    score["Class IoU"] = {}
    for i in range(len(visible_classes)):
        score["Class IoU"][all_labels[visible_classes[i]]] = class_iou[i]
    

    if threshold is not None:
        with open(model_path.replace(pth_extn,  "_" + run + '_T' + str(threshold) + ".json"), "w") as f:
            json.dump(score, f, indent=4, sort_keys=True)
    else:
        with open(model_path.replace(pth_extn,  "_" + run + ".json"), "w") as f:
            json.dump(score, f, indent=4, sort_keys=True)
    
    print(score["Class IoU"])


if __name__ == "__main__":
    print("Time Taken", timeit.timeit(main))


