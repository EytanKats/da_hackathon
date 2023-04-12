# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from parser_train import parser_, relative_path_to_absolute_path

from tqdm import tqdm
from data import create_dataset
from models import adaptation_modelv2
from utils import fliplr

from base_code.train import get_datasets

def test(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # datasets = create_dataset(opt, logger)
    train_loader_src, train_loader_tgt, val_loader = get_datasets()


    if opt.model_name == 'deeplabv2':
        checkpoint = torch.load(opt.resume_path)['ResNet101']["model_state"]
        model = adaptation_modelv2.CustomModel(opt, logger)
        model.BaseNet.load_state_dict(checkpoint)

    # validation(model, logger, datasets, device, opt)
    validation(model, logger, train_loader_tgt, device, opt)

def validation(model, logger, datasets, device, opt):
    _k = -1
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        # validate(datasets.target_train_loader, device, model, opt)
        validate(datasets, device, model, opt)
        #validate(datasets.target_valid_loader, device, model, opt)

def label2rgb(func, label):
    rgbs = []
    for k in range(label.shape[0]):
        rgb = func(label[k, 0].cpu().numpy())
        rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1))
    rgbs = torch.stack(rgbs, dim=0).float()
    return rgbs

def validate(data_loader, device, model, opt):
    ori_LP = os.path.join(opt.root, 'pseudo')

    if not os.path.exists(ori_LP):
        os.makedirs(ori_LP)

    sm = torch.nn.Softmax(dim=1)
    for data_i in tqdm(data_loader):
        images_val = data_i[0].to(device)
        # labels_val = data_i['label'].to(device)
        # filename = data_i['img_path']
        filename = data_i[4]

        out = model.BaseNet_DP(images_val)

        if opt.soft:
            threshold_arg = F.softmax(out['out'], dim=1)
            for k in range(images_val.shape[0]):
                name = os.path.basename(filename[k])
                np.save(os.path.join(ori_LP, name.replace('.pth', '.npy')), threshold_arg[k].cpu().numpy())
        else:
            if opt.flip:
                flip_out = model.BaseNet_DP(fliplr(images_val))
                flip_out['out'] = F.interpolate(sm(flip_out['out']), size=images_val.size()[2:], mode='bilinear', align_corners=True)
                out['out'] = F.interpolate(sm(out['out']), size=images_val.size()[2:], mode='bilinear', align_corners=True)
                out['out'] = (out['out'] + fliplr(flip_out['out'])) / 2

            confidence, pseudo = out['out'].max(1, keepdim=True)
            #entropy = -(out['out']*torch.log(out['out']+1e-6)).sum(1, keepdim=True)
            pseudo_rgb = label2rgb(decode_segmap, pseudo).float() * 255
            for k in range(images_val.shape[0]):
                name = os.path.basename(filename[k])
                Image.fromarray(pseudo[k,0].cpu().numpy().astype(np.uint8)).save(os.path.join(ori_LP, name.replace('.pth', '.png')))
                Image.fromarray(pseudo_rgb[k].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(os.path.join(ori_LP, name[:-4] + '_color.png'))
                np.save(os.path.join(ori_LP, name.replace('.pth', '_conf.npy')), confidence[k, 0].cpu().numpy().astype(np.float16))
                #np.save(os.path.join(ori_LP, name.replace('.png', '_entropy.npy')), entropy[k, 0].cpu().numpy().astype(np.float16))
                
def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    file_path = os.path.join(logdir, 'run.log')
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger


colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

label_colours = dict(zip(range(19), colors))

def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 5):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--save_path', type=str, default='./Pseudo', help='pseudo label update thred')
    parser.add_argument('--soft', action='store_true', help='save soft pseudo label')
    parser.add_argument('--flip', action='store_false')
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'debug')
    opt.noaug = True
    opt.noshuffle = True

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    test(opt, logger)

#python generate_pseudo_label.py --name gta2citylabv2_warmup_soft --soft --resume_path ./logs/gta2citylabv2_warmup/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
#python generate_pseudo_label.py --name gta2citylabv2_stage1Denoise --flip --resume_path ./logs/gta2citylabv2_stage1Denoisev2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
#python generate_pseudo_label.py --name gta2citylabv2_stage2 --flip --resume_path ./logs/gta2citylabv2_stage2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast --bn_clr --student_init simclr
#python generate_pseudo_label.py --name syn2citylabv2_warmup_soft --soft --src_dataset synthia --n_class 16 --src_rootpath Dataset/SYNTHIA-RAND-CITYSCAPES --resume_path ./logs/syn2citylabv2_warmup/from_synthia_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast