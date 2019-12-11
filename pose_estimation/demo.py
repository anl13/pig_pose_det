# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from core.inference import get_final_preds

import dataset
import models
from demo_utils import * 
import pickle 
import json 
from tqdm import tqdm 

import matplotlib.pyplot as plt 
import cv2 

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file

def main(seq, is_vis=False):
    args = parse_args()
    reset_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )

    load_state_dict_module(model, config.TEST.MODEL_FILE) 

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    model.eval() 

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    mytransforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    for frameid in tqdm(range(0,10000)):
        
        output_pkl  = "/home/al17/animal/pig-data/sequences/{}/keypoints_pig20/{:06d}.pkl".format(seq, frameid)
        output_json = "/home/al17/animal/pig-data/sequences/{}/keypoints_pig20/{:06d}.json".format(seq, frameid)

        camids = [0,1,2,5,6,7,8,9,10,11]
        boxfile = "/home/al17/animal/pig-data/sequences/{}/boxes/boxes_{:06d}.pkl".format(seq, frameid)
        keypoints_dict = {} 
        with open(boxfile, 'rb') as f: 
            boxes_allviews = pickle.load(f, encoding='latin1')
        for camid in camids: 
            imgfile = "/home/al17/animal/pig-data/sequences/{}/images/cam{}/{:06d}.jpg".format(seq, camid, frameid)
            rawimg = cv2.imread(imgfile) 
            boxes = boxes_allviews[str(camid)]
            if len(boxes) == 0:
                keypoints_dict.update({str(camid):[]})
                continue
            net_input, all_c, all_s = preprocess(rawimg, boxes, mytransforms=mytransforms)
            net_out = model(net_input) # heatmaps [N,20,288,384]
            num_samples = len(boxes) 
            all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32) 
            all_boxes = np.zeros((num_samples, 6))

            preds, maxvals = get_final_preds(
                config, net_out.detach().clone().cpu().numpy(), all_c, all_s)
            all_preds[0: num_samples, :, 0:2] = preds[:, :, 0:2]
            all_preds[0: num_samples, :, 2:3] = maxvals
            all_boxes[0: num_samples, 0:2] = all_c[:, 0:2]
            all_boxes[0: num_samples, 2:4] = all_s[:, 0:2]
            all_boxes[0: num_samples, 4] = np.prod(all_s*200, 1)
            all_boxes[0: num_samples, 5] = 1
            keypoints_dict.update({str(camid):all_preds.reshape(num_samples, 20*3).tolist()})

            if is_vis: 
                vis = draw_keypoints(rawimg, all_preds) 
                vis_rgb = vis[:,:,(2,1,0)]
                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.imshow("image", vis) 
                key = cv2.waitKey() 
                if key == 27: 
                    exit() 
        with open(output_pkl, 'wb') as f: 
            pickle.dump(keypoints_dict, f, protocol=2) 
        with open(output_json, "w") as f: 
            json.dump(keypoints_dict, f)

if __name__ == '__main__':
    seq1 = "20190704_morning" 
    seq2 = "20190704_noon"
    main(seq1, is_vis=True)
    # main(seq2) 
