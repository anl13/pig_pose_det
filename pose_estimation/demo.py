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
from config import cfg
from config import update_config
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        default="experiments/pig/w48_384x288-tiger.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()
    return args


def main(seq, is_vis=False):
    args = parse_args()
    update_config(cfg, args)
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    load_state_dict_module(model, cfg.TEST.MODEL_FILE)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
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
        
        output_pkl  = "/home/al17/animal/pig-data/sequences/{}/keypoints_hrnet/{:06d}.pkl".format(seq, frameid)
        output_json = "/home/al17/animal/pig-data/sequences/{}/keypoints_hrnet/{:06d}.json".format(seq, frameid)

        camids = [0,1,2,5,6,7,8,9,10,11]
        boxfile = "/home/al17/animal/pig-data/sequences/{}/boxes/boxes_{:06d}.pkl".format(seq,frameid)
        keypoints_dict = {} 
        with open(boxfile, 'rb') as f: 
            boxes_allviews = pickle.load(f, encoding='latin1')
        for camid in camids: 
            imgfile = "/home/al17/animal/pig-data/sequences/{}/images/cam{}/{:06d}.jpg".format(seq,camid, frameid)
            rawimg = cv2.imread(imgfile) 
            boxes = boxes_allviews[str(camid)]
            if len(boxes) == 0:
                keypoints_dict.update({str(camid):[]})
                continue
            net_input, all_c, all_s = preprocess(rawimg, boxes, mytransforms=mytransforms, image_size=cfg.MODEL.IMAGE_SIZE)
            net_out = model(net_input) # heatmaps [N,15,288,384]
            num_samples = len(boxes) 
            all_preds = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 3), dtype=np.float32) 
            all_boxes = np.zeros((num_samples, 6))

            preds, maxvals = get_final_preds(
                cfg, net_out.detach().clone().cpu().numpy(), all_c, all_s)
            all_preds[0: num_samples, :, 0:2] = preds[:, :, 0:2]
            all_preds[0: num_samples, :, 2:3] = maxvals
            all_boxes[0: num_samples, 0:2] = all_c[:, 0:2]
            all_boxes[0: num_samples, 2:4] = all_s[:, 0:2]
            all_boxes[0: num_samples, 4] = np.prod(all_s*200, 1)
            all_boxes[0: num_samples, 5] = 1
            keypoints_dict.update({str(camid):all_preds.reshape(num_samples, cfg.MODEL.NUM_JOINTS*3).tolist()})

            if is_vis: 
                vis = draw_keypoints(rawimg, all_preds, conf_thres=0.2, dataType=cfg.DATASET.DATASET) 
                vis_rgb = vis[:,:,(2,1,0)]
                cv2.namedWindow("hrnet", cv2.WINDOW_NORMAL)
                cv2.imshow("hrnet", vis) 
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
