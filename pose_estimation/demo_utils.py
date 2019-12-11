
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 
import numpy as np 
import sys 
import os 
import torch 
import collections
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform

def load_state_dict_module(model, state_file):
    D = torch.load(state_file) 
    newD = collections.OrderedDict() 
    for k,v in D.items():
        if 'module' in k: 
            newk = k[7:]
        else: 
            newk = k 
        newD.update({newk:v})
    model.load_state_dict(newD) 

def xywh2cs(x, y, w, h):
    aspect_ratio = 1920.0 / 1080.0 
    pixel_std = 200 
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    return center, scale

def box_xywh2cs(box_xywh):
    x,y,w,h = box 
    return xywh2cs(x,y,w,h) 

def box_xyxy2cs(box_xyxy):
    x1,y1,x2,y2 = box_xyxy 
    return xywh2cs(x1,y1,x2-x1,y2-y1)

# [input] img: img loaded by cv2.imread(filename) 
def preprocess(img, boxes, mytransforms=None, image_size=[384, 288]):
    data_numpy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_crops = torch.zeros((len(boxes), 3, image_size[1], image_size[0])) 
    all_c = [] 
    all_s = [] 
    for idx, box in enumerate(boxes):  
        c,s = box_xyxy2cs(box) 
        r = 0 
        trans = get_affine_transform(c,s,r, image_size)
        out = cv2.warpAffine(
            data_numpy,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
        if mytransforms:
            out = mytransforms(out) 
        out = out.view([1,3,image_size[1], image_size[0]])
        all_crops[idx:idx+1, :,:,:] = out 
        all_c.append(c) 
        all_s.append(s) 

    all_c = np.asarray(all_c) 
    all_s = np.asarray(all_s) 

    return all_crops, all_c, all_s 

COLORS = [
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,255,0],
    [255,0,255],
    [0,255,255],
    [128, 0, 0],
    [0,128,0],
    [0,0,128],
    [128,128,0],
    [128,0,128],
    [0,128,128]
]

# img: [H,W,C]
# preds: [N, 15, 3]
SKEL = {
    "pig": [[0, 2], [1, 2], [2, 14], [5, 6], [5, 14], [3, 4], [3, 14],
    [13, 14], [9, 8], [8, 7], [7, 13], [12, 11], [11, 10], [10, 13]], 
    "pig20" : [[0,1], [1,2], [0,10], [10,11], [0,19], [19,18], [18,9], 
        [19,12], [19, 3], [9, 15], [9,6], [12, 13], [13,14], 
        [15,16], [16,17], [3,4], [4,5], [6,7], [7,8] ]
}

def draw_keypoints(img, preds, conf_thres=0.8, dataType="pig"):
    out = img.copy() 
    for idx, pred in enumerate(preds):
        color = COLORS[idx]
        # draw points 
        for point in pred:
            x,y,c = point 
            if c < conf_thres: 
                continue 
            size = int(10 * c) + 1 
            cv2.circle(out, (int(x), int(y)), size, color, -1) 
        for skel in SKEL[dataType]: 
            p1 = pred[skel[0]]
            p2 = pred[skel[1]]
            if p1[2] < conf_thres or p2[2] < conf_thres: 
                continue 
            cv2.line(out, (int(p1[0]), int(p1[1])), 
                (int(p2[0]), int(p2[1])), color, 4)
    return out 
