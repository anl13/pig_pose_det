#!/bin/bash 

# python pose_estimation/train.py \
#     --cfg experiments/pig/resnet50/384x384_d256x3_adam_lr1e-3.yaml

# CUDA_VISIBLE_DEVICES=6,7 python pose_estimation/train.py \
#     --cfg 'experiments/pig/hrnet_w48_384x384.yaml'

CUDA_VISIBLE_DEVICES=0,1,2,3 python pose_estimation/train.py \
      --cfg 'experiments/pig/w48_384x384_univ.yaml'
