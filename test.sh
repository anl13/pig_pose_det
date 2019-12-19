#!/bin/bash 

# CUDA_VISIBLE_DEVICES=0 python pose_estimation/test.py \
#     --cfg experiments/pig/hrnet_w48_384x384.yaml

CUDA_VISIBLE_DEVICES=0 python pose_estimation/test.py \
    --cfg experiments/pig/w48_384x288-tiger.yaml
