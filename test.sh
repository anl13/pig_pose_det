#!/bin/bash 

CUDA_VISIBLE_DEVICES=0 python pose_estimation/test.py \
    --cfg experiments/iccv2019/w48_384x384_univ_iccv2019.yaml
