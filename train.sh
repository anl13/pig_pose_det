#!/bin/bash 

CUDA_VISIBLE_DEVICES=0 python pose_estimation/train.py \
      --cfg 'experiments/pig/w48_384x384_univ_example.yaml'
