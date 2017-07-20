#!/bin/bash

source ../../set_path.sh

CUDA_VISIBLE_DEVICES=1 python ../demo.py --train --data=box --motion_type=translation --method=unsupervised --train_epoch=1000 --test_interval=100 --test_epoch=10 --learning_rate=0.001 --batch_size=64 --image_size=32 --motion_range=1 --num_frame=3 --num_objects=2 2>&1 | tee train.log
