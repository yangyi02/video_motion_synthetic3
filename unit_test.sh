#!/bin/bash

source ./set_path.sh

python data/box_data.py --num_objects=4 --num_frame=5 --batch_size=32 --image_size=32 --motion_range=3 --bg_move --save_display --save_display_dir=./

python data/mnist_data.py --num_objects=4 --num_frame=5 --batch_size=32 --image_size=32 --motion_range=3 --save_display --save_display_dir=./

python base_demo.py --test_gt --motion_type=translation --test_epoch=1 --batch_size=32 --motion_range=2 --image_size=32 --num_frame=3 --display --save_display --save_display_dir=./

python base_demo.py --train --data=box --method=unsupervised --save_dir=./ --train_epoch=5 --test_interval=5 --test_epoch=1 --learning_rate=0.001 --batch_size=32 --image_size=32 --motion_range=1 --num_frame=3

python base_demo.py --test --init_model=model.pth --test_epoch=1 --batch_size=64 --motion_range=1 --image_size=32 --num_frame=3 --display --save_display --save_display_dir=./

sh trim.sh

rm model.pth
