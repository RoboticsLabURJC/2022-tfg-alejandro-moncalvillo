python3 balance_dataset.py --directory 'simple_holonomic'


python3 train.py --data_dir '../dataset_opencv/simple_holonomic/' --data_dir '../dataset_opencv/counter_simple_holonomic/' --data_dir '../dataset_opencv/counter_montmelo_holonomic/' --data_dir '../dataset_opencv/counter_nurburgring_holonomic/' --data_dir '../dataset_opencv/many_curves_holonomic/' --data_dir '../dataset_opencv/difficult_situations/difficult_1_holonomic/' --data_dir '../dataset_opencv/difficult_situations/difficult_2_holonomic/' --data_dir '../dataset_opencv/difficult_situations/difficult_3_holonomic/' --data_dir '../dataset_opencv/difficult_situations/difficult_4_holonomic/' --data_dir '../dataset_opencv/difficult_situations/difficult_8_holonomic/' --data_dir '../dataset_opencv/difficult_situations/difficult_6_holonomic/' --data_dir '../dataset_opencv/difficult_situations/difficult_7_holonomic/' --test_dir '../dataset_opencv/montreal_holonomic/' --preprocess 'crop' --data_augs 'gaussian' --base_dir testcase --num_epochs 100  --lr 1e-3 --shuffle True --batch_size 128 --save_iter 70 --print_terminal True --seed 211



python3 train.py  --data_dir '../dataset_opencv/difficult_situations/difficult_6_holonomic/' --data_dir '../dataset_opencv/difficult_situations/difficult_7_holonomic/' --test_dir '../dataset_opencv/montreal_holonomic/' --preprocess 'crop' --base_dir testcase --num_epochs 50  --lr 1e-3 --shuffle True --batch_size 128 --save_iter 30 --print_terminal True --seed 235




python3 train.py --data_dir './simple_ackermann_w_reduced/' --test_dir './simple_ackermann_w_reduced/' --preprocess 'crop' --data_augs 'gaussian' --base_dir testcase --num_epochs 100  --lr 1e-3 --shuffle True --batch_size 128 --save_iter 70 --print_terminal True --seed 533