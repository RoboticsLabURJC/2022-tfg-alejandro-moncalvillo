python3 balance_dataset.py --directory 'simple_holonomic'

python3 train.py --data_dir './datasets/simple_holonomic/' --data_dir './datasets/counter_nurburgring_holonomic/' --data_dir './datasets/montreal_holonomic/' --data_dir './datasets/difficult_situations/difficult_6_holonomic/' --data_dir './datasets/difficult_situations/difficult_7_holonomic/' --data_dir './datasets/difficult_situations/difficult_2_holonomic/' --test_dir './datasets/montmelo_holonomic/' --preprocess 'crop' --data_augs 'gaussian' --base_dir testcase --num_epochs 250  --lr 1e-3 --shuffle True --batch_size 128 --save_iter 50 --print_terminal True --seed 471


python3 train.py --data_dir './simple_ackermann_w_reduced/' --test_dir './simple_ackermann_w_reduced/' --preprocess 'crop' --data_augs 'gaussian' --base_dir testcase --num_epochs 100  --lr 1e-3 --shuffle True --batch_size 128 --save_iter 70 --print_terminal True --seed 533