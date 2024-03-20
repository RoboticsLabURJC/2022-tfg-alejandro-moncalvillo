python3 balance_dataset.py --directory 'simple_holonomic'

python3 train.py --data_dir './datasets/simple_holonomic/' --data_dir './datasets/counter_nurburgring_holonomic/' --data_dir './datasets/montreal_holonomic/' --data_dir './datasets/difficult_situations/difficult_6_holonomic/' --data_dir './datasets/difficult_situations/difficult_7_holonomic/' --data_dir './datasets/difficult_situations/difficult_2_holonomic/' --test_dir './datasets/montmelo_holonomic/' --preprocess 'crop' --data_augs 'gaussian' --base_dir testcase --num_epochs 250  --lr 1e-3 --shuffle True --batch_size 128 --save_iter 50 --print_terminal True --seed 471


python3 train.py --data_dir './datasets/curves_holonomic_track/' --data_dir './datasets/montmelo_holonomic_track/' --data_dir './datasets/montreal_holonomic_track/' --data_dir --test_dir './datasets/simple_holonomic_track/' --preprocess 'crop' --data_augs 'gaussian' --base_dir testcase --num_epochs 200  --lr 1e-3 --shuffle True --batch_size 128 --save_iter 20 --print_terminal True --seed 44
