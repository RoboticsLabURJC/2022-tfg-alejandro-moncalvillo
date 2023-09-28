python3 train.py --data_dir './montreal_data/' \
		--data_dir './simple_data/' \
		--data_dir './manycurves_data/' \
		--test_dir './montmelo_data/' \
		--preprocess 'crop' \
		--preprocess 'extreme' \
	    --base_dir testcase \
	    --comment 'Selected Augmentations: gaussian, affine' \
	    --data_augs 'gaussian' \
	    --data_augs 'affine' \
	    --num_epochs 30 \
	    --lr 1e-3 \
	    --shuffle True \
	    --batch_size 128 \
	    --save_iter 25 \
	    --print_terminal True \
	    --seed 145



python3 train.py --data_dir './montreal_data/' \
		--data_dir './simple_data/' \
		--data_dir './manycurves_data/' \
		--test_dir './montmelo_data/' \
		--preprocess 'crop' \
	    --base_dir testcase \
	    --comment 'Selected Augmentations: gaussian, affine' \
	    --data_augs 'gaussian' \
	    --data_augs 'affine' \
	    --num_epochs 30 \
	    --lr 1e-3 \
	    --shuffle True \
	    --batch_size 128 \
	    --save_iter 25 \
	    --print_terminal True \
	    --seed 178


python train.py --data_dir '../dataset_opencv/extended_simple_circuit_01_04_2022_anticlockwise_1/' \
		--data_dir '../dataset_opencv/extended_simple_circuit_01_04_2022_clockwise_1/' \
		--data_dir '../dataset_opencv/many_curves_01_04_2022_anticlockwise_1/' \
		--data_dir '../dataset_opencv/many_curves_01_04_2022_clockwise_1/' \
		--data_dir '../dataset_opencv/monaco_01_04_2022_anticlockwise_1/' \
		--data_dir '../dataset_opencv/monaco_01_04_2022_clockwise_1/' \
		--data_dir '../dataset_opencv/nurburgring_01_04_2022_anticlockwise_1/' \
		--data_dir '../dataset_opencv/nurburgring_01_04_2022_clockwise_1/' \
		--preprocess 'crop' \
		--preprocess 'extreme' \
	    --base_dir testcase \
	    --comment 'Selected Augmentations: gaussian, affine' \
	    --data_augs 'gaussian' \
	    --data_augs 'affine' \
	    --num_epochs 150 \
	    --lr 1e-3 \
	    --test_split 0.2 \
	    --shuffle True \
	    --batch_size 128 \
	    --save_iter 50 \
	    --print_terminal True \
	    --seed 122  