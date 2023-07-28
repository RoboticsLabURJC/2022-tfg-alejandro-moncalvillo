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

