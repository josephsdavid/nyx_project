#!/usr/bin/env bash
#SBATCH -J recsys
#SBATCH -p v100x8 --gres=gpu:2 --mem=60G
#SBATCH -e result/stdout%j.txt
#SBATCH -o result/stdout%j.txt
#SBATCH -t 120
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL
#SBATCH -s

python3 main_ae.py --trial_name example \
	--experiment_name debug \
	--optimizer sgd \
	--gpus 2 \
	--distributed_backend dp \
	--auto_lr_find lr \
	--precision 16 \
	--batch_size 64 \
	--scheduler none\
	--max_epochs 1000 \
	--min_epochs 100 \
	--val_check_interval 0.5 \
