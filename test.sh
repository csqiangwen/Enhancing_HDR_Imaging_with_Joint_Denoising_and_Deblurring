## For quantiative evaluation
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python3 ./test.py \
--gpu_ids 0 \
--which_iter 132000 \
--checkpoints_dir checkpoints_hdr \
--hdr_dararoot path_to_test_set \