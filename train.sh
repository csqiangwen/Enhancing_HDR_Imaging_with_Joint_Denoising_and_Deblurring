OMP_NUM_THREADS=8 python3 ./train.py --gpu_ids 0 --batchSize 1 --save_freq 1000 --niter_decay 25000 --shuffle
#  CUDA_VISIBLE_DEVICES=5