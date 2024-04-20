torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=gpu-13 \
train_single_source_gem_coslr_wb_balance_cos_ema_ddpmm_pattern_condition_torch2.py \
-ds anypattern_v31 -a vit_base_pattern_condition_first_dgg --margin 0.0 \
--num-instances 4 -b 512 -j 64 --warmup-step 5 \
--lr 0.00035 --iters 10000 --epochs 25 \
--data-dir /path/to/ \
--logs-dir logs/anypattern_v31/vit_two_losses_m0.6_ddpmm_8gpu_512_10000_torch2_pattern_condition_first_dgg \
--height 224 --width 224 \
--multiprocessing-distributed --world-size 1
