# AnyPattern
The official implementation of "AnyPattern: Towards In-context Image Copy Detection"

![image](https://github.com/WangWenhao0716/AnyPattern/blob/main/AnyPattern.jpg)

## Prepare Datasets

We release our datasets on [Hugging Face](https://huggingface.co/datasets/WenhaoWang/AnyPattern). Please follow the instructions on Hugging Face to download the dataset. After downloading and unzipping, we should have:

```
/path/to/

  anypattern_v31/
    anypattern_v31/
      0_0.jpg
      0_1.jpg
      ...

  original_images/
     T000000.jpg
     T000001.jpg
     ...

  reference_images/
     R000000.jpg
     R000001.jpg
     ...

  query_k1_10_v2_test/
     ...

  query_k1_10_v2_support_select10/
     ...

  query_k1_10_v2_support_ori_select10/
     ...
```

## Generate 

In the ``Generate`` folder, we provide all the code (~3000 lines) for 100 patterns (90 base + 10 novel) in our AnyPattern dataset. Since it is prohibitively expensive (distributed on 200 CPU nodes for 1 million CPU core hours) to generate by yourselves, we have provided the generated images (i.e. ``anypattern_v31`` you downloaded). 

## Train

Please first go to the ``Train`` folder by ``cd Train``, then you can train a new model by
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=gpu-13 \
train_single_source_gem_coslr_wb_balance_cos_ema_ddpmm_pattern_condition_torch2.py \
-ds anypattern_v31 -a vit_base_pattern_condition_first_dgg --margin 0.0 \
--num-instances 4 -b 512 -j 64 --warmup-step 5 \
--lr 0.00035 --iters 10000 --epochs 25 \
--data-dir /path/to/images/ \
--logs-dir logs/anypattern_v31/vit_two_losses_m0.6_ddpmm_8gpu_512_10000_torch2_pattern_condition_first_dgg \
--height 224 --width 224 \
--multiprocessing-distributed --world-size 1
```
or just ```base train.sh```.

The training can be finished in 14 hours with 8 A100 GPUs.

To promote reproduction, we also provide the [training log](https://github.com/WangWenhao0716/AnyPattern/blob/main/Train/train_log.txt) and our [trained model](https://huggingface.co/datasets/WenhaoWang/AnyPattern/blob/main/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar).
