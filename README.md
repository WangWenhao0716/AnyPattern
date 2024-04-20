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
--data-dir /path/to/ \
--logs-dir logs/anypattern_v31/vit_two_losses_m0.6_ddpmm_8gpu_512_10000_torch2_pattern_condition_first_dgg \
--height 224 --width 224 \
--multiprocessing-distributed --world-size 1
```
or just ```bash train.sh```.

The training can be finished in 14 hours with 8 A100 GPUs.

To promote reproduction, we also provide the [training log](https://github.com/WangWenhao0716/AnyPattern/blob/main/Train/train_log.txt) and our [trained model](https://huggingface.co/datasets/WenhaoWang/AnyPattern/blob/main/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar).

## Test

With the trained model ```vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar```, we can perform test step by step:

### Extract features of reference images
```
mkdir -p ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg
CUDA_VISIBLE_DEVICES=0 python extract_feature_pattern_condition_first.py \
      --bind ./bind_reference.pkl \
      --image_dir /path/to/reference_images/ \
      --image_dir_support /path/to/reference_images/ \
      --image_dir_support_o /path/to/reference_images/ \
      --o ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/reference_v1.hdf5 \
      --model vit_base_pattern_condition_first_dgg  \
      --checkpoint vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar --imsize 224 
```
or just ``bash extract_feature_o.sh``.

### Extract features of original images
```
mkdir -p ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg
CUDA_VISIBLE_DEVICES=0 python extract_feature_pattern_condition_first.py \
      --bind ./bind_original.pkl \
      --image_dir /path/to/original_images/ \
      --image_dir_support /path/to/original_images/ \
      --image_dir_support_o /path/to/original_images/ \
      --o ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/original_v1.hdf5 \
      --model vit_base_pattern_condition_first_dgg \
      --checkpoint vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar --imsize 224 
```
or just ``bash extract_feature_o.sh``.

### Search image-replica (example) pair
```
CUDA_VISIBLE_DEVICES=0 python extract_feature_pattern_condition_first_pattern.py \
      --bind ./bind_anyquery.pkl \
      --image_dir /path/to/query_k1_10_v2_test/ \
      --image_dir_support /path/to/query_k1_10_v2_test/ \
      --image_dir_support_o /path/to/query_k1_10_v2_test/ \
      --o ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/query_v1_test_v2_pattern.hdf5 \
      --model vit_base_pattern_condition_first_dgg \
      --checkpoint vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar --imsize 224 

CUDA_VISIBLE_DEVICES=0 python extract_feature_pattern_condition_first_pattern.py \
      --bind ./bind_anyquery.pkl \
      --image_dir /path/to/query_k1_10_v2_support/ \
      --image_dir_support /path/to/query_k1_10_v2_support/ \
      --image_dir_support_o /path/to/query_k1_10_v2_support/ \
      --o ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/query_v1_support_v2_pattern.hdf5 \
      --model vit_base_pattern_condition_first_dgg \
      --checkpoint vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar --imsize 224 

CUDA_VISIBLE_DEVICES=0 python bind.py
```
or just ``bash bind_q.sh``.

### Extract features of query images
```
CUDA_VISIBLE_DEVICES=0 python extract_feature_pattern_condition_first.py \
      --bind ./bind_k1_pattern_dgg_top0_v2.pkl \
      --image_dir /path/to/query_k1_10_v2_test/ \
      --image_dir_support/path/to/query_k1_10_v2_support/ \
      --image_dir_support_o /path/to/query_k1_10_v2_support_ori/ \
      --o ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/query_v1_appro_v2.hdf5 \
      --model vit_base_pattern_condition_first_dgg  \
      --checkpoint vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar --imsize 224
```
or just ``bash extract_feature_q.sh``.

### Match and get the final result
```
CUDA_VISIBLE_DEVICES=0 python score_normalization.py \
    --query_descs ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/query_0_v1_appro_v2.hdf5\
    --db_descs ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/reference_{0..19}_v1.hdf5 \
    --train_descs ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/original_{0..19}_v1.hdf5 \
    --factor 2 --n 10 \
    --o ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/predictions_v1_appro_v2.csv \
    --reduction avg --max_results 500000

python compute_metrics.py \
--preds_filepath ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/predictions_v1_appro_v2.csv \
--gt_filepath ./gt_v1_k1_v2_test.csv
```
or just ``bash match.sh``

You should get:
```
Average Precision: 0.56822
Recall at P90    : 0.54044
Threshold at P90 : -0.140579
Recall at rank 1:  0.60594
Recall at rank 10: 0.63890
```
Note that this reproduced result is a bit different from the one (``56.65% μAP`` and ``60.86% recall@1``) reported in our paper due to the randomness.


## Citation
```
@inproceedings{
    wang2024AnyPattern,
    title={AnyPattern: Towards In-context Image Copy Detection},
    author={Wang, Wenhao and Sun, Yifan and Tan, Zhentao and Yang, Yi},
    booktitle={In submission},
    year={2024},
}
```

## Contact

If you have any questions, feel free to contact [Wenhao Wang](https://wangwenhao0716.github.io/) (wangwenhao0716@gmail.com).


