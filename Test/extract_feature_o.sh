mkdir -p ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg
CUDA_VISIBLE_DEVICES=0 python extract_feature_pattern_condition_first.py \
      --bind ./bind_original.pkl \
      --image_dir /path/to/original_images/ \
      --image_dir_support /path/to/original_images/ \
      --image_dir_support_o /path/to/original_images/ \
      --o ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/original_v1.hdf5 \
      --model vit_base_pattern_condition_first_dgg \
      --checkpoint vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar --imsize 224 
