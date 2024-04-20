mkdir -p ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg
CUDA_VISIBLE_DEVICES=0 python extract_feature_pattern_condition_first.py \
      --bind ./bind_reference.pkl \
      --image_dir /path/to/reference_images/ \
      --image_dir_support /path/to/reference_images/ \
      --image_dir_support_o /path/to/reference_images/ \
      --o ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/reference_v1.hdf5 \
      --model vit_base_pattern_condition_first_dgg  \
      --checkpoint vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar --imsize 224 
