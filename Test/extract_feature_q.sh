CUDA_VISIBLE_DEVICES=0 python extract_feature_pattern_condition_first.py \
      --bind ./bind_k1_pattern_dgg_top0_v2.pkl \
      --image_dir /path/to/query_k1_10_v2_test/ \
      --image_dir_support/path/to/query_k1_10_v2_support/ \
      --image_dir_support_o /path/to/query_k1_10_v2_support_ori/ \
      --o ./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/query_v1_appro_v2.hdf5 \
      --model vit_base_pattern_condition_first_dgg  \
      --checkpoint vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg.pth.tar --imsize 224 
