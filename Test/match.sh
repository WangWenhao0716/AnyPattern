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
