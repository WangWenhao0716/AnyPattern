from isc.io import read_descriptors
name_s, feature_s = read_descriptors(['./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/query_0_v1_support_v2_pattern.hdf5'])
name_t, feature_t = read_descriptors(['./feature/vit_ddpmm_8gpu_512_torch2_ap31_pattern_condition_first_dgg/query_0_v1_test_v2_pattern.hdf5'])
import torch
feature_s = torch.from_numpy(feature_s).cuda()
feature_t = torch.from_numpy(feature_t).cuda()
matrix = feature_t@feature_s.T
values, indices = matrix.topk(10, dim=1)
largest_indices = indices[:, 0]
largest_values = values[:, 0]
bind = dict()
for num in range(25000):
    bind[name_t[num]] = name_s[largest_indices[num]]

import pickle
with open('bind_k1_pattern_dgg_top0_v2.pkl', 'wb') as f:
    pickle.dump(bind, f)
