# AnyPattern
[Arxiv, 2024] The official implementation of "AnyPattern: Towards In-context Image Copy Detection"

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
