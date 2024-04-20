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

In the ``Generate`` folder, we provide all the code for 100 patterns (90 base + 10 novel) in our AnyPattern dataset. Since it is prohibitively expensive (distributed on 200 CPU nodes for 1 million CPU core hours) to generate by yourselves, we have provided the generated images (i.e. ``anypattern_v31`` you downloaded). 
