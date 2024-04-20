from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import pickle

class Preprocessor(Dataset):
    def __init__(self, dataset, reverse_labels, root=None, transform=None, mutual=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual
        self.reverse_labels = reverse_labels

        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, ptarget = self.dataset[index]
        
        ptarget_tuple = tuple(ptarget.tolist())
        med = random.choice(self.reverse_labels[ptarget_tuple])
        
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
            
        support_path = fpath.rsplit('/', 1)[0] + '/%s.jpg'
        support_image = support_path%med
        support_image_o = support_path%(med.split('_')[0] + '_0')
        
        #print(fpath, support_image, support_image_o)

        img = Image.open(fpath).convert('RGB')
        img_support = Image.open(support_image).convert('RGB')
        img_support_o = Image.open(support_image_o).convert('RGB')
        
        if fpath.endswith('_0.jpg'):
            img_support = img
            img_support_o = img

        if self.transform is not None:
            img = self.transform(img)
            img_support = self.transform(img_support)
            img_support_o = self.transform(img_support_o)
            
        return img, img_support, img_support_o, fname, support_image, support_image_o, pid, camid, ptarget

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, img_2, pid
