from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob
import pandas as pd
import torch
import numpy as np
import pickle

class Anypattern_v31(object):

    def __init__(self, root, rank, combine_all=True):
        
        self.images_dir = '' 
        self.rank = rank
        self.img_path = osp.join(root) + '/anypattern_v31'
        self.train_path = self.img_path
        self.gallery_path = ''
        self.query_path = ''
        self.train = []
        self.gallery = []
        self.query = []
        self.num_train_pids = 0
        self.has_time_info = False
        
        with open('anypattern_v31.pkl', 'rb') as f:
            self.labels = pickle.load(f)
        
        self.load()

    def preprocess(self):
        fpaths = sorted(glob(osp.join(self.images_dir, self.train_path, '*g')))
        data = []
        all_pids = {}

        for fpath in fpaths:
            fname = osp.basename(fpath)  # filename: 000001_s10_c01_f000295.jpg #299_1.jpg
            fields = fname.split('_')
            pid = int(fields[0])
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]  # relabel
            camid = 0
            
            try:
                target_pattern = torch.zeros([90], dtype=torch.float32)
                select = self.labels[fname.split('.')[0]]
                target_pattern[select] = 1.0
            except:
                target_pattern = torch.zeros([90], dtype=torch.float32)
            
            data.append((self.img_path + '/' + fname, pid, camid, target_pattern))
        return data, int(len(all_pids))

    def load(self):
        self.train, self.num_train_pids = self.preprocess()

        if self.rank==0:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset | # ids | # images")
            print("  ---------------------------")
            print("  all    | {:5d} | {:8d}"
                  .format(self.num_train_pids, len(self.train)))