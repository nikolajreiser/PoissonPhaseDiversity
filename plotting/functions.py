#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:25:32 2021

@author: nikolaj
"""

import pathlib
import os
import numpy as np

def load_data(data_dir, v = -1):
    
    #define path to folder within data folder
    p = pathlib.Path('data')/data_dir

    #get list of all files and folders
    dirs = os.listdir(p)
    
    #filter only data_vXX folders
    dirs = [d for d in dirs if d[:6] == 'data_v']
    
    #get highest version folder (or folder corresponding to v)
    dir_nums = [int(d.split('v')[1]) for d in dirs]

    if v == -1: dir_num = max(dir_nums)
    else: dir_num = v
    
    #find path corresponding to folder
    folder = f'data_v{dir_num}'
    p = p/folder
    
    data_files = os.listdir(p)
    data_files = sorted(data_files)
    datas = []
    for d in data_files:
        datas.append(np.load(p/d))
    
    return datas

def bin_data(inds, bins, data):
    
    out = [[] for i in range(len(bins)-1)]
    for(ind, val) in zip(inds, data):
        out[ind-1].append(val)
        
    return out

def moving_average(a, w=3) :
    n = len(a)
    return [a[max(0, i-w):min(n, i+w)].mean() for i in range(n)]