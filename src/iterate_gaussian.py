#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:49:43 2020

@author: nikolaj
"""

import numpy as np
import time
from src.recon_gaussian import get_J, update_gaussian,  get_F
from src.image_functions import fft2, get_H2, ift2, ift, progress
from numpy.linalg import norm
from scipy.optimize import minimize


def iter_g0(zern, inds, imgs, theta, Sk0, c0, ff, max_iter = 10, show = False):  #perform gaussian iterations
    if show: print("\n")
    
    start = time.time()
    c = c0.copy()
    #gaussian parameters
    D = fft2(imgs)
    
    gamma = 1e-20
    
    reg = gamma
    dim = imgs[0].shape
    args = [D, zern, inds, theta, reg, Sk0, dim, ff]
    

    num_c = len(zern)
    c_all = np.zeros((1, num_c))
    c_all[0] = c.copy()
    g_cost = np.zeros((1))
    g_cost[0] = get_J(c, args)
    

    done_val = 1e-2
    finished = False
    min_iter = 2

    i = 1
    while not finished:
        if show: progress(str(i)+" iterations")
        dc = update_gaussian(c, args)
        c -= dc
        c_all = np.vstack((c_all, c))
        g_cost = np.append(g_cost,
                get_J(c, args))
        i += 1
        finished = ((norm(dc) < done_val) or (i > max_iter)) and i > min_iter
        
    end = time.time()
    if show: print(f"\nRuntime: {(end-start):.2f} seconds")
    
    H = get_H2(zern, inds, dim, theta, c)
    h = ift(H)
    S = fft2(np.abs(h)**2)
    F = get_F(S, D, reg)
    ob = ift2(F)

    return c, ob, [c_all, end-start]#, c_all
