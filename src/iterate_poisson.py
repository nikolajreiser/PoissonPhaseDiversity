#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:31:21 2020

@author: nikolaj
"""

import numpy as np
import time
from numpy.linalg import norm
from src.image_functions import get_H2
from src.image_functions import sft, fft2, ift2, ift, progress
from deconvolution.deconv import get_reg

def iter_p(zern, inds, imgs, theta, Sk0, c, ff, show = False, eps = 1e-3):  #perform poisson iterations

    nc = len(c)
    
    if show: print("\n")
    start = time.time()
    
    #optimization parameters
    max_iter = 1000
    min_iter = 100
    max_iter_linesearch = 10 #max number of iterations for the line search
    
    ss = 3e4 #step size
    ss_reduce = .3 #amount to reduce step size by (multiplicatively) in line search
    
    b = 0 #background intensity
    lam = 1e-2 #regularization strength
    reg_option = 0 #regularization option

    #initialize fields
    norm_g = 1+eps
    f = np.ones(imgs[0].shape)
    f *= imgs.mean()
    dc = np.zeros((nc))

    cost = np.zeros((max_iter))
    sss = np.zeros((max_iter))
    c_all = np.zeros((1, len(c)))
    c_all[0] = c.copy()

    num_imgs = len(imgs)
    dim = imgs[0].shape
    
    
    L0 = -np.inf
    L1 = 0


    f0 = f.sum()
    
    #main loop
    n_iter = 0
    while True:
        
        #line search
        for i in range(max_iter_linesearch):
            
            
            #guess for new c
            c_temp = c-ss*dc

            #compute quantites required for cost function
            H = get_H2(zern, inds, dim, theta, c_temp)
            h = ff.ift(H)  
            
            s = np.abs(h)**2
            S = ff.fft2(s)
        
            F = ff.fft2(f)
        
            G = F*S
            g = ff.ift2(G)+b
            
            #compute cost function
            L1 = (imgs*np.log(g) - g).mean()
    
                        
            #if cost function is increasing, step size is good,
            #and the line search can be exited
            if L1 > L0: break
        
            #if cost function is decreasing, step size must be reduced
            else:   ss *= ss_reduce
            
            
        #update coefficients and current cost function value
        L0 = L1
        c = c-ss*dc
        
        #update object estimation, using values computed on the last (and
        #therefore successful) iteration of line search
        q = imgs/(g-lam*get_reg(f, reg_option))
        Q = ff.fft2(q)
        
        dF = (np.conj(S)*Q).sum(axis = 0)
        df = ff.ift2(dF)/(Sk0*num_imgs)
        f *= df
        
        #normalize object estimate if background or regularizer are being used
        #also helps with instability
        f = f*(f0/f.sum())
                
        #find new search direction dc        
        temp1 = np.conj(h)*ff.ift2(Q*np.conj(F))
        temp2 = np.imag(H*ff.ift(temp1)).sum(axis = 0)
        
        dc_integral = temp2[inds] * zern    
        dc = 2*dc_integral.sum(axis = 1)/(dim[0]*dim[1])
                               
        #Stopping conditions
        if n_iter>=max_iter: break
    
        if n_iter>min_iter:
            
            #terminate if total step size is small
            norm_g = norm(ss*dc)
            if norm_g < eps: break
            
        cost[n_iter] = L1
        sss[n_iter] = norm_g
        c_all = np.vstack((c_all, c))

        
        n_iter +=1
        if show: progress(f"{n_iter} iterations")

    end = time.time()
    if show: print(f"\nRuntime: {(end-start):.2f} seconds")

    
    cost = cost[:n_iter]
    sss = sss[:n_iter]
    
    return c, c_all, cost, [sss, f, end-start]