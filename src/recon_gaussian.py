#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 06:22:34 2020

@author: nikolaj
"""

import numpy as np
from src.image_functions import fft, ift, fft2, ift2, sft, get_H2
import matplotlib.pyplot as plt

def update_gaussian(c, args):
    
    
    grad = update_gaussian_grad(c, args)
    
    H = get_Hgn(c, args)
        
    dc = np.linalg.inv(H)@grad

    return dc

def get_grad(H, h, V, zern, inds, Sk0, ff):
    
    grad_phi_sum = np.imag(np.conj(H)*ff.fft(h*np.real(ff.ift2(V))))
    grad_phi = -2*np.sum(grad_phi_sum, axis = 0)[inds]
    
    grad = np.sum(zern*grad_phi, axis = 1)
    return grad

def get_Hgn(c, args):
    
    D, zern, inds, theta, reg, Sk0, dim, ff = args
    
    N = len(zern)
    Hgn = np.zeros((N, N))
    
    H = get_H2(zern, inds, dim, theta, c)
    h = ff.ift(H)
    S = ff.fft2(np.abs(h)**2)

    for n in range(N):
        Hgn_mn = get_Hgn_mn(D, zern[n], inds, S, H, h, reg, ff)
        
        for m in range(N):
            Hgn[m,n] = np.sum(Hgn_mn[inds]*zern[m])
            
    return Hgn
    
def get_J(c, args):
    
    D, zern, inds, theta, reg, Sk0, dim, ff = args
    
    S = np.zeros(D.shape, dtype = np.complex128)
    
    H = get_H2(zern, inds, dim, theta, c)
    h = ff.ift(H)
    S = ff.fft2(np.abs(h)**2)

    F = get_F(S, D, reg)
    G = F*S
    dif = (np.abs(G-D)**2).sum(axis = 0)
    reg_term = np.abs(reg*F)**2
    J = (dif+reg_term).mean()

    return J/2


def update_gaussian_grad(c, args):
    
    D, zern, inds, theta, reg, Sk0, dim, ff = args
            
    V = np.zeros(D.shape, dtype = np.complex128)
    
    H = get_H2(zern, inds, dim, theta, c)
    h = ff.ift(H)
    S = ff.fft2(np.abs(h)**2)

    F = get_F(S, D, reg)
    V = get_V(F, S, D, reg)

    grad = get_grad(H, h, V, zern, inds, Sk0, ff)
    
    return grad

def get_F(S, D, reg):
    
    n = np.sum(np.conj(S)*D, axis = 0)
    d = reg+np.sum(np.abs(S)**2, axis = 0)
    
    return n/d

def get_V(F, S, D, reg):
    
    V = np.conj(F)*D-S*np.abs(F)**2
    return V


def get_Hgn_mn(D, zn, inds, S, H, h, reg, ff):
    
    K = len(D)
    Hgn_mn = np.zeros(H[0].shape)
    
    Q = reg + np.sum(np.abs(S)**2, axis = 0)
    DQ = D/np.sqrt(Q)
    DQc = np.conj(DQ)
    
    Hc = np.conj(H)
    hc = np.conj(h)

    Hz = np.zeros(H.shape, dtype = np.complex128)
    Hz[:, inds[0], inds[1]] = H[:, inds[0], inds[1]]*zn
    for k in range(K):
        for j in range(k):
                    
            Dj = DQ[j]
            Dk = DQ[k]
            Dcj = DQc[j]
            Dck = DQc[k]

            U1 = Dj*ff.fft2(np.imag(hc[k]*ff.ift(Hz[k])))
            U2 = Dk*ff.fft2(np.imag(hc[j]*ff.ift(Hz[j])))
            Ujk = U1-U2

            H1 = Hc[j]*ff.fft(h[j]*ff.ift2(Dck*Ujk))
            H2 = Hc[k]*ff.fft(h[k]*ff.ift2(Dcj*Ujk))
            
            Hgn_mn += np.imag(H1-H2)
                
    return 4*Hgn_mn