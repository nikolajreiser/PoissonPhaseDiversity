#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:26:41 2020

@author: nikolaj
"""

import numpy as np
from numpy.random import rand, randint, poisson, normal, seed
from skimage import draw
from src.image_functions import fft, ift, ift2, fft2, sft, defocus, get_H2, scl_imgs
from src.zern import zernfun, pupil
from skimage.io import imread
from skimage import data
from skimage.filters import gaussian
from src.zern import normalize
from sys import stdout

def sim_im(ob3d, dim, phi, num_imgs, theta, zern, R, inds, trim = True):
    #create object for testing
    
    y = dim[0]
    x = dim[1]

    if len(ob3d.shape) == 3 :
        mid_plane = ob3d.shape[0]//2
        ob = ob3d[mid_plane]
    else:
        ob = ob3d
    
    if trim: ob = ob[x//2:-x//2, y//2:-y//2]
    
    imgs = np.zeros((num_imgs, y, x))

    F = fft2(ob)

    H = get_H2(zern, inds, dim, theta, phi)
    s = np.abs(ift(H))**2
    # from plotting import imshow
    # imshow(sft(s[-1]), cmap = 'gray')
    S = fft2(s)
    imgs = ift2(F*S)
    
    return imgs


def sim_im_2(ob3d, dim, phi, num_imgs, theta, zern, R, inds):
    #create object for testing
    
    y = dim[0]
    x = dim[1]
        
    
    if len(ob3d.shape) == 3 :
        mid_plane = ob3d.shape[0]//2
        ob = ob3d[mid_plane]
    else:
        ob = ob3d
    
    # ob = data.grass()
    # ob = np.float64(ob)
    ob -= ob.min()
    ob/=ob.max()
        
    imgs = np.zeros((num_imgs, y, x))
    
    F = fft2(ob)
    H = get_H2(zern, inds, dim, theta, phi)
    s = np.abs(ift(H))**2
    s = sft(np.pad(sft(s), ((0,0), (x//2, x//2), (y//2, y//2))))
    S = fft2(s)
    g = ift2(F*S)
    
    # from plotting import imshow
    # a = np.zeros((256, 256, 3))
    # a[:,:,0] = sft(s[-1])
    # a[:,:,1] = sft(s[-1])
    # a[:,:,2] = sft(s[-1])
    # a -= a.min()
    # a /= a.max()
    
    # a[64:-64, 64::128, 0] = 1 
    # a[64:-64, 64::128, 1:] = 0 
    # a[64::128, 64:-64, 0] = 1
    # a[64::128, 64:-64, 1:] = 0


    
    # imshow(a)

    imgs = g[:,x//2:-x//2, y//2:-y//2]
     
    return imgs

def sim_im_3(ob, dim, phi, num_imgs, theta, zern, R, inds, zern_mag, low_freq = False, show = True):
    
    y = dim[0]
    x = dim[1]
    
            
    ob -= ob.min()
    ob/=ob.max()
    obx = ob.shape[0]
    num_phi = len(phi)
    imgs = np.zeros((num_imgs, obx, obx))

    #create spatially variant zernike amplitudes
    if low_freq:
        X, Y = np.meshgrid(np.arange(obx), np.arange(obx))
        Xrand, Yrand = rand(2, num_phi)*2 - 1
        zern_amplitudes = X[None,:,:]*Xrand[:, None, None] + Y[None,:,:]*Yrand[:, None, None]
        
    else:    
        zern_amplitudes = rand(num_phi, obx, obx)*2 - 1
        zern_amplitudes = gaussian(rand(obx, obx, num_phi), obx/100).transpose((2, 1, 0))
        
    zern_amplitudes -= zern_amplitudes[:, x//2:-x//2, y//2:-y//2].mean(axis = (1, 2))[:, None, None]
    
    zern_avg = np.linalg.norm(zern_amplitudes*np.sqrt(normalize(np.arange(num_phi)))[:, None, None], axis = (0))
    zern_avg = np.mean(zern_avg)
    zern_amplitudes *= 2*np.pi*zern_mag/zern_avg

    for i in range(obx):
        if show:
            stdout.write("\r"+f"{i}/{obx}")
            stdout.flush()

        
        for j in range(obx):
            H = get_H2(zern, inds, dim, theta, phi+zern_amplitudes[:, i, j])
            s = np.abs(ift(H))**2
            s = sft(np.pad(sft(s), ((0,0), (x//2, x//2), (y//2, y//2))))
            s = np.roll(s, (i, j), (1, 2))
            
            imgs += ob[None, i, j]*s
    
    
    imgs = imgs[:,x//2:-x//2, y//2:-y//2]
     
    return imgs


def add_noise(imgs, num_photons = 1000, dark_noise = 50, read_noise = 10, qe = .6):
    
    dim = imgs.shape
    imgs[imgs<0] = 0 #make sure there are no negative values
    
    photons = qe*num_photons*imgs/imgs.mean()
    shot_noise = poisson(lam = photons)
    
    dark_current_im = poisson(lam = dark_noise, size = dim)
    read_noise_im = normal(scale = read_noise, size = dim)
    
    noise_imgs = shot_noise + dark_current_im + read_noise_im
    
    photon_snr = num_photons*qe
    snr = photon_snr/np.sqrt(photon_snr+dark_noise+read_noise**2)

    #note that images are unnormalized
    return snr, noise_imgs