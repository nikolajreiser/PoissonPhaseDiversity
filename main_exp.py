#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:42:47 2020

@author: nikolaj
"""


import numpy as np
from src.iterate_poisson import iter_p
from src.iterate_gaussian import iter_g0
from src.zern import get_zern
import matplotlib.pyplot as plt
from src.image_functions import get_theta, scl, progress, errA, save_data
from src.fast_fft import Fast_FFTs
from skimage.transform import downscale_local_mean as dlm
from skimage.io import imread

#constants and conversion
pixelSize = .096
NA = 1.2
l = .532
pupilSize = NA/l
Sk0 = np.pi*(pixelSize*pupilSize)**2
l2p = 2*np.pi/l #length to phase
ansi2noll = np.array([4, 3, 5, 7, 8, 6, 9, 12, 13, 11, 14, 10]) - 1
signs = -1*np.array([0, 0, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1])

rotang = -70
num_c = 12
imsize = 512  #size of image fed to algorithms
dsize = 512 #size of image for preprocessing

if dsize == 512:
    x0 = 250 #best for dsize 512
    y0 = 300
    
if dsize == 256:
    x0 = 373 #best for dsize 256
    y0 = 333
scale = dsize//imsize

# x0 = 200
# y0 = 300

imgs_idx = [0, 1, 2, 3, 4]
num_imgs = len(imgs_idx)


#precompute zernike polynomials and ffts
zern, R, Theta, inds = get_zern(imsize, pupilSize, pixelSize*scale, num_c, rotang = rotang)
ff = Fast_FFTs(imsize, num_imgs, 1)


show = True

    
# Load data
dataset = "9"
datadir = '/home/nikolaj/Downloads/20200820_Beads_PR/Progressive_batch_50/'
coeffs = np.loadtxt(datadir+'beads_'+dataset+'_coeff.txt')

imgs = imread(datadir+'beads_'+dataset+'.tif')
imgs = np.float64(imgs)

imgs = imgs[imgs_idx, x0:x0+dsize, y0:y0+dsize]
imgs -= 300
imgs[imgs<.1] = 0
imgs = dlm(imgs, (1, scale, scale))
imgs = scl(imgs)

from src.image_functions import imshow
imshow(imgs[0], True)


coeffs = coeffs[:,:len(signs)]*signs
coeffs = coeffs[:,[ansi2noll]][:,0,:]
coeffs *= l2p

theta0 = np.zeros((num_imgs, num_c))
theta0[1:] = coeffs[imgs_idx[1:]]
theta1 = get_theta(theta0, zern)

phi = coeffs[0]
c0 = np.zeros((num_c))+1e-10

c1 = iter_g0(zern, inds, imgs, theta1, Sk0, c0, ff, 10, show)[0]
c2 = iter_p(zern, inds, imgs, theta1, Sk0, c0, ff, show)[0].copy()

errA(c2*0, phi, show)
errA(c1, phi, show)
errA(c2, phi, show)
