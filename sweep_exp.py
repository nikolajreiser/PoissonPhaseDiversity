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
from src.image_functions import get_theta, scl_imgs, progress, errA, save_data
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

imgs_idx_1 = [0, 1, 2]
num_imgs_1 = len(imgs_idx_1)

imgs_idx_2 = [0, 2, 4]
num_imgs_2 = len(imgs_idx_2)

imgs_idx_3 = [0, 1, 3]
num_imgs_3 = len(imgs_idx_3)

imgs_idx_4 = [0, 1, 2, 3, 4]
num_imgs_4 = len(imgs_idx_4)


#precompute zernike polynomials and ffts
zern, R, Theta, inds = get_zern(imsize, pupilSize, pixelSize*scale, num_c, rotang = rotang)
ff_1 = Fast_FFTs(imsize, num_imgs_1, 1)
ff_2 = Fast_FFTs(imsize, num_imgs_4, 1)

#aggregate data
nd = 50
srp1 = np.zeros((nd))
srg1 = np.zeros((nd))
srp2 = np.zeros((nd))
srg2 = np.zeros((nd))
srp3 = np.zeros((nd))
srg3 = np.zeros((nd))
srp4 = np.zeros((nd))
srg4 = np.zeros((nd))
sri = np.zeros((nd))

show = False

for i in range(nd):
    progress(f'{i+1}/{nd}')
    
    # Load data
    dataset = str(i)
    datadir = '/home/nikolaj/Downloads/20200820_Beads_PR/Progressive_batch_50/'
    coeffs = np.loadtxt(datadir+'beads_'+dataset+'_coeff.txt')
    
    imgs = imread(datadir+'beads_'+dataset+'.tif')
    imgs = np.float64(imgs)
    
    imgs = imgs[:,x0:x0+dsize, y0:y0+dsize]
    imgs -= 300
    imgs[imgs<.1] = 0
    imgs = dlm(imgs, (1, scale, scale))
    imgs = scl_imgs(imgs)

    from src.image_functions import imshow
    imshow(imgs[0], True)
    
    
    imgs1 = imgs[imgs_idx_1]
    imgs2 = imgs[imgs_idx_2]
    imgs3 = imgs[imgs_idx_3]
    imgs4 = imgs[imgs_idx_4]

    coeffs = coeffs[:,:len(signs)]*signs
    coeffs = coeffs[:,[ansi2noll]][:,0,:]
    coeffs *= l2p
    
    theta01 = np.zeros((num_imgs_1, num_c))
    theta01[1:] = coeffs[imgs_idx_1[1:]]
    theta1 = get_theta(theta01, zern)
    
    theta02 = np.zeros((num_imgs_2, num_c))
    theta02[1:] = coeffs[imgs_idx_2[1:]]
    theta2 = get_theta(theta02, zern)
    
    theta03 = np.zeros((num_imgs_3, num_c))
    theta03[1:] = coeffs[imgs_idx_3[1:]]
    theta3 = get_theta(theta03, zern)
    
    theta04 = np.zeros((num_imgs_4, num_c))
    theta04[1:] = coeffs[imgs_idx_4[1:]]
    theta4 = get_theta(theta04, zern)
    
    import sys
    sys.exit()

    phi = coeffs[0]
    c0 = np.zeros((num_c))+1e-10

    c11 = iter_g0(zern, inds, imgs1, theta1, Sk0, c0, ff_1, 10, show)[0]
    c21 = iter_p(zern, inds, imgs1, theta1, Sk0, c0, ff_1, show)[0].copy()
    
    c12, wob1, g_cist = iter_g0(zern, inds, imgs2, theta2, Sk0, c0, ff_1, 10, show)
    c22 = iter_p(zern, inds, imgs2, theta2, Sk0, c0, ff_1, show)[0].copy()
    
    c13, wob1, g_cist = iter_g0(zern, inds, imgs3, theta3, Sk0, c0, ff_1, 10, show)
    c23 = iter_p(zern, inds, imgs3, theta3, Sk0, c0, ff_1, show)[0].copy()

    c14, wob1, g_cist = iter_g0(zern, inds, imgs4, theta4, Sk0, c0, ff_2, 10, show)
    c24 = iter_p(zern, inds, imgs4, theta4, Sk0, c0, ff_2, show)[0].copy()

    
    sri[i] = errA(c11*0, phi, show)
    
    srg1[i] = errA(c11, phi, show)
    srp1[i] = errA(c21, phi, show)
    
    srg2[i] = errA(c12, phi, show)
    srp2[i] = errA(c22, phi, show)
    
    srg3[i] = errA(c13, phi, show)
    srp3[i] = errA(c23, phi, show)

    srg4[i] = errA(c14, phi, show)
    srp4[i] = errA(c24, phi, show)

save_data(f'sweep_exp_{dsize}_{scale}', sri, srg1, srp1, srg2, srp2, srg3, srp3, srg4, srp4)