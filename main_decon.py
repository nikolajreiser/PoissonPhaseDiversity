#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:42:47 2020

@author: nikolaj
"""


import numpy as np
from numpy.random import rand, seed
from numpy.linalg import norm
from src.iterate_poisson import iter_p
from src.iterate_gaussian import iter_g0
from src.sim_im import sim_im, sim_im_2, add_noise
from src.zern import get_zern, normalize
from src.image_functions import scl, defocus, save_data, progress, errA, ift, dcts, get_H2, imshow
from src.fast_fft import Fast_FFTs
from src.sim_cell import cell_multi
from skimage.transform import downscale_local_mean as dlm
from deconvolution.deconv import rl
from skimage.metrics import structural_similarity as ssim


#precompute pupil and zernike polynomials
pixelSize = .096
NA = 1.2
l = .532
RI = 1.33
pupilSize = NA/l
Sk0 = np.pi*(pixelSize*pupilSize)**2
l2p = 2*np.pi/l #length to phase

dsize = 512
dim0 = (dsize, dsize)
num_imgs = 3
num_phi = 42
rotang = 0

seed(0)

imsize = 256
num_c = 12
num_theta = num_c


zern, R, Theta, inds = get_zern(imsize, pupilSize, pixelSize*2, num_phi)
zern0, R0, Theta, inds0, = get_zern(dsize, pupilSize, pixelSize, num_phi)

dim = (imsize,imsize)
ff = Fast_FFTs(imsize, num_imgs, 1)


div_mag = 1
div_mag *= l #convert waves to um

obname = "cells"
abmag0 = 2
abmag1 = abmag0/2

# ob = cell_multi(dsize*2, 300, (10, 60), e = .7, overlap = .2)
# obname = "ob1"

# ob = cell_multi(dsize*2, 400, (30, 60), e = .7, overlap = .05)
# obname = "ob2"

ob = cell_multi(dsize*2, 1000, (30, 60), e = .1, overlap = .5)
obname = "ob3"

# ob = cell_multi(dsize*2, 100, (10, 15), e = 1, overlap = .5)
# obname = "ob4"




show = False
print(obname)


theta = defocus(np.array([-div_mag, div_mag, 0]), R0, inds0, NA, l, RI)
noise_val = 30


phi0 = (rand(num_c)*2-1)
phi0 /= norm(phi0*np.sqrt(normalize(np.arange(num_c))))
phi0 *= abmag0

phi1 = (rand(num_phi-num_c)*2-1)
if num_c != num_phi:    
    phi1 /= norm(phi1*np.sqrt(normalize(np.arange(num_c, num_phi))))
    phi1 *= abmag1

phi = np.zeros((num_phi))
phi[:num_c] = phi0
phi[num_c:] = phi1

phi *= 2*np.pi #convert waves to radians

phi_corr = phi.copy()
phi_corr[:num_c] = 0

imgs0 = sim_im_2(ob, dim0, phi, num_imgs, theta, zern0, R0, inds0)
img_perfect = sim_im_2(ob, dim0, phi_corr, num_imgs, theta, zern0, R0, inds0)
snr, img_perfect = add_noise(img_perfect, num_photons = noise_val, dark_noise = 1, read_noise = 2)
img_perfect = scl(img_perfect[-1])

snr, imgs0 = add_noise(imgs0, num_photons = noise_val, dark_noise = 1, read_noise = 2)
imgs0 = scl(imgs0)

imgs = dlm(imgs0, (1, 2, 2))

if show: print(f"SNR: {snr:.2f}")

phi = phi0[:num_c]


c0 = np.zeros((num_c))+1e-10
ob_temp = ob[128:-128, 128:-128]


c2 = phi0*2*np.pi-.1*(rand(num_c)*2-1)
phi_corr[:num_c] = phi0*2*np.pi - c2

im_corr = sim_im_2(ob, dim0, phi_corr, num_imgs, theta, zern0, R0, inds0)
snr, im_corr = add_noise(im_corr, num_photons = noise_val, dark_noise = 1, read_noise = 2)
im_corr = scl(im_corr[-1])



H = get_H2(zern0[:num_c], inds0, dim0, theta, c2)
s = np.abs(ift(H))**2
decon = rl(imgs0, s, n_iter = 20, lam = 1e-2, reg_option = 0, b = 0)
decon_single = rl(imgs0[-1], s[-1], n_iter = 20, lam = 1e-2, reg_option = 0, b = 0)




w = 10 #amount to chop off each edge of image

print(f"ssim_decon: {ssim(scl(decon[w:-w, w:-w]), scl(img_perfect[w:-w, w:-w]), data_range = 1)}")
print(f"ssim_decon_single: {ssim(scl(decon_single[w:-w, w:-w]), scl(img_perfect[w:-w, w:-w]), data_range = 1)}")

print(f"ssim_aber: {ssim(scl(imgs0[-1][w:-w, w:-w]), scl(img_perfect[w:-w, w:-w]), data_range = 1)}")
print(f"ssim_corr: {ssim(scl(im_corr[w:-w, w:-w]), scl(img_perfect[w:-w, w:-w]), data_range = 1)}")
        
