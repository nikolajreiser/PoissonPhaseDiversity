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

#DONT FORGET TO RUN find . -size +100M | cat >> .gitignore
#afterwards
print(f"DONT FORGET TO RUN find . -size +100M | cat >> .gitignore")

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


div_mag = 2
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


noise_vals = 10**(np.linspace(1, 5, 10))
num_points = 10
errs_g = np.zeros((len(noise_vals), num_points))
errs_p = np.zeros((len(noise_vals), num_points))
error_decon = np.zeros((len(noise_vals), num_points))
error_decon_single = np.zeros((len(noise_vals), num_points))
error_im_aber = np.zeros((len(noise_vals), num_points))
error_im_corr = np.zeros((len(noise_vals), num_points))
sample_imgs = np.zeros((5, len(noise_vals), dsize, dsize))

for i, noise_val in enumerate(noise_vals):
    for j in range(num_points):
        progress(f"{i+1}/{len(noise_vals)} \t {j+1}/{num_points}")


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
        
        # c1, g_ob, g_cist = iter_g0(zern[:num_c], inds, imgs, theta, Sk0, c0, ff, 20, show)
        # errs_g[i, j] = errA(c1, phi0*2*np.pi, show)
        c2, cost2, num_iter2, sss = iter_p(zern[:num_c], inds, imgs, theta, Sk0, c0.copy(), ff, show)
        errs_p[i, j] = errA(c2, phi0*2*np.pi, show)
        
        
        phi_corr[:num_c] = phi0*2*np.pi-c2
        
        im_corr = sim_im_2(ob, dim0, phi_corr, num_imgs, theta, zern0, R0, inds0)
        snr, im_corr = add_noise(im_corr, num_photons = noise_val, dark_noise = 1, read_noise = 2)
        im_corr = scl(im_corr[-1])
        


        H = get_H2(zern0[:num_c], inds0, dim0, theta, c2)
        s = np.abs(ift(H))**2
        decon = rl(imgs0, s, n_iter = 20, lam = 1e-2, reg_option = 0, b = 0)
        decon_single = rl(imgs0[-1], s[-1], n_iter = 20, lam = 1e-2, reg_option = 0, b = 0)



        # error_decon[i, j] = dcts(decon[w:-w, w:-w])
        # error_im_aber[i, j] = dcts(imgs0[-1][w:-w, w:-w])
        # error_im_corr[i, j] = dcts(im_corr[w:-w, w:-w])
        # error_im_perf[i, j] = dcts(img_perfect[w:-w, w:-w])
        
        w = 10 #amount to chop off each edge of image

        error_decon[i, j] = ssim(scl(decon[w:-w, w:-w]), scl(img_perfect[w:-w, w:-w]), data_range = 1)
        error_decon_single[i, j] = ssim(scl(decon_single[w:-w, w:-w]), scl(img_perfect[w:-w, w:-w]), data_range = 1)

        error_im_aber[i, j] = ssim(scl(imgs0[-1][w:-w, w:-w]), scl(img_perfect[w:-w, w:-w]), data_range = 1)
        error_im_corr[i, j] = ssim(scl(im_corr[w:-w, w:-w]), scl(img_perfect[w:-w, w:-w]), data_range = 1)
        
        
        if j == 0:
            sample_imgs[0,i] = imgs0[-1]
            sample_imgs[1,i] = decon
            sample_imgs[2,i] = decon_single
            sample_imgs[3,i] = im_corr
            sample_imgs[4,i] = img_perfect
        

save_data('sweep_deconv_2', sample_imgs, error_decon, error_decon_single, error_im_aber, error_im_corr, noise_vals, errs_p)