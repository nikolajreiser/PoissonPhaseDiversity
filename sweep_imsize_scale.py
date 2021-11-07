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
from src.image_functions import scl_imgs, defocus, save_data, progress, errA
from src.fast_fft import Fast_FFTs
from src.sim_cell import cell_multi
from skimage.transform import downscale_local_mean as dlm

#precompute pupil and zernike polynomials
pixelSize = .096
NA = 1.2
l = .532
RI = 1.33
pupilSize = NA/l
Sk0 = np.pi*(pixelSize*pupilSize)**2
l2p = 2*np.pi/l #length to phase

seed(0)
num_imgs = 3
num_phi = 42
rotang = 0
show = False

num_c = 12

num_theta = num_c
div_mag = 1
div_mag *= l #convert waves to um
tm = np.array([-div_mag, div_mag, 0])

num_photons = 500
dark_noise = 1
read_noise = 2

z_64, R_64, T_64, i_64 = get_zern(64, pupilSize, pixelSize, num_phi)
z_128, R_128, T_128, i_128 = get_zern(128, pupilSize, pixelSize, num_phi)
z_256, R_256, T_256, i_256 = get_zern(256, pupilSize, pixelSize, num_phi)
z_512, R_512, T_512, i_512 = get_zern(512, pupilSize, pixelSize, num_phi)
z_1024, R_1024, T_1024, i_1024 = get_zern(1024, pupilSize, pixelSize, num_phi)

z_32d, R_32d, T_32d, i_32d = get_zern(32, pupilSize, pixelSize*2, num_phi)
z_64d, R_64d, T_64d, i_64d = get_zern(64, pupilSize, pixelSize*2, num_phi)
z_128d, R_128d, T_128d, i_128d = get_zern(128, pupilSize, pixelSize*2, num_phi)
z_256d, R_256d, T_256d, i_256d = get_zern(256, pupilSize, pixelSize*2, num_phi)
z_512d, R_512d, T_512d, i_512d = get_zern(512, pupilSize, pixelSize*2, num_phi)

t_64 = defocus(tm, R_64, i_64, NA, l, RI)
t_128 = defocus(tm, R_128, i_128, NA, l, RI)
t_256 = defocus(tm, R_256, i_256, NA, l, RI)
t_512 = defocus(tm, R_512, i_512, NA, l, RI)
t_1024 = defocus(tm, R_1024, i_1024, NA, l, RI)

t_32d = defocus(tm, R_32d, i_32d, NA, l, RI)
t_64d = defocus(tm, R_64d, i_64d, NA, l, RI)
t_128d = defocus(tm, R_128d, i_128d, NA, l, RI)
t_256d = defocus(tm, R_256d, i_256d, NA, l, RI)
t_512d = defocus(tm, R_512d, i_512d, NA, l, RI)

ff_32 = Fast_FFTs(32, num_imgs, 1)
ff_64 = Fast_FFTs(64, num_imgs, 1)
ff_128 = Fast_FFTs(128, num_imgs, 1)
ff_256 = Fast_FFTs(256, num_imgs, 1)
ff_512 = Fast_FFTs(512, num_imgs, 1)
ff_1024 = Fast_FFTs(1024, num_imgs, 1)

ob_1024 = cell_multi(1024*2, 1600, (30, 60), e = .7, overlap = .05)
ob_512 = dlm(ob_1024, (2, 2))
ob_256 = dlm(ob_512, (2, 2))
ob_128 = dlm(ob_256, (2, 2))
ob_64 =  dlm(ob_128, (2, 2))


abmag0 = 2
abmag1 = abmag0/2

num_points = 10
errs_gn = np.zeros((5, num_points))
errs_gd = np.zeros((5, num_points))
errs_pn = np.zeros((5, num_points))
errs_pd = np.zeros((5, num_points))

time_gn = np.zeros((5, num_points))
time_gd = np.zeros((5, num_points))
time_pn = np.zeros((5, num_points))
time_pd = np.zeros((5, num_points))

c0 = np.zeros((num_c))+1e-10

for i in range(num_points):
    
    progress(str(i))

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
    
    
    imgs_64 = sim_im_2(ob_64, (64, 64), phi, num_imgs, t_64, z_64, R_64, i_64)
    imgs_64 = add_noise(imgs_64, num_photons = num_photons, dark_noise = dark_noise, read_noise = read_noise)[1]
    imgs_64 = scl_imgs(imgs_64)
    
    imgs_128 = sim_im_2(ob_128, (128, 128), phi, num_imgs, t_128, z_128, R_128, i_128)
    imgs_128 = add_noise(imgs_128, num_photons = num_photons, dark_noise = dark_noise, read_noise = read_noise)[1]
    imgs_128 = scl_imgs(imgs_128)
    
    imgs_256 = sim_im_2(ob_256, (256, 256), phi, num_imgs, t_256, z_256, R_256, i_256)
    imgs_256 = add_noise(imgs_256, num_photons = num_photons, dark_noise = dark_noise, read_noise = read_noise)[1]
    imgs_256 = scl_imgs(imgs_256)
    
    imgs_512 = sim_im_2(ob_512, (512, 512), phi, num_imgs, t_512, z_512, R_512, i_512)
    imgs_512 = add_noise(imgs_512, num_photons = num_photons, dark_noise = dark_noise, read_noise = read_noise)[1]
    imgs_512 = scl_imgs(imgs_512)
    
    imgs_1024 = sim_im_2(ob_1024, (1024, 1024), phi, num_imgs, t_1024, z_1024, R_1024, i_1024)
    imgs_1024 = add_noise(imgs_1024, num_photons = num_photons, dark_noise = dark_noise, read_noise = read_noise)[1]
    imgs_1024 = scl_imgs(imgs_1024)
    
    
    imgs_32d = scl_imgs(dlm(imgs_64, (1, 2, 2)))
    imgs_64d = scl_imgs(dlm(imgs_128, (1, 2, 2)))
    imgs_128d = scl_imgs(dlm(imgs_256, (1, 2, 2)))
    imgs_256d = scl_imgs(dlm(imgs_512, (1, 2, 2)))
    imgs_512d = scl_imgs(dlm(imgs_1024, (1, 2, 2)))
        
    
    cg_32d, wob1, [g0, time_gd[0, i]] = iter_g0(z_32d[:num_c], i_32d, imgs_32d, t_32d, Sk0, c0.copy(), ff_32, 100, show)
    cp_32d, p0, num_iter2, [sss, junk, time_pd[0, i]] = iter_p(z_32d[:num_c], i_32d, imgs_32d, t_32d, Sk0, c0.copy(), ff_32, show)
    
    cg_64d, wob1, [g0, time_gd[1, i]] = iter_g0(z_64d[:num_c], i_64d, imgs_64d, t_64d, Sk0, c0.copy(), ff_64, 100, show)
    cp_64d, p0, num_iter2, [sss, junk, time_pd[1, i]] = iter_p(z_64d[:num_c], i_64d, imgs_64d, t_64d, Sk0, c0.copy(), ff_64, show)
    
    cg_128d, wob1, [g0, time_gd[2, i]] = iter_g0(z_128d[:num_c], i_128d, imgs_128d, t_128d, Sk0, c0.copy(), ff_128, 100, show)
    cp_128d, p0, num_iter2, [sss, junk, time_pd[2, i]] = iter_p(z_128d[:num_c], i_128d, imgs_128d, t_128d, Sk0, c0.copy(), ff_128, show)
    
    cg_256d, wob1, [g0, time_gd[3, i]] = iter_g0(z_256d[:num_c], i_256d, imgs_256d, t_256d, Sk0, c0.copy(), ff_256, 100, show)
    cp_256d, p0, num_iter2, [sss, junk, time_pd[3, i]] = iter_p(z_256d[:num_c], i_256d, imgs_256d, t_256d, Sk0, c0.copy(), ff_256, show)
    
    cg_512d, wob1, [g0, time_gd[4, i]] = iter_g0(z_512d[:num_c], i_512d, imgs_512d, t_512d, Sk0, c0.copy(), ff_512, 100, show)
    cp_512d, p0, num_iter2, [sss, junk, time_pd[4, i]] = iter_p(z_512d[:num_c], i_512d, imgs_512d, t_512d, Sk0, c0.copy(), ff_512, show)
    
    cg_64, wob1, [g0, time_gn[0, i]] = iter_g0(z_64[:num_c], i_64, imgs_64, t_64, Sk0, c0.copy(), ff_64, 100, show)
    cp_64, p0, num_iter2, [sss, junk, time_pn[0, i]] = iter_p(z_64[:num_c], i_64, imgs_64, t_64, Sk0, c0.copy(), ff_64, show)
    
    cg_128, wob1, [g0, time_gn[1, i]] = iter_g0(z_128[:num_c], i_128, imgs_128, t_128, Sk0, c0.copy(), ff_128, 100, show)
    cp_128, p0, num_iter2, [sss, junk, time_pn[1, i]] = iter_p(z_128[:num_c], i_128, imgs_128, t_128, Sk0, c0.copy(), ff_128, show)
    
    cg_256, wob1, [g0, time_gn[2, i]] = iter_g0(z_256[:num_c], i_256, imgs_256, t_256, Sk0, c0.copy(), ff_256, 100, show)
    cp_256, p0, num_iter2, [sss, junk, time_pn[2, i]] = iter_p(z_256[:num_c], i_256, imgs_256, t_256, Sk0, c0.copy(), ff_256, show)
    
    cg_512, wob1, [g0, time_gn[3, i]] = iter_g0(z_512[:num_c], i_512, imgs_512, t_512, Sk0, c0.copy(), ff_512, 100, show)
    cp_512, p0, num_iter2, [sss, junk, time_pn[3, i]] = iter_p(z_512[:num_c], i_512, imgs_512, t_512, Sk0, c0.copy(), ff_512, show)
    
    cg_1024, wob1, [g0, time_gn[4, i]] = iter_g0(z_1024[:num_c], i_1024, imgs_1024, t_1024, Sk0, c0.copy(), ff_1024, 100, show)
    cp_1024, p0, num_iter2, [sss, junk, time_pn[4, i]] = iter_p(z_1024[:num_c], i_1024, imgs_1024, t_1024, Sk0, c0.copy(), ff_1024, show)
    
    
    
    
    errs_gd[0, i] = errA(cg_32d, phi0*2*np.pi, show)
    errs_pd[0, i] = errA(cp_32d, phi0*2*np.pi, show)
    
    errs_gd[1, i] = errA(cg_64d, phi0*2*np.pi, show)
    errs_pd[1, i] = errA(cp_64d, phi0*2*np.pi, show)
    
    errs_gd[2, i] = errA(cg_128d, phi0*2*np.pi, show)
    errs_pd[2, i] = errA(cp_128d, phi0*2*np.pi, show)
    
    errs_gd[3, i] = errA(cg_256d, phi0*2*np.pi, show)
    errs_pd[3, i] = errA(cp_256d, phi0*2*np.pi, show)
    
    errs_gd[4, i] = errA(cg_512d, phi0*2*np.pi, show)
    errs_pd[4, i] = errA(cp_512d, phi0*2*np.pi, show)
    
    errs_gn[0, i] = errA(cg_64, phi0*2*np.pi, show)
    errs_pn[0, i] = errA(cp_64, phi0*2*np.pi, show)
    
    errs_gn[1, i] = errA(cg_128, phi0*2*np.pi, show)
    errs_pn[1, i] = errA(cp_128, phi0*2*np.pi, show)
    
    errs_gn[2, i] = errA(cg_256, phi0*2*np.pi, show)
    errs_pn[2, i] = errA(cp_256, phi0*2*np.pi, show)
    
    errs_gn[3, i] = errA(cg_512, phi0*2*np.pi, show)
    errs_pn[3, i] = errA(cp_512, phi0*2*np.pi, show)
    
    errs_gn[4, i] = errA(cg_1024, phi0*2*np.pi, show)
    errs_pn[4, i] = errA(cp_1024, phi0*2*np.pi, show)
    
        
save_data('sweep_imsize_scale', errs_pn, errs_pd, errs_gn, errs_gd, time_pn, time_pd, time_gn, time_gd)