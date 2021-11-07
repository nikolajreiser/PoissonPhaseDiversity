#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:55:23 2021

@author: nikolaj
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import load_data

noise_imgs, error_decon, error_im_aber, error_im_corr, error_im_perf, noise_vals, errs_p = load_data('sweep_deconv')
num_points = 100

fig, ax = plt.subplots()


# noise_vals *= dsize**2*num_imgs
ax2 = ax.twinx()

ax2.errorbar(noise_vals, errs_p.mean(axis = 1), c = 'black', ls = '--', yerr = errs_p.std(axis = 1)/np.sqrt(num_points), label = 'RWE (right scale)')

ax.errorbar(noise_vals*4/3, error_im_perf.mean(axis = 1), yerr = error_im_perf.std(axis = 1)/np.sqrt(num_points), label = 'Ideal Correction')
ax.errorbar(noise_vals*4/3, error_im_corr.mean(axis = 1), yerr = error_im_corr.std(axis = 1)/np.sqrt(num_points), label = 'Estimated Correction')
ax.errorbar(noise_vals,     error_decon.mean(axis = 1),   yerr = error_decon.std(axis = 1)/np.sqrt(num_points),   label = 'Deconvolved')
ax.errorbar(noise_vals*4/3, error_im_aber.mean(axis = 1), yerr = error_im_aber.std(axis = 1)/np.sqrt(num_points), label = 'Aberrated')



fig.legend(loc = 'upper right', bbox_to_anchor=(.9, .88))
# ax.set_xlabel("Total # of photons for image set")
ax.set_xlabel("# photons/pixel")
ax.set_xscale("log")
ax.set_ylabel(r"DCTS")
ax2.set_ylabel(r"RWE ($\lambda$)")
