#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:00:39 2021

@author: nikolaj
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import load_data

noise_vals, errs_g, errs_p, read_noises = load_data('sweep_noise_ob1')
num_points = 100


fig, ax = plt.subplots()

labels = ["Low Additive Noise", "High Additive Noise"]
lss = ["solid", "dashed", "dotted", "dashdot"]
for i in range(len(read_noises)):
    ax.errorbar(noise_vals, errs_g[i].mean(axis = 1), c = 'orange', ls = lss[i], yerr = errs_g[i].std(axis = 1)/np.sqrt(num_points), label = f'Gaussian, {labels[i]}')
    ax.errorbar(noise_vals, errs_p[i].mean(axis = 1), c = 'blue', ls = lss[i], yerr = errs_p[i].std(axis = 1)/np.sqrt(num_points), label = f'Poisson, {labels[i]}')


fig.legend(loc = 'upper right', bbox_to_anchor=(.9, .88))
ax.set_xlabel("# photons/pixel")
ax.set_xscale("log")
ax.set_ylabel(r"RWE ($\lambda$)")
