#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 10:06:15 2021

@author: nikolaj
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import load_data

phase_mags, errs_g, errs_p = load_data('sweep_phase_noise')
num_points = 100

fig, ax = plt.subplots()

ax.errorbar(phase_mags, errs_g.mean(axis = 1), c = 'orange', yerr = errs_g.std(axis = 1)/np.sqrt(num_points), label = 'Gaussian')
ax.errorbar(phase_mags, errs_p.mean(axis = 1), c = 'blue', yerr = errs_p.std(axis = 1)/np.sqrt(num_points), label = 'Poisson')


fig.legend(loc = 'upper left', bbox_to_anchor=(.125, .88))
ax.set_xlabel(r"Phase Noise RMSE ($\lambda$)")
ax.set_ylabel(r"RWE ($\lambda$)")
