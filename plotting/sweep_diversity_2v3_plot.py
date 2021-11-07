#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:23:23 2021

@author: nikolaj
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import load_data

fig, ax = plt.subplots()
num_points = 100
abmag0 = 2
l = .532

div_mags, errs_g, errs_p, errs_g_2, errs_p_2 = load_data('sweep_diversity_2v3_ob4_2')

div_mags /= l #convert from um back to waves

ax.errorbar(div_mags, errs_g.mean(axis = 1), c = 'orange', ls = '-', yerr = errs_g.std(axis = 1)/np.sqrt(num_points), label = 'Gaussian, 3 Im')
ax.errorbar(div_mags, errs_p.mean(axis = 1), c = 'blue', ls = '-', yerr = errs_p.std(axis = 1)/np.sqrt(num_points), label = 'Poisson, 3 Im')

# ax.errorbar(div_mags, errs_g_2.mean(axis = 1), c = 'orange', ls = '--', yerr = errs_g_2.std(axis = 1)/np.sqrt(num_points), label = 'Gaussian, 2 Im')
# ax.errorbar(div_mags, errs_p_2.mean(axis = 1), c = 'blue', ls = '--', yerr = errs_p_2.std(axis = 1)/np.sqrt(num_points), label = 'Poisson, 2 Im')

fig.legend(loc = 'upper right', bbox_to_anchor=(.9, .88))
ax.set_xlabel(r"Diversity Phase Magnitude ($\lambda$)")
ax.set_ylabel(r"RMS Wavefront Error ($\lambda$)")
ax.set_ylim(0, abmag0)
