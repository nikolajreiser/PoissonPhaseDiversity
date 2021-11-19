#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:04:57 2021

@author: nikolaj
"""
import matplotlib.pyplot as plt
import numpy as np
from functions import load_data


abmags, errs_g, errs_p = load_data('sweep_magnitude_single_ob4')
num_points = 10
yticks = np.arange(0, 5, .5)
fig, ax = plt.subplots()

ax.errorbar(abmags, errs_g.mean(axis = 1), c = 'orange', yerr = errs_g.std(axis = 1)/np.sqrt(num_points), label = 'Gaussian')
ax.errorbar(abmags, errs_p.mean(axis = 1), c = 'blue', yerr = errs_p.std(axis = 1)/np.sqrt(num_points), label = 'Poisson')


fig.legend(loc = 'upper left', bbox_to_anchor=(.125, .88))
ax.set_xlabel(r"Initial Wavefront RMS ($\lambda$)")
ax.set_ylabel(r"RWE ($\lambda$)")
ax.set_yticks(yticks)

