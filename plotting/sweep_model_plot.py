#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:31:37 2021

@author: nikolaj
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import load_data

eg, ep = load_data('model_test')

ticks = ['LO\nBC','HO\nBC','LO\nRC','HO\nRC',]
order = [0,2,1,3]

pos = np.arange(4)
w = .4
d = .01
fig, ax = plt.subplots()

ax.bar(pos, eg[order], width = w, color = 'orange', label = "Gaussian")
ax.bar(pos+w+d, ep[order], width = w, color = 'blue', label = "Poisson")

ax.set_xticks(pos+w/2+d/2)
ax.set_xticklabels([ticks[o] for o in order])
fig.legend(loc = 'upper left', bbox_to_anchor=(.125, .88))

ax.set_ylabel(r"RWE ($\lambda$)")
