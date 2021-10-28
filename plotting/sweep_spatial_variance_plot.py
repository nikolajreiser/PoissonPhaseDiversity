#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:59:33 2021

@author: nikolaj
"""

import numpy as np
import matplotlib.pyplot as plt

#data order: 
# No sv: gaussian, poisson
# Low freq mag 1: gaussian, poisson
# High freq mag 1: gaussian, poisson
# Low freq mag 4: gaussian, poisson
# High freq mag 4: gaussian, poisson


vals = np.array([[0.311648581236687, 0.2599147565436785],
                 [0.33788549192276873, 0.274758871310687],
                 [0.4228253806464511, 0.4119078525585895],
                 [0.7541614402263297, 0.610175792384856],
                 [2.109952285307927, 0.8221803644614002]])

ticks = ['No SV', 'LF\nLM','HF\nLM','LF\nHM','HF\nHM',]
pos = np.arange(5)
w = .4
d = .01
fig, ax = plt.subplots()

ax.bar(pos, vals[:, 0], width = w, color = 'orange', label = "Gaussian")
ax.bar(pos+w+d, vals[:, 1], width = w, color = 'blue', label = "Poisson")

ax.set_xticks(pos+w/2+d/2)
ax.set_xticklabels(ticks)
fig.legend(loc = 'upper left', bbox_to_anchor=(.125, .88))

ax.set_ylabel(r"RWE ($\lambda$)")
