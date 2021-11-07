#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:16:28 2021

@author: nikolaj
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import load_data

errs_pn, errs_pd, errs_gn, errs_gd, time_pn, time_pd, time_gn, time_gd = load_data('sweep_imsize_scale')
# errs_pn, errs_pd, errs_gn, errs_gd, time_pn, time_pd, time_gn, time_gd = load_data('sweep_imsize_crop')


ticks = [r'$64^2$', r'$128^2$', r'$256^2$', r'$512^2$', r'$1024^2$']
pos = np.arange(len(ticks))*5

if False:
    def set_box_color(bp, color, solid = True):
        if solid: 
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
        else: 
            plt.setp(bp['boxes'], color=color, linestyle = 'dashed')
            plt.setp(bp['whiskers'], color=color, linestyle = 'dashed')
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    
    fig, ax = plt.subplots()
    
    bpl = ax.boxplot(errs_pn.T, positions=pos-1.2, sym='', widths=0.6)
    bpr = ax.boxplot(errs_pd.T, positions=pos+0.6, sym='', widths=0.6)
    set_box_color(bpl, 'blue')
    set_box_color(bpr, 'blue', False)
    
    bpl = ax.boxplot(errs_gn.T, positions=pos-0.3, sym='', widths=0.6)
    bpr = ax.boxplot(errs_gd.T, positions=pos+1.5, sym='', widths=0.6)
    set_box_color(bpl, 'orange')
    set_box_color(bpr, 'orange', False)
    
    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c='blue', label='Poisson')
    ax.plot([], c='orange', label='Gaussian')
    ax.plot([], c='blue', linestyle = 'dashed', label='Poisson, Downscaled')
    ax.plot([], c='orange', linestyle = 'dashed', label='Gaussian, Downscaled')
    
    fig.legend(loc = 'upper right', bbox_to_anchor=(.9, .88))
    ax.set_ylabel(r"RWE ($\lambda$)")
    ax.set_ylim(.07, 4)
    ax.set_xlabel("Image Size (pixels)")
    ax.set_yscale('log')
    ax.set_xticks(pos)
    ax.set_xticklabels(ticks)
    
    fig2, ax2 = plt.subplots()
    
    bpl = ax2.boxplot(time_pn.T, positions=pos-1.2, sym='', widths=0.6)
    bpr = ax2.boxplot(time_pd.T, positions=pos+0.6, sym='', widths=0.6)
    set_box_color(bpl, 'blue')
    set_box_color(bpr, 'blue', False)
    
    bpl = ax2.boxplot(time_gn.T, positions=pos-0.3, sym='', widths=0.6)
    bpr = ax2.boxplot(time_gd.T, positions=pos+1.5, sym='', widths=0.6)
    set_box_color(bpl, 'orange')
    set_box_color(bpr, 'orange', False)
    
    # draw temporary red and blue lines and use them to create a legend
    ax2.plot([], c='orange', label='Gaussian')
    ax2.plot([], c='blue', label='Poisson')
    ax2.plot([], c='orange', linestyle = 'dashed', label='Gaussian, Downscaled')
    ax2.plot([], c='blue', linestyle = 'dashed', label='Poisson, Downscaled')
    
    fig2.legend(loc = 'upper left', bbox_to_anchor=(.125, .88))
    ax2.set_ylabel("Runtime (s)")
    ax2.set_xlabel("Image Size (pixels)")
    ax2.set_yscale('log')
    ax2.set_xticks(pos)
    ax2.set_xticklabels(ticks)
    
if True:
    def set_box_color(bp, color, solid = True, lw = 1):
        if solid: 
            plt.setp(bp['cbars'], color=color, linewidth = lw)
        else: 
            plt.setp(bp['cbars'], color=color, linewidth = lw, linestyle = 'dashed')
            
        plt.setp(bp['bodies'], color=color)
        plt.setp(bp['cmins'], color=color, linewidth = lw)
        plt.setp(bp['cmaxes'], color=color, linewidth = lw)
        plt.setp(bp['cmeans'], color=color, linewidth = lw)
        
    fig, ax = plt.subplots()
    
    w = .8
    
    bpl = ax.violinplot(errs_pn.T, showmeans = True, positions=pos-1.2, widths=w)
    bpr = ax.violinplot(errs_pd.T, showmeans = True, positions=pos+0.6, widths=w)
    set_box_color(bpl, 'blue')
    set_box_color(bpr, 'blue', False)
    
    bpl = ax.violinplot(errs_gn.T, showmeans = True, positions=pos-0.3, widths=w)
    bpr = ax.violinplot(errs_gd.T, showmeans = True, positions=pos+1.5, widths=w)
    set_box_color(bpl, 'orange')
    set_box_color(bpr, 'orange', False)
    
    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c='blue', label='Poisson')
    ax.plot([], c='orange', label='Gaussian')
    ax.plot([], c='blue', linestyle = 'dashed', label='Poisson, Downscaled')
    ax.plot([], c='orange', linestyle = 'dashed', label='Gaussian, Downscaled')
    
    fig.legend(loc = 'upper right', bbox_to_anchor=(.9, .88), fontsize = 8)
    ax.set_ylabel(r"RWE ($\lambda$)")
    ax.set_ylim(.07, 5)
    ax.set_xlabel("Image Size (pixels)")
    ax.set_yscale('log')
    ax.set_xticks(pos)
    ax.set_xticklabels(ticks)
    
    fig2, ax2 = plt.subplots()
    
    bpl = ax2.violinplot(time_pn.T, showmeans = True, positions=pos-1.2, widths=w)
    bpr = ax2.violinplot(time_pd.T, showmeans = True, positions=pos+0.6, widths=w)
    set_box_color(bpl, 'blue')
    set_box_color(bpr, 'blue', False)
    
    bpl = ax2.violinplot(time_gn.T, showmeans = True, positions=pos-0.3, widths=w)
    bpr = ax2.violinplot(time_gd.T, showmeans = True, positions=pos+1.5, widths=w)
    set_box_color(bpl, 'orange')
    set_box_color(bpr, 'orange', False)
    
    # draw temporary red and blue lines and use them to create a legend
    ax2.plot([], c='orange', label='Gaussian')
    ax2.plot([], c='blue', label='Poisson')
    ax2.plot([], c='orange', linestyle = 'dashed', label='Gaussian, Downscaled')
    ax2.plot([], c='blue', linestyle = 'dashed', label='Poisson, Downscaled')
    
    fig2.legend(loc = 'upper left', bbox_to_anchor=(.125, .88))
    ax2.set_ylabel("Runtime (s)")
    ax2.set_xlabel("Image Size (pixels)")
    ax2.set_yscale('log')
    ax2.set_xticks(pos)
    ax2.set_xticklabels(ticks)

