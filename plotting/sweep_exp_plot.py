#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 20:04:38 2021

@author: nikolaj
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import load_data, bin_data as bd, moving_average as ma

# sri, srg1, srp1, srg2, srp2, srg3, srp3, srg4, srp4 = load_data('sweep_exp_256_1')
# sri, srg1d, srp1d, srg2d, srp2d, srg3d, srp3d, srg4d, srp4d = load_data('sweep_exp_256_2')

# sri, srg1d, srp1d, srg2d, srp2d, srg3d, srp3d, srg4d, srp4d = load_data('sweep_exp_256_1')
# sri, srg1d, srp1d, srg2d, srp2d, srg3d, srp3d, srg4d, srp4d = load_data('sweep_exp_256_2')

sri, srg1, srp1, srg2, srp2, srg3, srp3, srg4, srp4 = load_data('sweep_exp_512_1')
sri, srg1d, srp1d, srg2d, srp2d, srg3d, srp3d, srg4d, srp4d = load_data('sweep_exp_512_2')


xs = np.argsort(sri)

#scatter/line plot
if False:
    fig, ax = plt.subplots()
    p1 = srp3
    g1 = srg3
    w = 10
    
    ax.scatter(sri[xs], g1[xs], marker = 'x', c = 'orange', alpha = .5, linewidth = 1, label = "Gaussian")
    ax.scatter(sri[xs], p1[xs], marker = '+', c = 'blue', alpha = .5, linewidth = 1, label = "Poisson")
    ax.plot(sri[xs], ma(g1[xs], w), c = 'orange', linewidth = 1, label = "Moving Average Gaussian")
    ax.plot(sri[xs], ma(p1[xs], w), c = 'blue', linewidth = 1, label = "Moving Average Poisson")
    ax.plot(sri[xs], sri[xs], c = 'black', ls = '--', linewidth = 1, label = "Initial Aberration")    
    
    ax.set_ylabel(r"RWE ($\lambda$)")
    ax.set_xlabel(r"Initial Wavefront RMS ($\lambda$)")
    
    handles, labels = ax.get_legend_handles_labels()
    order = (3, 4, 0, 1, 2)
    handles = [handles[o] for o in order]
    labels = [labels[o] for o in order]
    fig.legend(handles, labels, loc = 'upper left', bbox_to_anchor=(.12, .88))
    fig.show()

#boxplot
if False:
    bins = np.arange(7)/2
    inds = np.digitize(sri, bins)
        
    p1 = bd(inds, bins, srp1)
    p2 = bd(inds, bins, srp2)
    g1 = bd(inds, bins, srg1)
    g2 = bd(inds, bins, srg2)
    p3 = bd(inds, bins, srp3)
    p4 = bd(inds, bins, srp4)
    g3 = bd(inds, bins, srg3)
    g4 = bd(inds, bins, srg4)
    
    p1d = bd(inds, bins, srp1d)
    p2d = bd(inds, bins, srp2d)
    g1d = bd(inds, bins, srg1d)
    g2d = bd(inds, bins, srg2d)
    p3d = bd(inds, bins, srp3d)
    p4d = bd(inds, bins, srp4d)
    g3d = bd(inds, bins, srg3d)
    g4d = bd(inds, bins, srg4d)

    ticks = [f"{bins[i]}-{bins[i+1]}\nn={len(p1[i])}" for i in range(len(bins)-1)]
    pos = np.arange(len(ticks))*5
        
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
    
    bpl = ax.boxplot(p4, positions=pos-1.2, sym='', widths=0.6)
    bpr = ax.boxplot(p4d, positions=pos+0.6, sym='', widths=0.6)
    set_box_color(bpl, 'blue')
    set_box_color(bpr, 'blue', False)
    
    bpl = ax.boxplot(g4, positions=pos-0.3, sym='', widths=0.6)
    bpr = ax.boxplot(g4d, positions=pos+1.5, sym='', widths=0.6)
    set_box_color(bpl, 'orange')
    set_box_color(bpr, 'orange', False)
    
    
    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c='blue', label='Poisson')
    ax.plot([], c='orange', label='Gaussian')
    ax.plot([], c='blue', linestyle = 'dashed', label='Poisson, Downscaled')
    ax.plot([], c='orange', linestyle = 'dashed', label='Gaussian, Downscaled')
    
    fig.legend(loc = 'upper left', bbox_to_anchor=(.12, .88))
    ax.set_ylabel(r"RWE ($\lambda$)")
    # ax.set_ylim(.07, 4)
    ax.set_xlabel(r"Initial Wavefront RMS Range ($\lambda$)"+"\nNumber of Samples for Range")
    # ax.set_yscale('log')
    ax.set_xticks(pos)
    ax.set_xticklabels(ticks)
    
#another box plot
if False:
    bins = np.arange(7)/2
    inds = np.digitize(sri, bins)
        
    p1 = bd(inds, bins, srp1)
    p2 = bd(inds, bins, srp2)
    g1 = bd(inds, bins, srg1)
    g2 = bd(inds, bins, srg2)
    p3 = bd(inds, bins, srp3)
    p4 = bd(inds, bins, srp4)
    g3 = bd(inds, bins, srg3)
    g4 = bd(inds, bins, srg4)
    
    p1d = bd(inds, bins, srp1d)
    p2d = bd(inds, bins, srp2d)
    g1d = bd(inds, bins, srg1d)
    g2d = bd(inds, bins, srg2d)
    p3d = bd(inds, bins, srp3d)
    p4d = bd(inds, bins, srp4d)
    g3d = bd(inds, bins, srg3d)
    g4d = bd(inds, bins, srg4d)

    ticks = [f"{bins[i]}-{bins[i+1]}\nn={len(p1[i])}" for i in range(len(bins)-1)]
    pos = np.arange(len(ticks))*5
        
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
    
    bpl = ax.boxplot(p3, positions=pos-.8, sym='', widths=1.2)
    set_box_color(bpl, 'blue')
    
    bpr = ax.boxplot(g3, positions=pos+.8, sym='', widths=1.2)
    set_box_color(bpr, 'orange')
    
    
    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c='blue', label='Poisson')
    ax.plot([], c='orange', label='Gaussian')
    
    fig.legend(loc = 'upper left', bbox_to_anchor=(.12, .88))
    ax.set_ylabel(r"RWE ($\lambda$)")
    # ax.set_ylim(.07, 4)
    ax.set_xlabel(r"Initial Wavefront RMS Range ($\lambda$)"+"\nNumber of Samples for Range")
    # ax.set_yscale('log')
    ax.set_xticks(pos)
    ax.set_xticklabels(ticks)

#violin plot
if True:
    bins = np.arange(7)/2
    inds = np.digitize(sri, bins)
    w = .8
        
    p1 = bd(inds, bins, srp1)
    p2 = bd(inds, bins, srp2)
    g1 = bd(inds, bins, srg1)
    g2 = bd(inds, bins, srg2)
    p3 = bd(inds, bins, srp3)
    p4 = bd(inds, bins, srp4)
    g3 = bd(inds, bins, srg3)
    g4 = bd(inds, bins, srg4)
    
    p1d = bd(inds, bins, srp1d)
    p2d = bd(inds, bins, srp2d)
    g1d = bd(inds, bins, srg1d)
    g2d = bd(inds, bins, srg2d)
    p3d = bd(inds, bins, srp3d)
    p4d = bd(inds, bins, srp4d)
    g3d = bd(inds, bins, srg3d)
    g4d = bd(inds, bins, srg4d)

    ticks = [f"{bins[i]}-{bins[i+1]}\nn={len(p1[i])}" for i in range(len(bins)-1)]
    pos = np.arange(len(ticks))*5
        
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
    
    bpl = ax.violinplot(p4,showmeans = True, positions=pos-1.2, widths=w)
    bpr = ax.violinplot(p3, showmeans = True, positions=pos+.6, widths=w)
    set_box_color(bpl, 'blue')
    set_box_color(bpr, 'blue', False)
    
    bpl = ax.violinplot(g4, showmeans = True, positions=pos-.3, widths=w)
    bpr = ax.violinplot(g3, showmeans = True, positions=pos+1.5, widths=w)
    set_box_color(bpl, 'orange')
    set_box_color(bpr, 'orange', False)
    
    
    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c='blue', label='Poisson, 5 Im')
    ax.plot([], c='orange', label='Gaussian, 5 Im')
    ax.plot([], c='blue', linestyle = 'dashed', label='Poisson, 3 Im')
    ax.plot([], c='orange', linestyle = 'dashed', label='Gaussian, 3 Im')
    
    fig.legend(loc = 'upper left', bbox_to_anchor=(.12, .88))
    ax.set_ylabel(r"RWE ($\lambda$)")
    # ax.set_ylim(.07, 4)
    ax.set_xlabel(r"Initial Wavefront RMS Range ($\lambda$)"+"\nNumber of Samples for Range")
    # ax.set_yscale('log')
    ax.set_xticks(pos)
    ax.set_xticklabels(ticks)

