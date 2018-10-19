import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
from scipy.stats import spearmanr, kendalltau, pearsonr
import random

import math
from scipy import stats
from matplotlib import gridspec
import pandas as pd
import seaborn as sns
import scipy   

import warnings
warnings.filterwarnings('ignore')


def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2


def getDistribution(keys, normalized = True):
    
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    distr = np.bincount(bins) 

    if normalized == 1: distr = distr/float(np.sum(distr)) 

    return np.asarray(uniq_keys.tolist()), np.asarray(distr.tolist())


def getPercentileBinnedDistribution(x, y, nbins):

    x, y = zip(*sorted(zip(x, y), key=lambda tup: tup[0]))
    elements_per_bin = int(len(x)/float(nbins))

    xx  = [np.mean(x[i*elements_per_bin:(i+1)*elements_per_bin]) for i in range(nbins)]
    yy  = [np.mean(y[i*elements_per_bin:(i+1)*elements_per_bin]) for i in range(nbins)]
    std = [np.std(y[i*elements_per_bin:(i+1)*elements_per_bin])  for i in range(nbins)]

    return xx, yy, std


def getLogBinnedDistribution(x, y, nbins):

    bins   = 10 ** np.linspace(np.log10(min(x)), np.log10(max(x)), nbins)  
    values = [ np.mean([y[j]  for j in range(len(x)) if x[j] >= bins[i] and x[j] < bins[i+1]])  for i in range(nbins-1)]    
    error  = [ np.std( [y[j]  for j in range(len(x)) if x[j] >= bins[i] and x[j] < bins[i+1]])  for i in range(nbins-1)]
    bins   = (bins[1:] + bins[:-1])/2

    return bins, values, error


def format_axis(ax):
    
    for pos in ['right', 'top', 'left']:
        ax.spines[pos].set_edgecolor('white')    

    ax.tick_params(axis='x', length=6, width=2, colors='black')
    ax.tick_params(axis='y', length=6, width=2, colors='black') 



field_title = { 'director':      'Movie directors',
                'art-director':  'Movie art directors',
                'producer':      'Movie producers' ,
                'composer':      'Soundtrack composers',
                'writer':        'Plot writers',
                'authors':       'Book authors',
                'electro':       'Electronic music artists', 
                'rock':          'Rock musicians',
                'pop':           'Pop musicians',
                'jazz':          'Jazz musicians',
                'folk':          'Folk musicians',
                'funk':          'Funk musicians',
                'hiphop':        'Hip-hop artists',
                'classical':     'Classical musicians',
               } 




def get_luck_curves_data(fields, title, ax, color, lim, mmin, labelfont = 17, titlefont = 20, legendfont = 12):
    
  
    ax.set_xscale('log')
    ax.set_title(title, fontsize = 15)
    ax.set_xlim(lim)    
    ax.set_ylim([-0.01, 1.05])        

    
    for field in fields:
        
        ps = []
                  
        field = field + '_0'
        ps  +=  [float(line.strip()) for line in 
                  open('pQData/p_distribution_' + field + '.dat')]

    
        X, Y = getDistribution(ps)

        X = [float(x) for x in X]
        Y = [float(y) for y in Y]
        Y = [1 - yy for yy in np.cumsum(Y)]
        mm = max(Y)
        Y = [yy/mm for yy in Y]

        ax.plot(X, Y, linewidth = 1.5, alpha = 0.9, color = color)#, label = 'Aggregated data')  
        
     
    ax.set_ylabel('P(> $p_{i,\\alpha}$)', fontsize = labelfont)
    ax.set_xlabel('$p_{i,\\alpha}$', fontsize = labelfont)    
    format_axis(ax)
    ax.legend(loc = 'lower left', fontsize = legendfont)



fig, ax    = plt.subplots(1,1, figsize = (12,8))

sci_fields = ['mathematics', 'psychology', 'physics', 'health_science', 'zoology', 'agronomy', 'environmental_science', 
          'engineering', 'theoretical_computer_science', 'applied_physics', 'space_science_or_astronomy', 'chemistry', 
          'political_science', 'biology', 'geology']

     
music  = [a + '-80' for a in ['electro', 'pop', 'rock', 'folk', 'funk', 'jazz', 'hiphop', 'classical']]    
movies = ['director-10', 'producer-10','art-director-20', 'composer-10', 'writer-10']    
    
    


get_luck_curves_data(movies, 'All',  ax, 'steelblue', [0.02,   150], 50, labelfont = 13, titlefont = 14, legendfont = 8)
get_luck_curves_data(music , 'All',  ax, 'darkred', [0.02,   150], 50, labelfont = 13, titlefont = 14, legendfont = 8)
get_luck_curves_data(['authors-50'], 'All',  ax, 'darkgreen', [0.02,   150], 50, labelfont = 13, titlefont = 14, legendfont = 8)

get_luck_curves_data(sci_fields , 'All',  ax, 'darkorange', [0.02,   150], 50, labelfont = 13, titlefont = 14, legendfont = 8)

   
plt.tight_layout()

plt.savefig('luckfull_2.png')
