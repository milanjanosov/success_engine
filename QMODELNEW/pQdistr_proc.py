import os
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
from scipy import stats
import pandas as pd
import seaborn as sn
import pandas as pd
from multiprocessing import Process




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




def do_p_Q_plots(field, bx, color, tipus, nbins):
    
    Qdata   = 'DataToPlot/3_pQ_distributions/' + tipus + '_distribution_data_' + field + '_0.dat' 
        
    Qx,  Qy  = zip(*[ [float(fff) for fff in line.strip().split('\t')] for line in open(Qdata)   if 'nan' not in line and float(line.strip().split('\t')[0]) !=  0.0])


    
    bx.plot(Qx, Qy, 'o', color = 'lightgrey', alpha = 0.3, linewidth = 0, markersize = 4)
    bx.set_xscale('log')
    bx.set_yscale('log')
    bx.set_title(field + ' - P(' + tipus + ')', fontsize = 15)  
    
#    bx.plot(bQx, bQy, linewidth = 3, color = 'r')

    bx.set_ylabel('P(' + tipus + ')', fontsize = 15)
    bx.set_xlabel(tipus, fontsize = 15)

    
    if tipus == 'Q':
        Qs    = [float(line.strip().split('\t')[1]) for line in open('pQData/' + tipus + '_distribution_' + field + '_0.dat')]
    else:
        Qs = [float(line.strip().split('\t')[0]) for line in open('pQData/' + tipus + '_distribution_' + field + '_0.dat')]

    if tipus == 'Q' : 
        counts, bins, bars = bx.hist(Qs, normed = True, bins = 10 ** np.linspace(np.log10(min(Qx)), np.log10(max(Qx)), nbins), alpha=0.0, cumulative=0)
        bins = (bins[1:] + bins[:-1])/2    

        Qerror_values = {}
        for ind, qqss in enumerate(Qx):
            yyss = Qy[ind]
            for i in range(1, nbins-1):
                if qqss > bins[i-1] and qqss <= bins[i]:
                    if i not in Qerror_values:
                        Qerror_values[i] = [yyss]
                    else:
                        Qerror_values[i].append(yyss)
            if qqss > bins[-2]: 
                if (nbins - 1) not in Qerror_values:
                    Qerror_values[nbins - 1] = [yyss]
                else:
                    Qerror_values[nbins - 1 ].append(yyss)


        Qerror_bins = [np.std(vals) for i,vals in Qerror_values.items()]

        ppbins, ppcounts, ppQerror_bins = zip(*[(bins[i], counts[i], Qerror_bins[i]) for i in range(len(Qerror_bins)) if counts[i] > 0.1*min(Qy)     ])
    
    else:
 
        ppbins, ppcounts, ppQerror_bins = getLogBinnedDistribution(Qx, Qy, nbins)

    bx.errorbar(ppbins, ppcounts, yerr = ppQerror_bins, linewidth = 2, color = color)
    bx.fill_between(ppbins, np.asarray(ppcounts)- np.asarray(ppQerror_bins), np.asarray(ppcounts) + np.asarray(ppQerror_bins), color = color, alpha = 0.3)
    
   
    
    p0    = stats.lognorm._fitstart(Qs)
    p1    = stats.lognorm.fit(Qs, p0[0], loc  = p0[1],scale = p0[2])
    param = stats.lognorm.fit(Qs, p1[0], loc  = p1[1],scale = p1[2])
    
    
    ppdf_fitted = stats.lognorm.pdf(Qx, param[0], loc=param[1], scale=param[2])
    
    
    Qxpp, ppdf_fittedpp = zip(*[(Qx[i], ppdf_fitted[i]) for i in range(len(ppdf_fitted)) if ppdf_fitted[i] > 0.5 * min(Qy)])
    
    
    bx.plot(Qxpp, ppdf_fittedpp, color = 'k', linewidth = 2)
    
    
    bx.set_ylim([0.25*min(Qy), 2*max(Qy)])
    bx.set_xlim([0.5*min(Qx), 0.8*max(Qx)])
    
    
    bx.set_ylabel('P(Q)', fontsize = 15)
    bx.set_xlabel('Q', fontsize = 15)
    
    for pos in ['right', 'top', 'left']:
        bx.spines[pos].set_edgecolor('white')    

    bx.tick_params(axis='x', length=6, width=2, colors='black')
    bx.tick_params(axis='y', length=6, width=2, colors='black')       
    bx.legend(loc = 'lower left', fontsize = 10)
  
    bx.set_title(field.split('-')[0], fontsize = 17)

    for pos in ['right', 'top', 'left']:
        bx.spines[pos].set_edgecolor('white')    

    bx.tick_params(axis='x', length=6, width=2, colors='black')
    bx.tick_params(axis='y', length=6, width=2, colors='black')       
    bx.legend(loc = 'lower left', fontsize = 10)

    


    folderout = 'DataToPlot/3_pQ_distributions_processed'
    if not os.path.exists(folderout):
        os.makedirs(folderout)



    fout = open(folderout + '/' + field+ '_' + tipus + 'data.dat', 'w')
    for i in range(len(Qx)):
        fout.write(str(Qx[i]) + '\t' + str(Qy[i]) + '\n')

 
    fout = open(folderout + '/' + field+ '_' + tipus + 'fit.dat', 'w')
    for i in range(len(Qxpp)):
        fout.write(str(Qxpp[i]) + '\t' + str(ppdf_fittedpp[i]) + '\n')

    fout = open(folderout + '/' + field + '_' + tipus + 'binned.dat', 'w')
    for i in range(len(ppbins)):
        fout.write(str(ppbins[i]) + '\t' + str(ppcounts[i]) + '\t' + str(ppQerror_bins[i]) + '\n')


    


fields = {  'director'     : '10', 
            'art-director' : '20', 
            'producer'     : '10', 
            'writer'       : '10', 
            'composer'     : '10', 
            'electro'      : '80', 
            'rock'        : '80', 
            'pop'          : '80', 
            'funk'         : '80', 
            'folk'         : '80', 
            'hiphop'       : '80', 
            'jazz'         : '80', 
            'classical'    : '80', 
            'authors'      : '50' }






 




Pros = []   

for fn, lim in fields.items():
  
    f, ax = plt.subplots(2,3, figsize = (15,8,))


    for tipus in ['Q', 'p']:


        print fn, '\t', lim, '\t', tipus
        #(fn + '-' + lim, ax[0,0], 'steelblue', 'Q', 10) 
        p = Process(target = do_p_Q_plots, args=(fn + '-' + lim, ax[0,0], 'steelblue', tipus, 10, ))
        Pros.append(p)
        p.start()


        #print fn, '\t', lim, '\t', 'p'
        #do_p_Q_plots(fn + '-' + lim, ax[1,0], 'steelblue', 'p', 10)


    plt.tight_layout()
    plt.close()


for t in Pros:
    t.join()




