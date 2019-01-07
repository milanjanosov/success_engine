import numpy as np  
import random
from scipy import stats
import os
import sys
import math
import scipy




def getDistribution(keys, normalized = True):
    
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    distr = np.bincount(bins) 

    if normalized == 1: distr = distr/float(np.sum(distr)) 

    return np.asarray(uniq_keys.tolist()), np.asarray(distr.tolist())


def getLogBinnedDistribution(x, y, nbins):



    bins   = 10 ** np.linspace(np.log10(min(x)), np.log10(max(x)), nbins)  
    values = [ np.mean([y[j]  for j in range(len(x)) if x[j] >= bins[i] and x[j] < bins[i+1]])  for i in range(nbins-1)]    
    error  = [ np.std( [y[j]  for j in range(len(x)) if x[j] >= bins[i] and x[j] < bins[i+1]])  for i in range(nbins-1)]
    bins   = (bins[1:] + bins[:-1])/2

    return bins, values, error




def fit_field(field):



    print 'Fitting ' + field +  ' ...'

    impacts = [ float(line.strip()) for line in open('DataToPlot_linrescaled_final/1_impact_distribution/' + '1_impact_distribution_raw_'+field+'.dat') if 'nan' not in line]
    impacts = [int(i) for i in impacts if i > 0.0 ]

    x_rand, p_rand = getDistribution(np.asarray(impacts), True)
    x_rand = list(x_rand)
    p_rand = list(p_rand)
    bx, by, byerr = getLogBinnedDistribution(x_rand, p_rand, 10)


    impacts = [i + random.random()/10.0 for i in impacts]

    p0    = stats.lognorm._fitstart(impacts)


    p0    = stats.lognorm.fit(impacts, p0[0], loc  = p0[1],scale = p0[2])
    p1    = stats.lognorm.fit(impacts, p0[0], loc  = p0[1],scale = p0[2])
    param = stats.lognorm.fit(impacts, p1[0], loc  = p1[1],scale = p1[2])




    ppdf_fitted = stats.lognorm.pdf(x_rand, param[0], loc=param[1], scale=param[2])
    sigma = param[0]
    mu =  np.log(param[2])

    sp = sum(ppdf_fitted)
    ppdf_fitted = [p/sp for p in ppdf_fitted]



    #sk_results_norm = stats.ks_2samp(cdf_fitted, np.cumsum(p_rand)) 

    fout =  open('DataToPlot_linrescaled_final/1_impact_distribution/' + '1_impact_distribution_fitted_'+field+'.dat', 'w')

    mmm = min(np.asarray(by) - np.asarray(byerr))
    for i in range(len(ppdf_fitted)):
        if ppdf_fitted[i] > 0.8 * mmm:
            fout.write(str(x_rand[i]) + '\t' + str(ppdf_fitted[i]) + '\n')
    fout.close()



    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(p_rand, ppdf_fitted)
    fstatname = open('DataToPlot_linrescaled_final/1_impact_distribution/R2stat.dat', 'a')
    fstatname.write(field + '\t' + str(r_value**2) + '\n')
    fstatname.close()


    print field, r_value**2




fstatname = open('DataToPlot_linrescaled_final/1_impact_distribution/R2stat.dat', 'w')
fstatname.close()

fields = list(set([o.split('-')[0] for o in os.listdir('Qparamfit_linrescaled_final') if 'art' not in o]))


Pros = []
for field in fields:
    p = Process(target = fit_field, args=(field, ))
    Pros.append(p)
    p.start()
   
for t in Pros:
    t.join()











