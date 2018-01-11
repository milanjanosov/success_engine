import numpy as np
import powerlaw
import random
import os
from matplotlib.colors import LogNorm
from scipy import stats
from CareerTrajectory.careerTrajectory import getDistribution
from binningFunctions import getDistribution
import os


def write_row(filename, data):

    f = open(filename, 'w')
    [f.write(str(dat)+'\n') for dat in data ]
    f.close()    


def fitPowerLaw(filename, ax, label, cutoff = -1, writeout = True, numbins = 15, noise = 0):


    rand = np.asarray([float(line.strip()) + noise for line in open(filename) if float(line.strip()) > cutoff]) 
    x_rand, p_rand = getDistribution(rand)                  
    ax.set_title(label, fontsize = 18)               
    
    
    # histogram
    counts, bins, bars = ax.hist(rand, normed = True, bins = 10 ** np.linspace(np.log10(min(x_rand)), np.log10(max(x_rand)), 1000), log=True,alpha=0.0, cumulative=1)
    ax.plot((bins[1:] + bins[:-1])/2, counts, 's-', color = 'royalblue', alpha = 0.7, markersize = 0, linewidth = 5)
    ax.set_ylim([ min(counts), 1.05*max(counts)])
    ax.set_xlim([ min(x_rand),  max(bins)])

    
    # fit and plot the powerlaw   
    print 'Fit and plot the powerlaw...'
    results  = powerlaw.Fit(rand, xmin = min(x_rand), fit_method = 'KS')
    alpha    = results.power_law.alpha
    xmin     = results.power_law.xmin 
    D        = results.power_law.KS()  
    parassms = results.power_law.plot_cdf(color='r',ax=ax,linestyle='-',linewidth=3,label='$\\alpha$= '+str(round(alpha,2))+', $x_{min}$='+str(round(xmin,2))+'\n$D$='+str(round(D, 2)))     
  
   
    # fit and plot the powerlaw   
    print 'Fit and plot the lognormal...' + label
    p0 = stats.lognorm._fitstart(rand)
    p1 = stats.lognorm.fit(rand, p0[0], loc  = p0[1],scale = p0[2])
    param = stats.lognorm.fit(rand, p1[0], loc  = p1[1],scale = p1[2])

    pdf_fitted = stats.lognorm.cdf(x_rand, param[0], loc=param[1], scale=param[2])
    mu =  np.log(param[2])
    sigma = param[0]
 
    sk_results_norm = stats.ks_2samp(pdf_fitted, np.cumsum(p_rand))   # stats.ks_2samp(np.cumsum(p_rand), np.cumsu 
    ax.plot(x_rand,pdf_fitted,'k-', linewidth = 4, label = 'Lognormal fit, $\\mu$=' + str(round(mu,2)) + '\n$\\sigma$=' + str(round(sigma, 2)) + ', $D$='+str(round(sk_results_norm[0], 2)))
   

    ax.set_xlabel(label, fontsize = 20)
    ax.set_ylabel('CDF of ' + label, fontsize = 20)

 
    if writeout:
        out_folder = 'ResultData/1_impact_distributions/'
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
     
     
        xfit = parassms.lines[1].get_xdata()
        yfit = parassms.lines[1].get_ydata()  
        
        write_row(out_folder + label + '_powerlaw_hist_' + '.dat', rand)
        write_row(out_folder + label + '_powerlaw_fit_'  + '.dat', [str(xfit[i]) + '\t' + str(yfit[i]) for i in range(len(xfit))] )   

    

    return sk_results_norm[0], D

  

def fitSkewedNormal(filename, ax, label, alpha_hist  = 0.2, color_line = 'r'):
   

    rand = np.asarray([float(line.strip()) for line in open(filename)])

    print 'Fitting normal...'
    param = stats.skewnorm.fit(rand)
    x_rand, p_rand = getDistribution(rand)
    pdf_fitted = stats.skewnorm.pdf(x_rand,  param[0], loc=param[1], scale=param[2])
            
    mean = stats.skewnorm.mean( param[0], loc=param[1], scale=param[2])
    maxx = str(x_rand[pdf_fitted.tolist().index(max(pdf_fitted))])       
    counts, bins, bars = ax.hist(rand, bins = np.linspace(min(x_rand), max(x_rand), 25), normed = True, alpha = alpha_hist)

    sk_results = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.skewnorm.cdf(x_rand, param[0], loc=param[1], scale=param[2]))   
    ax.plot(x_rand,pdf_fitted,'-', color = color_line, linewidth = 3, label = '$\\mu$=' + str(round(mean, 2)) + ', $\\mu^{*}$=' +maxx+'\n$D$='+str(round(sk_results[0], 2))+ ', $p$='+str(round(sk_results[1],2)))
    ax.set_title(label, fontsize = 18)  


    ax.set_yticks(np.linspace(0, max(counts), 5))
    ax.set_yticklabels([str(int(100*y)) + '%' for y in np.linspace(0, 1.05*max(counts)/(sum(counts)), 5)])
    

   
