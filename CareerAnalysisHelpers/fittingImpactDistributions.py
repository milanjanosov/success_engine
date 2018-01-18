import numpy as np
import powerlaw
import random
import os
from matplotlib.colors import LogNorm
from scipy import stats
from CareerTrajectory.careerTrajectory import getDistribution
from binningFunctions import getDistribution
import os
import sys
import math


def write_row(filename, data):

    f = open(filename, 'w')
    [f.write(str(dat)+'\n') for dat in data ]
    f.close()    


def fitPowerLaw(filename, ax, label = '', out_folder = '', name = '', cutoff = -sys.maxint, writeout = True, noise = False, distancefile = ''):

    rand = []


    if 'log_p' in name:
        rand = np.asarray([math.exp(float(line.strip())) + noise for line in open(filename) if  float(line.strip()) > cutoff]) 
    elif 'log_Q' in name:
        rand = np.asarray([math.exp(float(line.strip().split('\t')[1]))  for line in open(filename) if  len(line.strip().split('\t')) > 1 and float(line.strip().split('\t')[1]) > cutoff]) 
    elif noise:
        rand = np.asarray([float(line.strip()) + random.random() for line in open(filename) if float(line.strip()) > cutoff]) 
    else:     
        rand = np.asarray([float(line.strip()) + noise for line in open(filename) if float(line.strip()) > cutoff]) 
    

    x_rand, p_rand = getDistribution(rand)                  
    ax.set_title(label, fontsize = 18)               
    
   
    # histogram
    counts, bins, bars = ax.hist(rand, normed = True, bins = 10 ** np.linspace(np.log10(min(x_rand)), np.log10(max(x_rand)), 1000), log=True,alpha=0.0, cumulative=1)
    ax.plot((bins[1:] + bins[:-1])/2, counts, 's-', color = 'royalblue', alpha = 0.7, markersize = 0, linewidth = 5)
    bins = (bins[1:] + bins[:-1])/2    
    ax.set_ylim([ min(counts), 1.05*max(counts)])
    ax.set_xlim([ min(x_rand),  max(bins)])

    
    # fit and plot the powerlaw   
    print 'Fit and plot the powerlaw...'
    results  = powerlaw.Fit(rand, xmin = min(x_rand), fit_method = 'KS')
    alpha    = results.power_law.alpha
    D        = results.power_law.KS()  
    parassms = results.power_law.plot_cdf(color='r',ax=ax,linestyle='-',linewidth=3,label='$\\alpha$= '+str(round(alpha,2))+', $D$='+str(round(D, 2)))     
  
   
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
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
     
     
        xfit = parassms.lines[1].get_xdata()
        
        yfit = parassms.lines[1].get_ydata()  
        write_row(out_folder + '/' + label + '_' + name + '_powerlaw_hist_' + '.dat', [str(bins[i])   + '\t' + str(counts[i]) for i in range(len(counts))] )
        write_row(out_folder + '/' + label + '_' + name + '_powerlaw_fit_'  + '.dat', [str(xfit[i])   + '\t' + str(yfit[i])       for i in range(len(xfit))] )   
        write_row(out_folder + '/' + label + '_' + name + '_lognormal_'     + '.dat', [str(x_rand[i]) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand))] )   

        f_Ddata = open(distancefile, 'a')
        f_Ddata.write(label + '\t' + str(D) + '\t' + str(sk_results_norm[0]) + '\n')
        f_Ddata.close()
         
    
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
    



def fitAndStatsSkewedNormal(filename, ax, label, outfolder, name, statfile, filterparam, alpha_hist  = 0.2, color_line = 'r'):
   
    if 'log_Q' in name:
        rand = np.asarray([float(line.strip().split('\t')[1]) for line in open(filename) if len(line.strip().split('\t')) > 1])
    else:
        rand = np.asarray([float(line.strip()) for line in open(filename)])

    print 'Fitting normal...'
    param = stats.skewnorm.fit(rand)

    x_rand, p_rand = getDistribution(rand)
    pdf_fitted = stats.skewnorm.pdf(x_rand,  param[0], loc=param[1], scale=param[2])          

    mean, variance, skewness, kurtosity = stats.skewnorm.stats(param[0], loc=param[1], scale=param[2], moments='mvsk')
    maxx = x_rand[pdf_fitted.tolist().index(max(pdf_fitted))]    
    counts, bins, bars = ax.hist(rand, bins = np.linspace(min(x_rand), max(x_rand), 25), normed = True, alpha = 0)
    D = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.skewnorm.cdf(x_rand, param[0], loc=param[1], scale=param[2])) [0]  

    counts, bins, bars = ax.hist(rand, bins = np.linspace(min(x_rand), max(x_rand), 25), normed = True, alpha = alpha_hist)
    ax.plot(x_rand,pdf_fitted,'-', color = color_line, linewidth = 3, label = 'D = ' + str(D) + '\nvarQ = ' + str(variance))
    ax.set_title('Skewed normal fit')    
    legend = ax.legend(loc='left', shadow=True, fontsize = 20)


    write_row(outfolder + '/' + label + '_' + name + '_original_fit'               + '.dat', [str(x_rand[i])        + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand)) ])
    write_row(outfolder + '/' + label + '_' + name + '_mean_centered_fit'          + '.dat', [str(x_rand[i] - mean) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand)) ])
    write_row(outfolder + '/' + label + '_' + name + '_peak_centered_fit'          + '.dat', [str(x_rand[i] - maxx) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand)) ])
    write_row(outfolder + '/' + label + '_' + name + '_mean_centered_fit_sample'   + '.dat', [str(x_rand[i] - mean) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand))[0::filterparam] ])
    write_row(outfolder + '/' + label + '_' + name + '_peak_centered_fit_sample'   + '.dat', [str(x_rand[i] - maxx) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand))[0::filterparam] ])
    write_row(outfolder + '/' + label + '_' + name + '_original_hist_'             + '.dat', rand)
    print 'SKEW',  mean, variance, skewness, kurtosity 
    fout = open(statfile, 'a')
    fout.write(label + '\t' + str(D) + '\t' + str(mean) + '\t' + str(variance) + '\t' + str(skewness) + '\t' + str(kurtosity) + '\n')
    fout.close()


    

def fitAndStatsNormal(filename, ax, label, outfolder, name, statfile, filterparam, alpha_hist  = 0.2, color_line = 'r'):
   
    if 'log_Q' in name:
        rand = np.asarray([float(line.strip().split('\t')[1]) for line in open(filename) if len(line.strip().split('\t')) > 1])
    else:
        rand = np.asarray([float(line.strip()) for line in open(filename)])

    print 'Fitting normal...'
    param = stats.norm.fit(rand)

    print param
    x_rand, p_rand = getDistribution(rand)
    pdf_fitted = stats.norm.pdf(x_rand,  loc=param[0], scale=param[1])          

    mean, variance = stats.norm.stats( loc=param[0], scale = param[1], moments='mv')
    maxx = x_rand[pdf_fitted.tolist().index(max(pdf_fitted))]    
    counts, bins, bars = ax.hist(rand, bins = np.linspace(min(x_rand), max(x_rand), 25), normed = True, alpha = alpha_hist)
    D = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.norm.cdf(x_rand,loc=param[0], scale=param[1])) [0]  

    counts, bins, bars = ax.hist(rand, bins = np.linspace(min(x_rand), max(x_rand), 25), normed = True, alpha = alpha_hist)
    ax.plot(x_rand,pdf_fitted,'-', color = color_line, linewidth = 3, label = 'D = ' + str(D) + '\nvarQ = ' + str(variance))
    ax.set_title('Normal fit')    
    legend = ax.legend(loc='left', shadow=True, fontsize = 20)
 
    print 'NORM',  mean, variance

    write_row(outfolder + '/' + label + '_' + name + '_fnorm_original_fit'               + '.dat', [str(x_rand[i])        + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand)) ])
    write_row(outfolder + '/' + label + '_' + name + '_fnorm_mean_centered_fit'          + '.dat', [str(x_rand[i] - mean) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand)) ])
    write_row(outfolder + '/' + label + '_' + name + '_fnorm_peak_centered_fit'          + '.dat', [str(x_rand[i] - maxx) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand)) ])
    write_row(outfolder + '/' + label + '_' + name + '_fnorm_mean_centered_fit_sample'   + '.dat', [str(x_rand[i] - mean) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand))[0::filterparam] ])
    write_row(outfolder + '/' + label + '_' + name + '_fnorm_peak_centered_fit_sample'   + '.dat', [str(x_rand[i] - maxx) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand))[0::filterparam] ])
    write_row(outfolder + '/' + label + '_' + name + '_fnorm_original_hist_'             + '.dat', rand)
 
    fout = open(statfile, 'a')
    fout.write(label + '\t' + str(D) + '\t' + str(mean) + '\t' + str(variance) + '\n')
    fout.close()
    

    

def fitAndStatsTransformedNormal(filename, ax, label, outfolder, name, statfile, filterparam, alpha_hist  = 0.2, color_line = 'r'):
   
    if 'log_Q' in name:
        rand = np.asarray([float(line.strip().split('\t')[1]) for line in open(filename) if len(line.strip().split('\t')) > 1])
    else:
        rand = np.asarray([float(line.strip()) for line in open(filename)])

    mmin = min(rand)
    rand = [math.log(rr - mmin + 1) for rr in rand]

    print 'Fitting normal...'
    param = stats.norm.fit(rand)

    print param
    x_rand, p_rand = getDistribution(rand)
    pdf_fitted = stats.norm.pdf(x_rand,  loc=param[0], scale=param[1])          

    mean, variance = stats.norm.stats( loc=param[0], scale = param[1], moments='mv')
    maxx = x_rand[pdf_fitted.tolist().index(max(pdf_fitted))]    
    counts, bins, bars = ax.hist(rand, bins = np.linspace(min(x_rand), max(x_rand), 25), normed = True, alpha = alpha_hist)
    D = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.norm.cdf(x_rand,loc=param[0], scale=param[1])) [0]  

    counts, bins, bars = ax.hist(rand, bins = np.linspace(min(x_rand), max(x_rand), 25), normed = True, alpha = alpha_hist)
    ax.plot(x_rand,pdf_fitted,'-', color = color_line, linewidth = 3, label = 'D = ' + str(D) + '\nvarQ = ' + str(variance))
    ax.set_title('Normal fit')    
    legend = ax.legend(loc='left', shadow=True, fontsize = 20) 

    print 'NORM',  mean, variance

    write_row(outfolder + '/' + label + '_' + name + '_tnorm_original_fit'               + '.dat', [str(x_rand[i])        + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand)) ])
    write_row(outfolder + '/' + label + '_' + name + '_tnorm_mean_centered_fit'          + '.dat', [str(x_rand[i] - mean) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand)) ])
    write_row(outfolder + '/' + label + '_' + name + '_tnorm_peak_centered_fit'          + '.dat', [str(x_rand[i] - maxx) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand)) ])
    write_row(outfolder + '/' + label + '_' + name + '_tnorm_mean_centered_fit_sample'   + '.dat', [str(x_rand[i] - mean) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand))[0::filterparam] ])
    write_row(outfolder + '/' + label + '_' + name + '_tnorm_peak_centered_fit_sample'   + '.dat', [str(x_rand[i] - maxx) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand))[0::filterparam] ])
    write_row(outfolder + '/' + label + '_' + name + '_tnorm_original_hist_'             + '.dat', rand)
 
    fout = open(statfile, 'a')
    fout.write(label + '\t' + str(D) + '\t' + str(mean) + '\t' + str(variance) + '\n')
    fout.close()
    

    













   
