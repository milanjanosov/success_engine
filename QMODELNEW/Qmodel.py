import os
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
import sys
import pandas as pd




def getDistribution(keys, normalized = True):
    
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    distr = np.bincount(bins) 

    if normalized == 1: distr = distr/float(np.sum(distr)) 

    return np.asarray(uniq_keys.tolist()), np.asarray(distr.tolist())


def getBinnedDistribution(x, y, nbins):

    n, bins   = np.histogram(x, bins=nbins)
    sy, _  = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy/n
     
    std = np.sqrt(sy2/n - mean*mean) 

    return _, mean, std


def getLogBinnedDistribution(x, y, nbins):

    bins   = 10 ** np.linspace(np.log10(min(x)), np.log10(max(x)), nbins)  
    values = [ np.mean([y[j]  for j in range(len(x)) if x[j] >= bins[i] and x[j] < bins[i+1]])  for i in range(nbins-1)]    
    error  = [ np.std( [y[j]  for j in range(len(x)) if x[j] >= bins[i] and x[j] < bins[i+1]])  for i in range(nbins-1)]
    bins   = (bins[1:] + bins[:-1])/2

    return bins, values, error


def getPercentileBinnedDistribution(x, y, nbins):

    x, y = zip(*sorted(zip(x, y), key=lambda tup: tup[0]))
    elements_per_bin = int(len(x)/float(nbins))

    xx  = [np.mean(x[i*elements_per_bin:(i+1)*elements_per_bin]) for i in range(nbins)]
    yy  = [np.mean(y[i*elements_per_bin:(i+1)*elements_per_bin]) for i in range(nbins)]
    std = [np.std(y[i*elements_per_bin:(i+1)*elements_per_bin])  for i in range(nbins)]

    return xx, yy, std


def divideUnequal(list1, list2):
    counter=0
    step=0
    divided=[]
    for count in list1:
        count = int(count)
        step= counter+ count
        sublist= list2[counter: step]
        counter= step
        divided.append(sublist)
    return divided








def read_data(infolder):

    files   = [infolder + '/' + fn for fn in  os.listdir(infolder)]
    id_data = {}

    for fn in files:

        imdbid = fn.split('/')[-1].split('_')[0].replace('.dat','')
        data   = []

        for line in open(fn):
            data.append(  (line.strip().split('\t')[0], float(line.strip().split('\t')[1]), float(line.strip().split('\t')[2]) )  )

        id_data[imdbid] = data

    return id_data
   


def get_impact_distribution(id_data, nbins, fileout, title):    

    impacts = []


    for imdbid, data in id_data.items():
        impacts += [d[2] for d in data]
        
    ximpacts,  pimpacts        = getDistribution(impacts, normalized = False)
    bximpacts, bpimpacts, err  = getLogBinnedDistribution(ximpacts, pimpacts, nbins)

    f, ax = plt.subplots(1,1, figsize = (14, 6))
    ax.plot(ximpacts,  pimpacts)
    ax.plot(bximpacts, bpimpacts, color = 'r', linewidth = 3)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rating value')
    ax.set_ylabel('Rating frequency')
    ax.set_title(title, fontsize = 17)


    plt.tight_layout()
    plt.savefig(fileout)
    plt.close()
  #  plt.show()












def get_N_star_N(id_data, bins, fileout, title):

    N_star_N   = []
    N_star_N_r = []

    for ind, (imdbid, data) in enumerate(id_data.items()):
        

        data.sort(key=lambda tup: tup[1]) 
        
        
        titles, times, impacts = zip(*data)
                  
        #if ind == 100: break

        N      = len(impacts)
        Istar  = max(impacts)    
        Nstars = [i for i, j in enumerate(impacts) if j == Istar]  
        
        for Nstar in Nstars: 
            N_star_N.append(float(Nstar)/N)

            

    f, ax = plt.subplots(1,1, figsize = (7, 6))

    x = np.arange(0,1, 0.1)
    ax.hist(N_star_N, bins = 100, cumulative = True, normed = True, alpha = 0.5)
    ax.plot(x,x, color ='r', linewidth = 4)
    ax.set_title(title, fontsize = 17)

    plt.tight_layout()
    plt.savefig(fileout)
    plt.close()







def get_Q(career,  Qfitparams):


    mu_N     = Qfitparams[0]
    mu_p     = Qfitparams[1]
    mu_Q     = Qfitparams[2]
    sigma_N  = Qfitparams[3]
    sigma_Q  = Qfitparams[4]
    sigma_p  = Qfitparams[5]
    sigma_pQ = Qfitparams[6]
    sigma_pN = Qfitparams[7]
    sigma_QN = Qfitparams[8]
    K_QN     = sigma_Q**2 * sigma_N**2 - sigma_QN**2  



    logN_i    = math.log(len(career))
    N_i       = len(career)
    avg_log_c = np.mean([ np.log(e) for e in career if e > 0])

    
    TERM1 = (  sigma_N**2 * sigma_p**2 * mu_Q + sigma_QN * sigma_p**2 * (logN_i - mu_N)  ) / (N_i * K_QN)
    TERM2 = sigma_N**2 * sigma_p**2 / ( N_i * K_QN)

    logQ  =  (avg_log_c - mu_p + TERM1)  /  (1.0 + TERM2)
    
    return math.exp(logQ)




def get_p(career, Q):

    logQ        = math.log(Q)
    log_impacts = [math.log(e) for e in career if e > 0]    

    return [math.exp(i - logQ) for i in log_impacts]



def get_Q_model_stats(id_data, Qfitparams, fileout, title):

    imdbid_Q = {}

    for ind, (imdbid, data) in enumerate(id_data.items()):
        imdbid_Q[imdbid] = get_Q([d[2] for d in data], Qfitparams)


    Qs = [round(q) for q in imdbid_Q.values() if not np.isnan(q)]
    xQ, pQ = getDistribution(Qs)

    bxQ, bpQ, err = getLogBinnedDistribution(xQ, pQ, nbins)



    ps = []

    for imdb, Q in imdbid_Q.items():
        career = [d[2] for d in id_data[imdb]]
        ps += get_p(career, Q)


    pss = [round(p) for p in ps]
    xp, pp = getDistribution(pss)
    bxp, bpp, err = getLogBinnedDistribution(xp, pp, nbins)



    f, ax = plt.subplots(1,2, figsize = (12, 5))
    ax[0].set_title('Q - ' + title, fontsize = 17)
    ax[0].plot(xQ, pQ, 'o')
    ax[0].plot(bxQ, bpQ, 'r-')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[1].set_title('p - ' + title, fontsize = 17)
    ax[1].plot(xp, pp, 'o')
    ax[1].plot(bxp, bpp, 'r-')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')

 
    plt.tight_layout()
    plt.savefig(fileout)
    plt.close()



    return imdbid_Q, ps






def bests_career_length(id_data, imdbid_Q, ps, nbins, fileout, title):


    ''' JUST THE DATA '''

    N_Istar     = {}
    N_Istar_avg = {}

    for ind, (imdbid, data) in enumerate(id_data.items()):

        N       = len(data)
        Istar   = max([d[2] for d in data])
        
        if N not in N_Istar:
            N_Istar[N]  = [Istar]
        else:
            N_Istar[N].append(Istar)
        

    for N, Istars in N_Istar.items():
        N_Istar_avg[N] = np.mean(Istars)
        
         
    Ns,  Istars     = zip(*[(n, i) for n, i in  N_Istar_avg.items()])
    bNs, bIstars, e = getLogBinnedDistribution(Ns, Istars, nbins)




    ''' ADD THE R MODEL '''

    NsS           = []
    Impacts_S     = []
    N_Istar_S     = {}
    N_Istar_avg_S = {}


    for ind, (imdbid, data) in enumerate(id_data.items()):
        NsS.append(len(data))
        Impacts_S += [d[2] for d in data]

        
    for i in range(20):        

        random.shuffle(Impacts_S)    
        Scareers = divideUnequal(NsS, Impacts_S) 
        
        for ind, data in enumerate(Scareers):
            N     = len(data)
            Istar = max(data)


            if N not in N_Istar_S:
                N_Istar_S[N]  = [Istar]
            else:
                N_Istar_S[N].append(Istar)


    for N, IstarsS in N_Istar_S.items():
        N_Istar_avg_S[N] = np.mean(IstarsS)

        
    NsS, IstarsS = zip(*[(n, i) for n, i in  N_Istar_avg_S.items()])   
    bNsS, bIstarsS, err = getLogBinnedDistribution(NsS,   IstarsS, nbins)




    ''' ADD THE Q MODEL '''



    Impacts_Q     = []
    N_Istar_Q     = {}
    N_Istar_avg_Q = {}


    for i in range(20):

        psQ = [p for p in ps]
        random.shuffle(psQ)    


        for ind, (imdbid, Q) in enumerate(imdbid_Q.items()):  

            N  = len([d for d in id_data[imdbid]])

  

            career = []
            for i in range(N):
                career.append(psQ[0] * Q)
                del psQ[0]
            IstarQs = max(career)


            if N not in N_Istar_Q:
                N_Istar_Q[N] = [IstarQs]
            else:
                N_Istar_Q[N].append(IstarQs)


    for n, istars in N_Istar_Q.items():        
        N_Istar_avg_Q[n] = np.mean(istars)
    

    
 
    NsSQ,  IstarsSQ = zip(*[(n, i) for n, i in  N_Istar_avg_Q.items()])   
    bNsSQ, bIstarsSQ, err = getLogBinnedDistribution(NsSQ,   IstarsSQ, nbins)

        














    f, ax = plt.subplots(1,1, figsize = (7, 6))
    ax.plot(Ns,  Istars, 'o')
    ax.plot(bNs, bIstars, 'r', linewidth = 3)

    ax.plot(bNsS, bIstarsS, 'k', linewidth = 3)
    ax.plot(bNsSQ,  bIstarsSQ, 'g', linewidth = 3)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_title(title, fontsize = 17)

    plt.tight_layout()
    plt.savefig(fileout)
    plt.close()













if __name__ == '__main__':  



    

    fields = {  'director'     : 'film', 
                'art-director' : 'film', 
                'producer'     : 'film', 
                'writer'       : 'film', 
                'composer'     : 'film', 
                'electro'      : 'music', 
                'rock'         : 'music', 
                'pop'          : 'music', 
                'funk'         : 'music', 
                'folk'         : 'music', 
                'hiphop'       : 'music', 
                'jazz'         : 'music', 
                'classical'    : 'music', 
                'authors'      : 'book' }




    nbins     = 12
    resfolder = 'Optimize/atlasz/evolution/test/Results/'
    resfiles  = [resfolder + res for res in os.listdir(resfolder)]



    if sys.argv[1] == 'auto':


        opt_rank = 0



        for resfile in resfiles:

            field_o   = resfile.split('_', 1)[1]
            limit     = field_o.split('-')[1]
            field     = field_o.split('-')[0].replace('_','-')
            infolder  = 'Data/' + field + '/' + fields[field] + '-' + field + '-simple-careers-limit-' + limit
            folderout = 'ResultFigs/' 


            if not os.path.exists(folderout):
                os.makedirs(folderout)

            print field_o

            Qfitparams = []
 
            for ind, line in enumerate(open('../QFitConstants/' + field + '.dat')):
                if ind == opt_rank:
                    mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN = [float(f) for f in line.strip().split('\t')][1:]
                
            Qfitparams = (mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)


            id_data = read_data(infolder)
            get_impact_distribution(id_data, nbins, folderout + '1_impact_distribution_' + field_o + '.png', field_o) 
            get_N_star_N(           id_data, nbins, folderout + '2_N_star_N_' + field_o + '.png', field_o)
        
            imdbid_Q, ps = get_Q_model_stats(id_data, Qfitparams, folderout + '3_p_and_Q_distr_' + field_o + '.png', field_o)	   

            if len(Qfitparams) > 0:
                bests_career_length(id_data, imdbid_Q, ps, nbins, folderout + '4_R_Q_model_test_' + field_o + '_' + str(opt_rank) + '.png', field_o)



    elif sys.argv[1] == 'manual':

        label    = sys.argv[2]
        field    = sys.argv[3]  #'director'
        LIMIT    = int(sys.argv[4])

        infolder = 'Data/' + label + '-' + field + '-simple-careers'


        '''or ind, line in enumerate(open('../QFitConstants/' + field + '.dat')):
            if ind == 1:
                mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN = [float(f) for f in line.strip().split('\t')][1:]
            
        Qfitparams = (mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)





        id_data = read_data(infolder, LIMIT)


        #get_impact_distribution(id_data, nbins) 
        #get_N_star_N(id_data, nbins)
        imdbid_Q, ps = get_Q_model_stats(id_data, Qfitparams)	
        bests_career_length(id_data, imdbid_Q, ps, nbins)

        '''

