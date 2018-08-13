import os
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random
import math
import sys
import pandas as pd
from multiprocessing import Process





''' ============== --------------------------- ============== '''
''' ----------------------   helpers    --------------------- '''
''' ============== --------------------------- ============== '''


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





''' ============== --------------------------- ============== '''
''' ----------------------   parsing    --------------------- '''
''' ============== --------------------------- ============== '''


def read_data(infolder, outfolder, title):

    print 'Reading the data... '

    files   = [infolder + '/' + fn for fn in  os.listdir(infolder)]
    id_data = {}
    nnn     = len(files)

    for ind, fn in enumerate(files):

        if ind % 500 == 0:
            print infolder, '\t', ind, '/', nnn

        imdbid = fn.split('/')[-1].split('_')[0].replace('.dat','')
        data   = []

        for line in open(fn):

            fields = line.strip().split('\t')

            data.append(  (fields[0], float(fields[1]), float(fields[2]) )  )

        id_data[imdbid] = data

        

    fout = open(outfolder + '/' + title + '_id_data.dat', 'w')
    for career in id_data.values():
        fout.write( '\t'.join([str(c[2]) for c in career]) + '\n')
    fout.close()


    fout = open(outfolder + '/' + title + '_career_length.dat', 'w')
    for name, career in id_data.items():   
    #fout.write(name + '\t' + str(len(set([ c[1] for c in career]))) + '\n')
        fout.write(name + '\t' + str(len(career)) + '\n')
    fout.close()



    return id_data
   




''' ============== --------------------------- ============== '''
''' ----------------------   impact     --------------------- '''
''' ============== --------------------------- ============== '''


def get_impact_distribution(id_data, nbins, fileout, title):    

    impacts = []


    for imdbid, data in id_data.items():
        impacts += [d[2] for d in data]
        
    ximpacts,  pimpacts        = getDistribution(impacts, normalized = False)
    bximpacts, bpimpacts, err  = getLogBinnedDistribution(ximpacts, pimpacts, nbins)

    f, ax = plt.subplots(1,1, figsize = (14, 6))
    ax.plot(ximpacts,  pimpacts)
    ax.plot(bximpacts, bpimpacts, color = 'r', linewidth = 3)



    datafolder = 'DataToPlot/1_impact_distribution/'
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)


    fout = open(datafolder + '1_impact_distribution_data_' + title + '.dat', 'w')
    for i in range(len(ximpacts)):
        fout.write( str(ximpacts[i]) + '\t' +  str(pimpacts[i]) + '\n'  )
    fout.close()

    fout = open(datafolder + '1_impact_distribution_binned_' + title + '.dat', 'w')
    for i in range(len(bximpacts)):
        fout.write( str(bximpacts[i]) + '\t' +  str(bpimpacts[i]) + '\n'  )
    fout.close()




    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rating value')
    ax.set_ylabel('Rating frequency')
    ax.set_title(title, fontsize = 17)


    plt.tight_layout()
    plt.savefig(fileout)
    plt.close()





''' ============== --------------------------- ============== '''
''' ----------------------     N*/N     --------------------- '''
''' ============== --------------------------- ============== '''


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



    datafolder = 'DataToPlot/2_N_Nstar/'
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)

    fout = open(datafolder + '2_N_Nstar' + title + '.dat', 'w')
    fout.write('\n'.join([str(n) for n in N_star_N]))
    fout.close()



    plt.tight_layout()
    plt.savefig(fileout)
    plt.close()





''' ============== --------------------------- ============== '''
''' ----------------------   helpers    --------------------- '''
''' ============== --------------------------- ============== '''


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



def get_Q_model_stats(id_data, Qfitparams, fileout, folder2, jind, title):

    imdbid_Q = {}
    imdbid_p = {}

    for ind, (imdbid, data) in enumerate(id_data.items()):
        imdbid_Q[imdbid] = get_Q([d[2] for d in data], Qfitparams)


    Qs = [round(q) for q in imdbid_Q.values() if not np.isnan(q)]
    xQ, pQ = getDistribution(Qs)

    bxQ, bpQ, err = getLogBinnedDistribution(xQ, pQ, nbins)



    ps  = []
    nnn = len(imdbid_Q)

    for ind, (imdb, Q) in enumerate(imdbid_Q.items()):


        if ind % 500 == 0:
            print title, '\t', ind, '/', nnn

        career   = [d[2] for d in id_data[imdb]]
        career_p = get_p(career, Q)
        ps += career_p

 
        imdbid_p[imdb] = career_p




    pss = [round(p) for p in ps]
    xp, pp = getDistribution(pss)
    bxp, bpp, err = getLogBinnedDistribution(xp, pp, nbins)


    title = title.replace('art-', 'art_')

    fout = open(folder2 + '/' + 'p_distribution_' + title + '_' + str(jind) + '.dat', 'w')
    fout.write('\n'.join([str(f) for f in ps]))
    fout.close()

    fout = open(folder2 + '/' + 'Q_distribution_' + title + '_' + str(jind) + '.dat', 'w')
    fout.write('\n'.join([ imdb + '\t' + str(Q)   for imdb, Q in imdbid_Q.items()]))
    fout.close()
    


    datafolder = 'DataToPlot/3_pQ_distributions/'
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)







    fout = open(datafolder + '/' + 'p_distribution_data_' + title + '_' + str(jind) + '.dat', 'w')
    for i in range(len(xp)):
        fout.write( str(xp[i]) + '\t' + str(pp[i]) + '\n')
    fout.close()

    fout = open(datafolder + '/' + 'p_distribution_binned_' + title + '_' + str(jind) + '.dat', 'w')
    for i in range(len(bxp)):
        fout.write( str(bxp[i]) + '\t' + str(bpp[i]) + '\n')
    fout.close()


    
    fout = open(datafolder + '/' + 'p_values_of_users_' + title + '_' + str(jind) + '.dat', 'w')
    for imdb, ppp in imdbid_p.items():
        for pppp in ppp:
            fout.write(imdb + '\t' + str(pppp) + '\n')
    fout.close()



    fout = open(datafolder + '/' + 'Q_distribution_data_' + title + '_' + str(jind) + '.dat', 'w')
    for i in range(len(xQ)):
        fout.write( str(xQ[i]) + '\t' + str(pQ[i]) + '\n')
    fout.close()


    fout = open(datafolder + '/' + 'Q_distribution_binned_' + title + '_' + str(jind) + '.dat', 'w')
    for i in range(len(bxQ)):
        fout.write( str(bxQ[i]) + '\t' + str(bpQ[i]) + '\n')
    fout.close()










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

 

    #plt.show()
    #plt.savefig(fileout)
    plt.close()



    return imdbid_Q, ps






def bests_career_length(nbins, fileout, folder2, folder3, title):




    title = title.replace('art-', 'art_')



    ps       = [float(line.strip()) for line in open(folder2 + '/' + 'p_distribution_' + title + '.dat')]
    imdbid_Q = { line.strip().split('\t')[0] : float(line.strip().split('\t')[1])  for line in open(folder2 + '/' + 'Q_distribution_' + title + '.dat') if len(line.strip().split('\t')) == 2 }    
    careers  = [ [float(fff) for fff in line.strip().split('\t')] for line in open(folder3 + '/' + title.rsplit('_',1)[0].replace('_', '-') + '_id_data.dat')]

    careers_length = { line.strip().split('\t')[0] : int(line.strip().split('\t')[1])  for line in open(folder3 + '/' + title.rsplit('_',1)[0].replace('_', '-') + '_career_length.dat') if len(line.strip().split('\t')) == 2  }      


    ''' JUST THE DATA '''

    N_Istar     = {}
    N_Istar_avg = {}

    print title, '  read ...'

    for ind, career in enumerate(careers):

        N       = len(career)
        Istar   = max(career)
        
        if N not in N_Istar:
            N_Istar[N]  = [Istar]
        else:
            N_Istar[N].append(Istar)
        

    for N, Istars in N_Istar.items():
        N_Istar_avg[N] = math.log(np.mean(Istars))
        
         
    Ns,  Istars     = zip(*[(n, i) for n, i in  N_Istar_avg.items()])
    bNs, bIstars, e = getLogBinnedDistribution(Ns, Istars, nbins)


    ''' ADD THE R MODEL '''


    print title, '  R model ...'

    NsS           = []
    Impacts_S     = []
    N_Istar_S     = {}
    N_Istar_avg_S = {}


    for ind, career in enumerate(careers):
        NsS.append(len(career))
        Impacts_S += career

        
    for i in range(25):        

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
        N_Istar_avg_S[N] = math.log(np.mean(IstarsS))

        
    NsS, IstarsS = zip(*[(n, i) for n, i in  N_Istar_avg_S.items()])   
    bNsS, bIstarsS, err = getLogBinnedDistribution(NsS,   IstarsS, nbins)

   


    ''' ADD THE Q MODEL '''

    print title, '  Q model ...'


    Impacts_Q     = []
    N_Istar_Q     = {}
    N_Istar_avg_Q = {}


    for i in range(25):


        psQ = [p for p in ps]

#        psQ = random.reshuffle(psQ)

        for ind, (imdbid, Q) in enumerate(imdbid_Q.items()):  

            if imdbid in careers_length:
                N       = careers_length[imdbid]
                IstarQs =  max([psQ[ind + ijk] * Q for ijk in range(N)  ])  

                if N not in N_Istar_Q:
                    N_Istar_Q[N] = [IstarQs]
                else:
                    N_Istar_Q[N].append(IstarQs)


    for n, istars in N_Istar_Q.items():        
        N_Istar_avg_Q[n] = math.log(np.mean(istars))
    

 
    NsSQ,  IstarsSQ = zip(*[(n, i) for n, i in  N_Istar_avg_Q.items()])   
    bNsSQ, bIstarsSQ, err = getLogBinnedDistribution(NsSQ,   IstarsSQ, nbins)

        




    print title, '  plot ...'


    f, ax = plt.subplots(1,1, figsize = (12, 7))
    ax.plot(Ns,  Istars, 'o', markersize = 12, alpha = 0.25, color = 'lightgrey', label = 'Data')#    ax.plot(bNs, bIstars, 'r', linewidth = 3)
    ax.plot(bNs, bIstars,  'o', color = 'r', markersize = 15, label = 'Binned data')
    ax.errorbar(bNs, bIstars, yerr = e, color = 'r', markersize = 15)
    ax.fill_between(bNs, np.asarray(bIstars)- np.asarray(e), np.asarray(bIstars) + np.asarray(e), color = 'r', alpha = 0.1)

    ax.plot(bNsS,   bIstarsS,  'k', linewidth = 3, label= 'Rmodel')
    ax.plot(bNsSQ,  bIstarsSQ, 'g', linewidth = 5, color = 'steelblue', label = 'Qmodel')
    ax.set_xscale('log')
    ax.set_ylim([1, 1.2*max(bIstarsS)])
    ax.legend(loc = 'best', fontsize = 13)
    ax.set_xlabel('Career length', fontsize = 15)
    ax.set_ylabel('Log I*', fontsize = 15)


 

    field    = title.split('_')[0].split('-')[0]
    length   = title.split('-')[-1].split('_')[0]
    solution = title.split('_')[-1]

    if 'art' == field:
        field = 'art director'





    datafolder = 'DataToPlot/4_RQModel/'
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)

    fout = open(datafolder + '4_RQModel_data_' + title + '.dat', 'w')
    for i in range(len(Ns)):
        fout.write( str(Ns[i]) + '\t' + str(Istars[i])  + '\n' )
    fout.close()


    fout = open(datafolder + '4_RQModel_binned_' + title + '.dat', 'w')
    for i in range(len(bNs)):
        fout.write( str(bNs[i]) + '\t' + str(bIstars[i]) + '\t' + str(e[i]) + '\n' )
    fout.close()


    fout = open(datafolder + '4_RQModel_rmodel_' + title + '.dat', 'w')
    for i in range(len(bNsS)):
        fout.write( str(bNsS[i]) + '\t' + str(bIstarsS[i]) + '\n' )
    fout.close()

    fout = open(datafolder + '4_RQModel_qmodel_' + title + '.dat', 'w')
    for i in range(len(bNsSQ)):
        fout.write( str(bNsSQ[i]) + '\t' + str(bIstarsSQ[i]) + '\n' )
    fout.close()






    fileout  = 'ResultFigs/4_R_Q_model_test_'  + title + '.png'
    title    = field + ',  mincareer length = ' + length + ',   optimization solution = ' + solution

    ax.set_title(title, fontsize = 18)










    plt.tight_layout()
    plt.savefig(fileout)
    plt.close()

   # plt.show()



















def process_Qs_paralel(resfile):


    field_o    = resfile.split('_', 1)[1]
    limit      = field_o.split('-')[1]
    field      = field_o.split('-')[0].replace('_','-')
    infolder   = 'Data/' + field + '/' + fields[field] + '-' + field + '-simple-careers-limit-' + limit
    Qfitparams = []

    for ind, line in enumerate(open('Qparamfit/' + field.replace('-', '_') +  '-' + str(limit) + '_qmodel_params.dat')):

        if ind == 3: break

        mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN = [float(f) for f in line.strip().split('\t')][1:]
        
        Qfitparams = (mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)



      #  id_data = read_data(infolder, folderout3, field + '-' + str(limit))
     #   get_impact_distribution(id_data, nbins, folderout + '1_impact_distribution_' + field_o + '.png', field_o) 
     #   get_N_star_N(           id_data, nbins, folderout + '2_N_star_N_' + field_o + '.png', field_o)
     #   get_Q_model_stats(id_data, Qfitparams, folderout + '3_p_and_Q_distr_' + field_o + '_' + str(ind) + '.png', folderout2, ind, field_o)	   

        bests_career_length( nbins, folderout + '4_R_Q_model_test_'  +  field + '-' + str(limit) + '.png',  folderout2, folderout3, field.replace('-','_') + '-' + str(limit) + '_' + str(ind))





if __name__ == '__main__':  


    fields = {  'director'     : 'film', 
                'art-director' : 'film', 
                'producer'     : 'film', 
                'writer'       : 'film', 
                'composer'     : 'film', 
                'electro'      : 'music', 
                'rock'        : 'music', 
                'pop'          : 'music', 
                'funk'         : 'music', 
                'folk'         : 'music', 
                'hiphop'       : 'music', 
                'jazz'         : 'music', 
                'classical'    : 'music', 
                'authors'      : 'book' }


    nbins      = 10
    resfolder  = 'Optimize/atlasz/evolution/test/Results/'
    resfiles   = [resfolder + res for res in os.listdir(resfolder)]
    folderout  = 'ResultFigs/' 
    folderout2 = 'pQData/' 
    folderout3 = 'IdData/' 


    if not os.path.exists(folderout):  os.makedirs(folderout)
    if not os.path.exists(folderout2): os.makedirs(folderout2)
    if not os.path.exists(folderout3): os.makedirs(folderout3)


    if sys.argv[1] == 'auto':

        Pros = []
   
        for resfile in resfiles:
            p = Process(target = process_Qs_paralel, args=(resfile, ))
            Pros.append(p)
            p.start()
           
        for t in Pros:
            t.join()










    elif sys.argv[1] == 'manual':

        label    = sys.argv[2]
        field    = sys.argv[3]  
        LIMIT    = int(sys.argv[4])
        infolder = 'Data/' + field + '/' + fields[field] + '-' + field + '-simple-careers-limit-' + str(LIMIT)

        for ind, line in enumerate(open('Qparamfit/' + field.replace('-', '_') +  '-' + str(LIMIT) + '_qmodel_params.dat')):

            if ind ==0:
                mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN = [float(f) for f in line.strip().split('\t')][1:]
            
        Qfitparams = (mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)



        id_data = read_data(infolder, folderout3, field + '-' + str(LIMIT))
   #     get_impact_distribution(id_data, nbins, folderout + '1_impact_distribution_' +  field + '-' + str(LIMIT) + '.png',  field + '-' + str(LIMIT)) 
      #  get_N_star_N( id_data, nbins, folderout + '2_N_star_N_' +  field + '-' + str(LIMIT)  + '.png',  field + '-' + str(LIMIT) )  
        get_Q_model_stats(id_data, Qfitparams, folderout + '3_p_and_Q_distr_' + field   + '-' + str(LIMIT) + '.png', folderout2, 0, field + '-' + str(LIMIT))	  
  #      bests_career_length( nbins, folderout + '4_R_Q_model_test_'  +  field + '-' + str(LIMIT) + '.png',  folderout2, folderout3, field.replace('-','_') + '-' + str(LIMIT) + '_0')
        



