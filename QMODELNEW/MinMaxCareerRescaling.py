import numpy as np
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
import os
import sys
import math
import sys




def getDistribution(keys, normalized = True):
    
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    distr = np.bincount(bins) 

    if normalized == 1: distr = distr/float(np.sum(distr)) 
    
    return np.asarray(uniq_keys.tolist()), np.asarray(distr.tolist())




def get_impact_values(folder):
    
    #files = sorted(os.listdir(folder))[0:1000]
    files = os.listdir(folder)   
    
    yall = []
    for fn in files:
        
        try:
            imdb, X, Y, cat = zip(*[line.strip().split('\t') for line in open(folder+ fn) if 'year' not in line] )
        except:
            imdb, X, Y = zip(*[line.strip().split('\t') for line in open(folder+ fn) if 'year' not in line] )

        Y = [float(yyy) for yyy in Y]
        yall += Y
        
    impactmin = min(yall)
    impactmax = max(yall)
    
    if impactmin == 0.0: impactmin = 1.0 
    
    return impactmin, impactmax




def get_data_distr(folder):
    
    #files = sorted(os.listdir(folder))[0:1000]
    files = os.listdir(folder)

    yall = []
    for fn in files:
        
        try:
            imdb, X, Y, cat = zip(*[line.strip().split('\t') for line in open(folder+ fn) if 'year' not in line] )
        except:
            imdb, X, Y = zip(*[line.strip().split('\t') for line in open(folder+ fn) if 'year' not in line] )
      
        Y = [float(yy) for yy in Y]
        yall += Y
    
    fig, ax  = plt.subplots(1,1, figsize = (15,7))
    nbins = 10
    minx = min(yall)
    if minx == 0: minx = 1
    pxall, pyall = getDistribution(yall, True)        
    counts, bins, bars = ax.hist(yall, normed = True, bins = 10 ** np.linspace(np.log10(minx), np.log10(max(pxall)), nbins), log=True,alpha=0.0, cumulative=0)
    ax.plot((bins[1:] + bins[:-1])/2, counts, 's-', color = 'royalblue', alpha = 0.0, markersize = 0, linewidth = 5)
    bins = (bins[1:] + bins[:-1])/2    
                    
    plt.close()
    
    return (bins, counts)





def get_impact(impacts):
    
    fig, ax  = plt.subplots(1,1, figsize = (15,7))
    nbins = 20
    minx = min(impacts)
    pxall, pyall = getDistribution(impacts, True)    
    if minx == 0: minx = 1
    counts, bins, bars = ax.hist(impacts, normed = True, bins = 10 ** np.linspace(np.log10(minx), np.log10(max(pxall)), nbins), log=True,alpha=0.0, cumulative=0)
    ax.plot((bins[1:] + bins[:-1])/2, counts, 's-', color = 'royalblue', alpha = 0.0, markersize = 0, linewidth = 5)
    bins = (bins[1:] + bins[:-1])/2    
                    
    plt.close()
    
    return (bins, counts)




# https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
def linrescale(impact, GLOBALMAX, MINMAX):   
    miny, maxy = MINMAX
    impact = impact
    miny   = miny
    maxy   = maxy
    maxG   = GLOBALMAX

    newimpact = (impact - miny) / (maxy - miny) * (maxG)  
    
    return newimpact

def logrescale(impact, GLOBALMAX, MINMAX):   
    miny, maxy = MINMAX
    impact = math.log(impact)
    miny   = math.log(miny)
    maxy   = math.log(maxy)
    maxG   = math.log(GLOBALMAX)

    newimpact = math.exp(  (impact - miny) / (maxy - miny) * (maxG)  )
    
    return newimpact






fields = [ ('art-director'       , 'Data/art-director/film-art-director-simple-careers-limit-20/'),
           ('writer'             , 'Data/writer/film-writer-simple-careers-limit-10/'),
           ('composer'           , 'Data/composer/film-composer-simple-careers-limit-10/'),
           ('authors'            , 'Data/authors/book-authors-simple-careers-limit-10/'),
          ('pop'         , 'Data/pop/music-'+'pop'+'-simple-careers-limit-80/')   ,
          ('rock'        , 'Data/rock/music-'+'rock'+'-simple-careers-limit-80/')   ,
          ('electro'     , 'Data/electro/music-'+'electro'+'-simple-careers-limit-80/')   ,
          ('folk'        , 'Data/folk/music-'+'folk'+'-simple-careers-limit-80/')   ,
          ('funk'        , 'Data/funk/music-'+'funk'+'-simple-careers-limit-80/')   ,
          ('hiphop'      , 'Data/hiphop/music-'+'hiphop'+'-simple-careers-limit-80/')   ,

            ('engineering', '../Data/Science/science-' + 'engineering' + '-simple-careers/'),
            ('political_science', '../Data/Science/science-' + 'political_science' + '-simple-careers/'),
            ('environmental_science', '../Data/Science/science-' + 'environmental_science' + '-simple-careers/'),
            ('psychology', '../Data/Science/science-' + 'psychology' + '-simple-careers/'),
            ('agronomy', '../Data/Science/science-' + 'agronomy' + '-simple-careers/'),
            ('geology', '../Data/Science/science-' + 'geology' + '-simple-careers/'),
            ('space_science_or_astronomy', '../Data/Science/science-' + 'space_science_or_astronomy' + '-simple-careers/'),
            ('applied_physics', '../Data/Science/science-' + 'applied_physics' + '-simple-careers/'),
            ('health_science', '../Data/Science/science-' + 'health_science' + '-simple-careers/'),
            ('theoretical_computer_science', '../Data/Science/science-' + 'theoretical_computer_science' + '-simple-careers/'),
            ('biology', '../Data/Science/science-' + 'biology' + '-simple-careers/'),
            ('mathematics', '../Data/Science/science-' + 'mathematics' + '-simple-careers/'),
            ('unfiled', '../Data/Science/science-' + 'unfiled' + '-simple-careers/'),
            ('chemistry', '../Data/Science/science-' + 'chemistry' + '-simple-careers/'),
            ('physics', '../Data/Science/science-' + 'physics' + '-simple-careers/'),
            ('zoology', '../Data/Science/science-' + 'zoology' + '-simple-careers/')

         ]



fields = [ ('director'       , 'Data/art-director/film-director-simple-careers-limit-10/'),
           ('producer'       , 'Data/writer/film-producer-simple-careers-limit-10/'),
         
          ('classical'         , 'Data/pop/music-'+'classical'+'-simple-careers-limit-80/')   ,
          ('jazz'        , 'Data/rock/music-'+'jazz'+'-simple-careers-limit-80/')   ,
          ('electro'     , 'Data/electro/music-'+'electro'+'-simple-careers-limit-80/') 

         ]










GLOBALMAX    = 0
GENRE_MINMAX = {}

for field, folder in fields:
    #if 'math' in field or 'dir' in field:
    if 2 == 2:
        print field
        impactmin, impactmax = get_impact_values(folder)
        if impactmax > GLOBALMAX:
            GLOBALMAX = impactmax
        GENRE_MINMAX[field] = (impactmin, impactmax)


#print GLOBALMAX
#print GENRE_MINMAX






f, ax = plt.subplots(1,3,figsize = (15,4))

print 'STAAART'

for field, folder in fields:
    
    #if 'math' in field or 'dir' in field:
    if 2 == 2:

        impacts_genre_log = []
        impacts_genre_o   = []
        impacts_genre_lin = []
     
        #folderout = folder.replace('../', '').replace('/Science/', '/').split('-limit')[0].replace('Data', 'Data_linrescaled')     
        #folderout = folderout.replace(field + '/', '', 1) 
        #if not os.path.exists(folderout):
        #    os.makedirs(folderout)
        
        
        MINMAX = GENRE_MINMAX[field]
        miny, maxy  = MINMAX
        files = os.listdir(folder)
        
        nnn = len(files)

        print field, len(files)
        
        for ind, fn in enumerate(files):
            
            print ind, '/', nnn   

            #fileout = open(folderout + '/' + fn, 'w')
            
            for line in open(folder+fn):
                
                cat = ''
                
                if 'year' not in line:
                    if 'science' in folder:
                        paper_id, year, c10, cat = line.strip().split('\t')
                        c10 = float(c10)
                    else:
                        paper_id, year, c10 = line.strip().split('\t')
                        c10 = float(c10)


                    if c10 == 0: c10 = 1 
                    c100 = float(c10)

                    c10 = linrescale(c10, GLOBALMAX, MINMAX)           
                    impacts_genre_lin.append(c10)


                    impacts_genre_o.append(c10)

                    c12 = logrescale(c100, GLOBALMAX, MINMAX)
                    impacts_genre_log.append(c12)               


                    
                    #if len(cat) > 0:          
                    #    fileout.write(paper_id + '\t' + year + '\t' + str(c10) + '\t' + cat + '\n')
                    #else:
                    #    fileout.write(paper_id + '\t' + year + '\t' + str(c10) + '\n')
            
                        

            #fileout.close()        
                    


            

        
        Xy, Yy = get_impact(impacts_genre_o)
        ax[0].plot(Xy,Yy, 'o-', label = field)    
              
        Xy, Yy = get_impact(impacts_genre_lin)
        ax[1].plot(Xy,Yy, 'o-', label = field) 

       # print set(impacts_genre_lin)

        Xy, Yy = get_impact(impacts_genre_log)
        ax[2].plot(Xy,Yy, 'o-', label = field)


        fout =  open('DataToPlot_linrescaled_final/1_impact_distribution/' + '1_impact_distribution_RESCALED_'+field+'.dat', 'w')
        for i in range(len(Xy)):
            fout.write(  str(Xy[i])    + '\t' + str(Yy[i])   + '\n'    )  
        fout.close()

        


  
ax[0].legend(loc = 'best')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_title('Original', fontsize = 17)
 
ax[1].legend(loc = 'best')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_title('Lin Rescale', fontsize = 17)    
    
ax[2].legend(loc = 'best')
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].set_title('Log Rescaled', fontsize = 17)


plt.show()
#plt.savefig('20181120_4_demo_impact_distr_rescaled.png')




