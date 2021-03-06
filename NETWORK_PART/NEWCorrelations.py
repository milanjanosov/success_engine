import numpy as np
import os
import sys  
import gzip
import math
import random
import scipy.stats
from dtw import dtw
from numpy.linalg import norm
from scipy.stats import spearmanr


measures      = ['degree', 'clustering', 'pagerank', 'betweenness', 'closeness', 'constraint']  
top_directors = {   'nm0000184' : 'G Lucas',
                    'nm0000233' : 'Q Tarantino',
                    'nm0000229' : 'S Spielberg',
                    'nm0000040' : 'S Kubrick',
                    'nm0634240' : 'C Nolan',
                    'nm0000033' : 'A Hitchcock',
                    'nm0000122' : 'Charlie Chaplin',
                    'nm0000631' : 'Ridley Scott',
                    'nm0001053' : 'E Coen',
                    'nm0000142' : 'C Eastwood',
                    'nm0001392' : 'P Jackson',
                    'nm0000591' : 'R Polanski',
                    'nm0000154' : 'M Gibson',
                    'nm0001232' : 'M Forman',   
                    'nm0001628' : 'S Pollack'}




def get_year(year):

    if len(year) == 4: return int(year)
    else: return int(year.split('-')[0])





def get_plot_correl(centralities, dirid, name, meas, logimpact = False):
    
    x = []
    y = []
    z = []

    for line in gzip.open('simple-careers/film-director-simple-careers/' + dirid + '_director_simple_career.gz'):

        if 'year' not in line:

            fields = line.strip().split('\t')
            if len( fields[1] ) > 0:
                year   = get_year(fields[1])

                if 'None' != fields[3]:

                    impact = float(fields[3])

                    if year in centralities:

                        if impact > 0.0:

                            x.append(year)
                            
                            if logimpact:
                                if impact == 0.0: impact = 1.0
                                y.append(math.log(impact))
                            else:
                                y.append(impact)
                            z.append(centralities[year])
                    
    if len(y) > 1:

        y, z, x = zip(*sorted([ (y[ijk], z[ijk], x[ijk]) for ijk in range(len(x)) ], key=lambda tup: tup[2]))    
        zavg    = np.mean(z)

        z    = [zz/zavg for zz in z]
        yavg = np.mean(y)
        y    = [yy/yavg for yy in y]
        a    = list(y)
        random.shuffle(a)

        
        return y, z, spearmanr(y, z)[0], spearmanr(a, z)[0]







def shift_time_series(x, y, tau ):
    
    
    x2 = []
    y2 = []
    
    if tau >= 0:
        for i in range(tau, len(x)):
            x2.append(x[i])

        for i in range(len(y)-tau):
            y2.append(y[i])        
    
    else:
        for i in range(len(x)+tau):
            x2.append(x[i])        
        for i in range(-tau, len(y)):
            y2.append(y[i])  

    a = list(x2)
    random.shuffle(a)
    
    return x2, y2, spearmanr(x2, y2)[0], spearmanr(a, y2)[0]



def get_tau_star(x,y):
    
    n       = len(x)/2
    maxc    = -10
    taustar = -1000000000
    
    maxcR    = -10
    taustarR = -1000000000



    #for tau in range(-n, n):
    for tau in range(-n, n):
        xx, yy, corr, rand = shift_time_series(x, y, tau)  
        if corr > maxc:
            taustar = tau  
            maxc = corr


        if rand > maxcR:
            taustarR = tau  
            maxcR = rand
          
    xstar, ystar, corrstar, aa = shift_time_series(x, y, taustar)    
    xstar, ystar, aa, corrand  = shift_time_series(x, y, taustarR)    
            
    return taustar, xstar, ystar, corrstar, taustarR, corrand
            
   


def dtw_timeserires(x, y):
    
    x_dtw = np.array(x).reshape(-1, 1)
    y_dtw = np.array(y).reshape(-1, 1)
    dist, cost, acc, path = dtw(x_dtw, y_dtw, dist=lambda x_dtw, y_dtw: norm(x_dtw - y_dtw, ord=1))
    map_x, map_y = path
   
    xnew = x_dtw[map_x]
    ynew = y_dtw[map_y]

    cdtw = spearmanr(xnew, ynew)
    return xnew, ynew, cdtw[0]




def get_centralities(dirid, measures, column = 1):

    centralities = {}

    if justQ == 'Q':
        for line in open('NEWTemporal/4_directors_centralities_QEVER/' +dirid+'.dat'):        
            fields = line.strip().split('\t')
            fields = [float(f) for f in fields]
            centralities[fields[0]] = fields[column]


    else:
        for line in open('NEWTemporal/4_directors_centralities/' +dirid+'.dat'):
            fields = line.strip().split('\t')
            fields = [float(f) for f in fields]
            centralities[fields[0]] = fields[column]


        
    return centralities, measures[column-1]


#_QEVER


justQ = sys.argv[1]



if justQ == 'Q':

    fout      = open('NEWTemporal/1_career_centrality_correlations_QEVER.dat', 'w')
    gout      = open('NEWTemporal/2_shiftwindow_sizes_QEVER.dat', 'w')
    ggout     = open('NEWTemporal/2_shiftwindow_sizes_QEVER_random.dat', 'w')
    jout      = open('NEWTemporal/3_corr_shift_QEVER.dat', 'w')
    jjout     = open('NEWTemporal/3_corr_shift_QEVER_random.dat', 'w')

   
    directors = [aaa.replace('.dat', '') for aaa in os.listdir('NEWTemporal/4_directors_centralities_QEVER')]


else:
    fout      = open('NEWTemporal/1_career_centrality_correlations.dat', 'w')
    gout      = open('NEWTemporal/2_shiftwindow_sizes.dat', 'w')
    ggout       = open('NEWTemporal/2_shiftwindow_sizes_random.dat', 'w')
    jout      = open('NEWTemporal/3_corr_shift.dat', 'w')
    jjout      = open('NEWTemporal/3_corr_shift_random.dat', 'w')
    directors  = [aaa.replace('.dat', '') for aaa in os.listdir('NEWTemporal/4_directors_centralities')]






#fout.write('directorid\tcareer_length\tc_random\tc_original\tc_shifted\tc_dtw\n')

nnn = len(directors)



for i in range(1):

    for ind, directorid in enumerate(directors):

        centralities, meas  = get_centralities(directorid, measures, column = 3)

   #    if 'nm0000184' == directorid:
        if 2 == 2:
            print ind, '/', nnn
            results = get_plot_correl(centralities, directorid, '', meas, logimpact = False)

           # if ind == 1000: break
           # print meas

            if results:
         
                x,   y,   c0, c_rand  = results


                #print len(x)
                taustar, xstar, ystar, rc_tau, rtaustar, corrand = get_tau_star(x,y)



      
    
                print  taustar, rc_tau, '\t',  rtaustar, corrand

                x_d, y_d, cdtw  = dtw_timeserires(x,y)

                if taustar !=  -1000000000:

                    fout.write(directorid + '\t' + str(len(x)) + '\t' + str(c_rand) + '\t' + str(c0) + '\t' + str(rc_tau)+ '\t' + str(corrand) + '\t' + str(cdtw) + '\n')
                    gout.write( str(taustar) + '\n')
                    ggout.write( str(rtaustar) + '\n')
                    jout.write(directorid  + '\t' + str(taustar) + '\t' + str(rc_tau) + '\t' + str(len(x)) + '\n' )
                    jjout.write(directorid + '\t' + str(rtaustar) + '\t' + str(corrand) + '\t' + str(len(x)) + '\n' )



     
fout.close()
gout.close()
jout.close()

ggout.close()
jjout.close()

## source /opt/virtualenv-python2.7/bin/activate




