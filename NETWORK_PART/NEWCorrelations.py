import numpy as np
import os

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

        zavg = np.mean(z)



        z = [zz/zavg for zz in z]
        yavg = np.mean(y)

        y = [yy/yavg for yy in y]



        a = list(y)
        random.shuffle(a)

        
        return y, z, spearmanr(y, z)[0], spearmanr(a, z)[0]





def shifted_correl(ts1,ts2):
    
    cmaxabs  = 0
    cmax     = 1111.1    
    x        = []
    y        = []
    shiftm   = 0
    
    
    for shift in range(int(0.5*len(ts2))):
       
        ts1_s = []
        ts2_s = []
    
        for i in range(len(ts1) - shift):

            ts1_s.append(ts1[i])
            ts2_s.append(ts2[i-shift])
         
        c = spearmanr(ts1_s, ts2_s)[0]
        
        if abs(c) > cmaxabs:
            cmaxabs = abs(c)
            cmax = c
            x =  list(ts1_s)
            y =  list(ts2_s)
            shiftm = shift
            


            

    for shift in range(-int(0.5*len(ts2)), 0):

        ts1_s = []
        ts2_s = []
        
        for i in range(len(ts1) + shift):

            ts1_s.append(ts1[i])
            ts2_s.append(ts2[i-shift])
         
        c = spearmanr(ts1_s, ts2_s)[0]
        
        if abs(c) > cmaxabs:
            cmaxabs = abs(c)
            cmax = c
            x = list(ts1_s)
            y = list(ts2_s)           
            shiftm = shift
            
  
        
     
        
    return x, y, cmax, shiftm



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

    for line in open('NEWTemporal/4_directors_centralities/' +dirid+'.dat'):
        
        fields = line.strip().split('\t')
        fields = [float(f) for f in fields]
        centralities[fields[0]] = fields[column]
        
    return centralities, measures[column-1]


#_QEVER

fout = open('NEWTemporal/1_career_centrality_correlations.dat', 'w')
gout = open('NEWTemporal/2_shiftwindow_sizes.dat', 'w')
fout.write('directorid\tcareer_length\tc_random\tc_original\tc_shifted\tc_dtw\n')

directors = [aaa.replace('.dat', '') for aaa in os.listdir('NEWTemporal/4_directors_centralities')]

for ind, directorid in enumerate(directors):

    centralities, meas  = get_centralities(directorid, measures, column = 1)

    print ind
    results = get_plot_correl(centralities, directorid, '', meas, logimpact = True)

    if results:
 
        x,   y,   c0, c_rand  = results
        x_s, y_s, c_s, shiftm = shifted_correl(x,y)
        x_d, y_d, cdtw = dtw_timeserires(x,y)

        fout.write(directorid + '\t' + str(len(x)) + '\t' + str(c_rand) + '\t' + str(c0) + '\t' + str(c_s) + '\t' + str(cdtw) + '\n')
        gout.write( str(shiftm) + '\n')


fout.close()
gout.close()







