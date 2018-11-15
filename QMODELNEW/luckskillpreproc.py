import os
import sys
import math
import numpy as np


def get_stat(data):

    var  = np.var(data)
    mean = np.mean(data)

    return str(var) + '\t' + str(mean)


ps_fields = {}
Qs_fields = {}

logps_fields = {}
logQs_fields = {}


fields = ['authors', 'art_director' , 'writer', 'director', 'composer', 'producer', 'pop', 'rock', 'classical', 'jazz', 'electro', 'funk', 'folk', 'hiphop']
Ns     = [50,  20, 10, 10, 10, 10, 80, 80, 80, 80, 80, 80, 80 ,80]


fields = ['director','jazz']


skillz = {}

for ind, field in enumerate(fields):
    
    N = str(Ns[ind])

    print field, '\t', ind, '/', len(fields)

    #ps = [float(line.strip().split('\t')[-1]) for line in open('pQData/p_distribution_'+field+'-'+N+'_0.dat')]
    #Qs = [float(line.strip().split('\t')[-1]) for line in open('pQData/Q_distribution_'+field+'-'+N+'_0.dat')]
    
    ps = [float(line.strip().split('\t')[-1]) for line in open('pQData_rescaled/p_distribution_'+field + '_0.dat')]
    Qs = [float(line.strip().split('\t')[-1]) for line in open('pQData_rescaled/Q_distribution_'+field + '_0.dat')]
   


    ps_fields[field] = ps
    Qs_fields[field] = Qs
    
    logps_fields[field] = [math.log(p) for p in ps if p > 0.0]
    logQs_fields[field] = [math.log(q) for q in Qs if q > 0.0]







fields = ['mathematics', 'physics', 'health_science', 'zoology', 'agronomy', 'psychology', 'environmental_science', 
          'engineering', 'theoretical_computer_science', 'applied_physics', 'space_science_or_astronomy', 'chemistry', 
          'political_science', 'biology', 'geology']


fields = ['mathematics', 'psychology']




for ind, field in enumerate(fields):
    

    print field, '\t', ind, '/', len(fields)

    Qs = [float(line.strip().split('\t')[-1]) for line in open('pQData_rescaled/Q_distribution_'+field+'_0.dat')]
    ps = [float(line.strip().split('\t')[-1]) for line in open('pQData_rescaled/p_distribution_'+field+'_0.dat')]
   
    ps_fields[field] = ps
    Qs_fields[field] = Qs
    
    logps_fields[field] = [math.log(p) for p in ps if p > 0.0]
    logQs_fields[field] = [math.log(q) for q in Qs if q > 0.0]





skillfile = open('DataToPlot_rescaled/5_LuckSkill/art_sci_vars.dat', 'w')
skillfile.write( 'field\tp_var\tp_mean\tQ_var\tQ_mean\tlogp_var\tlogp_mean\tlogQ_var\tlogQ_mean\n')


for field in logps_fields.keys():

    ps    = ps_fields[field]
    Qs    = Qs_fields[field]
    logps = logps_fields[field] 
    logQs = logQs_fields[field]

    skillfile.write( field + '\t' + get_stat(ps) + '\t' + get_stat(Qs) + '\t' + get_stat(logps) + '\t' + get_stat(logQs) + '\n')


skillfile.close()


