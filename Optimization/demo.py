import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.optimize import minimize
import scipy.stats as stats
import sys


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False




def rosen(x, *args):
    
    I   = args[0][0]
    N   = args[0][1]
    ijk = args[0][2]
    
    ps = [np.exp(pp) for pp in list(np.random.normal(x[0],  1, len(I)))]
    Qs = [np.exp(pp) for pp in list(np.random.normal(x[1],  1, len(N)))]
    
    Isynt = []

    for index, n in enumerate(N):
        for i in range(n):
            Q = Qs[index]
            p = ps[0]
            ps.remove(p)
            impact = Q * p
            Isynt.append(impact)

            
    S = stats.ks_2samp(np.cumsum(I), np.cumsum(Isynt))[0]

    if S < 1.0: 
    #if S < 0.02:
        print ijk, '\t', x[0], '\t', x[1], '\t', S
    
    return S

 
    #return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)





def stage1():

    ''' data '''

    I = []
    N = []
    files = os.listdir('career_data')      
    for filename in files:
        c_i = [float(line.strip().split('\t')[1]) for line in open('career_data/' + filename)]
        N.append(len(c_i))
        for c in c_i:   
            I.append(c)


    ''' optimizer '''



    #for ijk in range(100):

        #mu_p0 = 40*random.random()
        #mu_Q0 = 40*random.random()
      
    for mu_Q0 in np.arange(0,40,2):

        for mu_p0 in np.arange(0,40,2):

          
            x0 = np.array([mu_p0, mu_Q0])#, 1.0, 6.5, 1.0])
            res = minimize(rosen, x0,  args=([I, N, 0] , ),  method='Nelder-Mead', options={'xtol': 1e-1, 'disp': True, 'maxiter' : 2})







def stage2():



    ''' input params '''
  
    mu = []

    for line in open('out.dat'):

        fields = [float(fff) for fff in line.strip().split('\t') if is_number(fff)]

        if len(fields) == 4:
            
            if fields[3] < 0.2:
            
                mu.append(( fields[1], fields[2]))



    ''' data '''

    I = []
    N = []
    files = os.listdir('career_data')      
    for filename in files:
        c_i = [float(line.strip().split('\t')[1]) for line in open('career_data/' + filename)]
        N.append(len(c_i))
        for c in c_i:   
            I.append(c)


    ''' optimizer '''

    for mu_p, mu_Q in mu:


        x0 = np.array([mu_p, mu_Q])#, 1.0, 6.5, 1.0])
        res = minimize(rosen, x0,  args=([I, N, 0] , ),  method='Nelder-Mead', options={'xtol': 1e-1, 'disp': True, 'maxiter' : 2})





if __name__ == '__main__':  

    if sys.argv[1] == 'stage1':
        stage1()
    elif sys.argv[1] == 'stage2':
        stage2()











