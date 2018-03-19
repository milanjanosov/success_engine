import os
import numpy as np
import math
import random
from scipy.optimize import minimize
import scipy.stats as stats
import sys
from multiprocessing import Process




def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



def dist(I, Isynt):

    S = 0

    for index, i in enumerate(I):
        
        S += (np.log(i) - np.log(Isynt[index]))**2

    return S



def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out



def rosen(x, *args):
    
    I   = args[0][0]
    N   = args[0][1]
    ijk = args[0][2]
    
    Qs = [np.exp(pp) for pp in list(np.random.normal(x[0],1, len(N)))]
    ps = [np.exp(pp) for pp in list(np.random.normal(x[1],1, len(I)))]

    
    Isynt = []

    for index, n in enumerate(N):
        for i in range(n):
            Q = Qs[index]
            p = ps[0]
            ps.remove(p)
            impact = Q * p
            Isynt.append(impact)

            
    #S = stats.ks_2samp(np.cumsum(I), np.cumsum(Isynt))[0]
    S = dist(I, Isynt)


    #if S < 1.0: 
    #if S < 0.02:
    print x[1], '\t', x[0], '\t', S
    
    return S

 
    #return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)



'''
mu_N 2.1 	2.13961975032
mu_Q 5.2 	5.23015861122
mu_p 2.5 	2.498606304
sigma_N 1.0 	1.03779120062
sigma_Q 1.0 	1.04111404623
sigma_p 1.0 	0.991256968701
sigma_NQ 0.05 	0.219626934701
sigma_pQ 0.02 	0.0131432351906
sigma_pN 0.02 	0.0204936942337

'''




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
      
    for mu_Q0 in np.arange(0.5,8,1):

        for mu_p0 in np.arange(0.5,8,1):
           
            x0 = np.array([mu_p0, mu_Q0])#, 1.0, 6.5, 1.0])
            res = minimize(rosen, x0,  args=([I, N, 0] , ),  method='Nelder-Mead', options={'xtol': 0.1,'disp': True, 'maxiter' : 10})







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
        res = minimize(rosen, x0,  args=([I, N, 0] , ),  method='Nelder-Mead', options={'xtol': 1e-1, 'disp': True, 'maxiter' : 20})










def rosen2(x, *args):


    
    I   = args[0][0]
    N   = args[0][1]
    ijk = args[0][2]
    
    Qs = [np.exp(pp) for pp in list(np.random.normal(x[0],  x[2], len(N)))]
    ps = [np.exp(pp) for pp in list(np.random.normal(x[1],  x[3], len(I)))]

    
    Isynt = []

    for index, n in enumerate(N):
        for i in range(n):
            Q = Qs[index]
            p = ps[0]
            ps.remove(p)
            impact = Q * p
            Isynt.append(impact)

            
    S = stats.ks_2samp(np.cumsum(I), np.cumsum(Isynt))[0]

    #if S < 1.0: 
    #if S < 0.02:

    fout = open('fasz/opt_res_out', 'a')
    fout.write( str(x[0]) + '\t' + str(x[1]) + '\t' + str(x[2]) + '\t' + str(x[3]) + '\t' + str(S) + '\n')
    fout.close()


    return S




'''

mu_N 2.1 	2.1722860517
mu_Q 5.2 	5.15758550742
mu_p 2.5 	2.49764458854
sigma_N 4.0 	4.11661853773
sigma_Q 3.0 	2.81875337551
sigma_p 2.0 	2.0007206235
sigma_NQ 0.05 	0.0268890373378
sigma_pQ 0.02 	0.00904585788318
sigma_pN 0.02 	0.00845189029387




'''


def stage3(inputs):

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
    for (mu_Q0, mu_p0, sigma_Q0, sigma_p0) in inputs:
          
        x0 = np.array([ mu_Q0, mu_p0, sigma_Q0, sigma_p0])#, 1.0, 6.5, 1.0])
    
        res = minimize(rosen2, x0,  args=([I, N, 0] , ),  method='Nelder-Mead', options={'xtol': 1e-1, 'disp': True, 'maxiter' : 10})


   




if __name__ == '__main__':  

    if sys.argv[1] == 'stage1':
        stage1()
    elif sys.argv[1] == 'stage2':
        stage2()
    elif sys.argv[1] == 'stage3':

        folder = 'fasz'
        if not os.path.exists(folder):
            os.makedirs(folder)

        fout = open('fasz/opt_res_out', 'w')
        fout.close()





        inputs = []

        for mu_Q0 in np.arange(0.3,8,1):

            for mu_p0 in np.arange(0.3,8,1):

                for sigma_Q0 in np.arange(0.3,8,1):

                    for sigma_p0 in np.arange(0.3,8,1):

                        inputs.append((mu_Q0, mu_p0, sigma_Q0, sigma_p0))


        num_threads = 40
        inputs_chunks = chunkIt(inputs, num_threads)
        Pros = []

        for i in range(num_threads):
            p = Process(target = stage3, args = (inputs_chunks[i], ))
            Pros.append(p)
            p.start()
           
        for t in Pros:
            t.join()












