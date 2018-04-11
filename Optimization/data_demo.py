import os
import numpy as np
import math
import random
from scipy.optimize import minimize
import scipy.stats as stats
import sys
from multiprocessing import Process
import gzip



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
    
    Qs = [np.exp(pp) for pp in list(np.random.normal(x[0], 2, len(N)))]
    ps = [np.exp(pp) for pp in list(np.random.normal(x[1], 2, len(I)))]

    
    Isynt = []

    for index, n in enumerate(N):
        for i in range(n):
            Q = Qs[index]
            p = ps[0]
            ps.remove(p)
            impact = Q * p
            Isynt.append(impact)
          
    S = dist(I, Isynt)

    print x[1], '\t', x[0], '\t', S
    
    return S

 


def get_I():


    files = os.listdir('DataSample/Film/film-art-director-simple-careers')

    I = []
    N = []
           
    for filename in files:

        c_i = []

        for line in gzip.open('DataSample/Film/film-art-director-simple-careers/' + filename):
            if 'rating_count' not in line:
                try:
                    ccc = float(line.strip().split('\t')[3])
                    if ccc > 0:
                        c_i.append(ccc)
                except:
                    pass


        N.append(len(c_i))
        for c in c_i:           
            I.append(c)


    return I, N




def stage1(I, N):

 
    ''' optimizer '''

    for mu_Q0 in np.arange(0.5,8,1):

        for mu_p0 in np.arange(0.5,8,1):
           
            x0  = np.array([mu_p0, mu_Q0])#, 1.0, 6.5, 1.0])
            res = minimize(rosen, x0,  args=([I, N, 0] , ),  method='Nelder-Mead', options={'xtol': 0.1,'disp': True, 'maxiter' : 10})







def stage2(I, N):



    ''' input params '''
  
    mu = []

    for line in open('out.dat'):

        fields = [float(fff) for fff in line.strip().split('\t') if is_number(fff)]

        if len(fields) == 4:
            
            if fields[3] < 0.2:
            
                mu.append(( fields[1], fields[2]))



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

            
    #S = stats.ks_2samp(np.cumsum(I), np.cumsum(Isynt))[0]
    S = dist(I, Isynt)

    #if S < 1.0: 
    #if S < 0.02:

    fout = open('fasz/opt_res_out2', 'a')
    fout.write( str(x[0]) + '\t' + str(x[1]) + '\t' + str(x[2]) + '\t' + str(x[3]) + '\t' + str(S) + '\n')
    fout.close()


    return S




'''

mu_N 2.1 	2.13491126007
mu_Q 5.2 	5.2178907034
mu_p 2.5 	2.49316451468
sigma_N 1.0 	0.947459198111
sigma_Q 3.0 	2.94723652079
sigma_p 1.0 	1.00710004916
sigma_NQ 0.05 	0.12297532998
sigma_pQ 0.02 	0.00382356487239
sigma_pN 0.02 	0.0211967298998




'''


def stage3(*args):


    inputs = args[0]
    I      = args[1]
    N      = args[2]


   


    ''' optimizer '''
    for (mu_Q0, mu_p0, sigma_Q0, sigma_p0) in inputs:
          
        x0 = np.array([ mu_Q0, mu_p0, sigma_Q0, sigma_p0])#, 1.0, 6.5, 1.0])
    
        res = minimize(rosen2, x0,  args=([I, N, 0] , ),  method='Nelder-Mead', options={'xtol': 1e-1, 'disp': True, 'maxiter' : 10})


   




if __name__ == '__main__':  


    I, N = get_I()


    if sys.argv[1] == 'stage1':
        stage1(I, N)
    elif sys.argv[1] == 'stage2':
        stage2(I, N)
    elif sys.argv[1] == 'stage3':

        folder = 'fasz'
        if not os.path.exists(folder):
            os.makedirs(folder)

        fout = open('fasz/opt_res_out2', 'w')
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
            p = Process(target = stage3, args = (inputs_chunks[i], I, N, ))
            Pros.append(p)
            p.start()
           
        for t in Pros:
            t.join()


        

        results = []

        for line in open('fasz/opt_res_out2'):
    
            value = float(line.strip().split('\t')[-1])

            results.append(( value, line))



        sorted_results = sorted(results, key=lambda tup: tup[0])

        f = open('fasz/opt_res_sorted.dat', 'w')

        for v, k in sorted_results[0:10000]:
            f.write(k)

        f.close()








