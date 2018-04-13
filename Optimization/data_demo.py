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

 


def get_I(field):


    files = os.listdir('DataSample/Film/film-' + field + '-simple-careers')

    I = []
    N = []
           
    for filename in files:

        c_i = []

        for line in gzip.open('DataSample/Film/film-' + field + '-simple-careers/' + filename):
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





def rosen2(x, *args):


    
    I   = args[0][0]
    N   = args[0][1]
    ijk = args[0][2]

    if x[2] > 0 and x[3] > 0:
    
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
        #S = dist(I, Isynt)

        #if S < 1.0: 
        #if S < 0.02:


        fout = open(filename, 'a')
        fout.write( str(x[0]) + '\t' + str(x[1]) + '\t' + str(x[2]) + '\t' + str(x[3]) + '\t' + str(S) + '\n')
        fout.close()


        return S







def stage3(*args):


    inputs   = args[0]
    I        = args[1]
    N        = args[2]
    filename = args[3]




    ''' optimizer '''
    for (mu_Q0, mu_p0, sigma_Q0, sigma_p0) in inputs:
          
        x0 = np.array([ mu_Q0, mu_p0, sigma_Q0, sigma_p0])#, 1.0, 6.5, 1.0])
    
        #print mu_Q0, mu_p0, sigma_Q0, sigma_p0

        res = minimize(rosen2, x0,  args=([I, N, 0] , ),  method='Nelder-Mead', options={'xtol': 1e-1, 'disp': True, 'maxiter' : 20})


   




if __name__ == '__main__':  





    if sys.argv[1] == 'stage1':
        I, N = get_I('art-director')
        stage1(I, N)
    elif sys.argv[1] == 'stage2':
        I, N = get_I('art-director')
        stage2(I, N)
    elif sys.argv[1] == 'stage3':

        folder = 'fasz'
        if not os.path.exists(folder):
            os.makedirs(folder)



        for i in range(10):
    

            fields = ['director', 'producer', 'writer', 'composer', 'art-director']

            for field in fields:


                I, N  = get_I(field)

                mu_Q_min = 0.1
                mu_Q_max = 6.1
                mu_p_min = 0.1
                mu_p_max = 6.1

                sigma_Q_min = 0.01
                sigma_Q_max = 0.6
                sigma_p_min = 0.01
                sigma_p_max = 0.6


                nnn = 15

                nmu_Q     = (mu_Q_max    - mu_Q_min)    / 15.0
                nmu_p     = (mu_p_max    - mu_p_min)    / 15.0
                nsigma_Q  = (sigma_Q_max - sigma_Q_min) / 15.0
                nsigma_p  = (sigma_p_max - sigma_p_min) / 15.0

            
                string = '_'.join([str(aa) for aa in [mu_Q_min, mu_Q_max, mu_p_min, mu_p_max, sigma_Q_min, sigma_Q_max, sigma_p_min, sigma_p_max]])


                filename = 'fasz/opt_res_out_' + field + '_' + string

                lfiles = len([fff.split('--')[0] for fff in os.listdir('fasz') if field in fff])

                filename = filename + '--' + str(lfiles/2)
                
                fout = open(filename, 'w')
                fout.close()


                #print nmu_Q, nmu_p


                


                inputs = []

      
                for mu_Q0 in np.arange( mu_Q_min, mu_Q_max, nmu_Q ):

                    for mu_p0 in np.arange( mu_p_min, mu_p_max, nmu_p ):

                        for sigma_Q0 in np.arange( sigma_Q_min, sigma_Q_max, nsigma_Q):

                            for sigma_p0 in np.arange( sigma_p_min, sigma_p_max, nsigma_p ):

                                inputs.append((mu_Q0, mu_p0, sigma_Q0, sigma_p0))

                '''

                muQs    = np.random.uniform(mu_Q_min,    mu_Q_max,    nnn)
                mups    = np.random.uniform(mu_p_min,    mu_p_max,    nnn)
                sigmaQs = np.random.uniform(sigma_Q_min, sigma_Q_max, nnn)
                sigmaps = np.random.uniform(sigma_p_min, sigma_p_max, nnn)


                for mu_Q0 in muQs:

                    for mu_p0 in mups:

                        for sigma_Q0 in sigmaQs:

                            for sigma_p0 in sigmaps:

                                inputs.append((mu_Q0, mu_p0, sigma_Q0, sigma_p0))
                '''






               
                num_threads = 40
                inputs_chunks = chunkIt(inputs, num_threads)
                Pros = []

                for i in range(num_threads):
                    p = Process(target = stage3, args = (inputs_chunks[i], I, N, filename, ))
                    Pros.append(p)
                    p.start()
                   
                for t in Pros:
                    t.join()

                
                

                results = []

                for line in open( filename):
            
                    value = float(line.strip().split('\t')[-1])

                    results.append(( value, line))



                sorted_results = sorted(results, key=lambda tup: tup[0])

                f = open( filename + '_sorted', 'w')

                for v, k in sorted_results[0:10000]:
                    f.write(k)

                f.close()
                





