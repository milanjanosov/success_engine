import os
import sys
import numpy.random as nr
import numpy as np
from scipy.stats.stats import pearsonr   
import math


def gen():


    if not os.path.exists('career_data'):
        os.makedirs('career_data')


    # N, Q, p


    cov3 = np.array([ [1.0, 0.5,  0.8,],
                      [0.5, 1.0,  0.8,],
                      [0.8, 0.8,  1.0 ]])

    cov2 = np.array([[1.0, 0.5],
                     [0.5, 1.0]])



    mu2  = [3.1, 20.2]
    mu3  = [3.1, 20.2, 6.5]







    
    Num = 10000

    logN, logQ         = zip(*np.random.multivariate_normal(mu2, cov2, Num))
    N                  = [int(np.exp(n)) for n in logN]
    logN2, logQ2, logp = zip(*np.random.multivariate_normal(mu3, cov3, sum(N)))
    



    print 'mu_N', mu2[0], '\t', np.mean(logN)
    print 'mu_Q', mu3[1], '\t', np.mean(logQ)
    print 'mu_p', mu3[2], '\t', np.mean(logp)

    print 'sigma_N', cov2[0,0], '\t', np.std(logN)**2
    print 'sigma_Q', cov2[1,1], '\t', np.std(logQ)**2
    print 'sigma_p', cov3[2,2], '\t', np.std(logp)**2

    print 'sigma_NQ', cov3[0,1], '\t', pearsonr(logN,  logQ)[0]
    print 'sigma_pQ', cov3[1,2], '\t', pearsonr(logQ2, logp)[0]
    print 'sigma_pN', cov3[0,2], '\t', pearsonr(logN2, logp)[0]


    print len(logQ), len(logQ2), len(logp)



    p = [math.exp(pp) for pp in logp]
    
    for index, n in enumerate(N):

        fout2 = open('career_data/' + str(index) + '_career_impacts.dat', 'w')

        for i in range(n):
            
            Q = math.exp(logQ[index])
            pp = p[0]
            p.remove(pp)

            impact = Q * pp

            fout2.write(str(i) + '\t' + str(impact) + '\n')

        fout2.close()


   



def test():


    careers = os.listdir('careers')

    Ns = []
    Qs = []
    ps = []

    
    for c in careers:

        for line in open('careers/' + c):

            fields = line.strip().split('\t')
            N      = float(fields[0])
            Q      = float(fields[1])
            impact = float(fields[2])
            p      = impact/Q


            ps.append(p)

        #Qs.append(math.log(Q))
        #Ns.append(math.log(N))
        
      
        Qs.append(Q)
        Ns.append(N)
              


    print math.log(np.mean(Ns))
    print math.log(np.mean(Qs))
    print math.log(np.mean(ps))
    print pearsonr(Qs, Ns)[0]



if __name__ == '__main__':  

    if sys.argv[1] == 'gen':
        gen()
    elif sys.argv[1] == 'test':
        test()












