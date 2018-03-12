import os
import sys
import numpy.random as nr
import numpy as np
from scipy.stats.stats import pearsonr   
import math


def gen():

    cov = np.array([[1.0,  0.05,  0.05,],
                    [0.05, 2.0,   0.05,],
                    [0.05, 0.05,  3.0]])


    mu  = [3.1, 20.2, 6.5]

    p1 = []
    p2 = []
    p3 = []
    m1 = []
    m2 = []
    m3 = []

    Ns = []
    Qs = []
    ps = []

    for i in range(4000):

        x = np.random.multivariate_normal(mu, cov, 10)

        #x = np.exp(x)



        N = [x[i][0] for i in range(10)]
        Q = [x[i][1] for i in range(10)]
        p = [x[i][2] for i in range(10)]


        Ns += N
        Qs += Q
        ps += p

        p1.append( pearsonr(N,Q)[0])
        p2.append( pearsonr(N,p)[0])
        p3.append( pearsonr(Q,p)[0])
        m1.append( np.mean(N))
        m2.append( np.mean(Q))
        m3.append( np.mean(p))


    print pearsonr(Ns,Qs)
    measured_Ns = [int(np.exp(n)) for n in Ns[0:200]]
    measured_Qs = [np.exp(q) for q in Qs[0:200]]

    total_N     =  sum(measured_Ns) 
    measured_ps = [np.exp(p) for p in ps]

    print np.mean([np.log(float(b)) for b in measured_Qs])
    #print pearsonr([math.log(n) for n in measured_Ns],[math.log(q) for q in measured_Qs])




    for index, N in enumerate(measured_Ns):

        fout = open('careers/' + str(index) + '_career.dat', 'w')
        fout2 = open('career_data/' + str(index) + '_career_impacts.dat', 'w')

        for i in range(N):
            
            Q = measured_Qs[index]
            p = np.random.choice(measured_ps, 1)[0]
            measured_ps.remove(p)

            impact = Q * p


            fout.write(str(N) + '\t' + str(Q) + '\t' + str(impact) + '\n')
            fout2.write(str(i) + '\t' + str(impact) + '\n')
    

        fout.close()
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












