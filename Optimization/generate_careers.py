import numpy.random as nr
import numpy as np
from scipy.stats.stats import pearsonr   




cov = np.array([[1.0, 0.0, 0.8,],
                [0.0, 1.0, 0.0,],
                [0.8, 0.0, 1.0]])




mu  = [3.1, 2.2, 1.0]

p1 = []
p2 = []
p3 = []
m1 = []
m2 = []
m3 = []

Ns = []
Qs = []
ps = []

for i in range(2000):

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


print np.mean(Ns)
measured_Ns = [int(np.exp(n)) for n in np.random.choice(Ns, 100, False) if int(np.exp(n)) > 0]
measured_Qs = [np.exp(q) for q in np.random.choice(Qs, 100, False)]

total_N     =  sum(measured_Ns) 
measured_ps = [np.exp(q) for q in np.random.choice(ps, total_N, False)]

#print len(measured_Ns), sum(measured_Ns), np.mean(measured_Ns)
print np.mean([np.log(float(b)) for b in measured_Ns])
print measured_ps[0:10]

'''
print np.mean(p1)    
print np.mean(p2)
print np.mean(p3)
print np.exp(np.mean(m1))
print np.mean(m2)
print np.mean(m3)


'''






for index, N in enumerate(measured_Ns):

    fout = open('careers/' + str(index) + '_career.dat', 'w')


    for i in range(N):
        
        Q = measured_Qs[index]
        p = np.random.choice(measured_ps, 1)[0]
        measured_ps.remove(p)

        impact = Q * p

        print Q, p, impact

        fout.write(str(Q) + '\t' + str(impact) + '\n')

    fout.close()


















