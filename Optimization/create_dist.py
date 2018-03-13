import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


files = os.listdir('career_data')

#fout = open('impacts.dat', 'w')


I = []
N = []


        
for filename in files:

    c_i = [float(line.strip().split('\t')[1]) for line in open('career_data/' + filename)]


    N.append(len(c_i))
    for c in c_i:
        
        I.append(c)



'''   cov = np.array([[1.0,  0.05,  0.05,],
                    [0.05, 2.0,   0.05,],
                    [0.05, 0.05,  3.0]])


    mu  = [3.1, 20.2, 6.5]
'''


#  3.0    6.5      
#sigma_p, mu_p, sigma_Q, mu_Q
#2	5	1	21.5




#  1.625	4.2083333333	0.5833333333	23


'''

for sigma_p in np.arange(1, 3, 0.25):

    for mu_p in np.arange(3, 6, 0.5):


        for sigma_Q in np.arange(0, 2, 0.5):

            for mu_Q in np.arange(18, 25, 0.5):

                ps = [np.exp(pp) for pp in list(np.random.normal(mu_p, sigma_p, len(I)))]


                Qs = [np.exp(pp) for pp in list(np.random.normal(mu_Q, sigma_Q, len(N)))]



                Isynt = []
                nnn = len(N)

                for index, n in enumerate(N):



                    for i in range(n):
                        
                        Q = Qs[index]
                        p = ps[0]#np.random.choice(ps, 1)[0]
                        ps.remove(p)

                        impact = Q * p
                        Isynt.append(impact)




                print  sigma_p, mu_p, sigma_Q, mu_Q, stats.ks_2samp(np.cumsum(I), np.cumsum(Isynt))[0]
'''










ps = [np.exp(pp) for pp in list(np.random.normal(20.2,  1,len(I)))]   		#  	 	2 	  	 		0.0546403893358



Qs = [np.exp(pp) for pp in list(np.random.normal( 6.5 , 1 , len(N)))]


Isynt = []
nnn = len(N)


for index, n in enumerate(N):



    for i in range(n):
        
        Q = Qs[index]
        p = ps[0]#np.random.choice(ps, 1)[0]
        ps.remove(p)

        impact = Q * p
        Isynt.append(impact)




print  stats.ks_2samp(np.cumsum(I), np.cumsum(Isynt))[0]
print KL(np.cumsum(I), np.cumsum(Isynt))


plt.yscale('log')
plt.hist(I, bins = 40, color = 'b', alpha = 0.5)
plt.hist(Isynt, bins = 40, color = 'r', alpha = 0.5)
plt.show()



'''

3.625220546
19.5405405565
5.67637379148

'''








'''
mu_p =
mu_Q =


p, Q = obs = np.random.normal(1,4,50000)
ini = [0,1]
print(estimation(obs,lambda ob,p:norm.logpdf(ob,p[0],p[1]),ini))


print len(I), len(N)

'''

