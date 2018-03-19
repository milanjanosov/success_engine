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





'''


mu_N 2.1 	2.13177426619
mu_Q 5.2 	5.23806551897
mu_p 2.5 	2.50297436238
sigma_N 4.0 	3.80484247699
sigma_Q 3.0 	3.08320546509
sigma_p 2.0 	1.99614001753
sigma_NQ 0.05 	0.0431036532926
sigma_pQ 0.02 	0.00736569736911
sigma_pN 0.02 	-0.000952501660668


'''


def dist(I, Isynt):

    S = 0

    for index, i in enumerate(I):
        
        S += (np.log(i) - np.log(Isynt[index]))**2

    return S

        


	


ps = [np.exp(pp) for pp in list(np.random.normal(1.40625 ,  1,len(I)))]   		#  	 	2 	  	 		0.0546403893358


Qs = [np.exp(pp) for pp in list(np.random.normal( 6.5140625 , 1 , len(N)))]


Isynt = []
nnn = len(N)


for index, n in enumerate(N):



    for i in range(n):
        
        Q = Qs[index]
        p = ps[0]#np.random.choice(ps, 1)[0]
        ps.remove(p)

        impact = Q * p
        Isynt.append(impact)



print dist(I, Isynt)
print  stats.ks_2samp(np.cumsum(I), np.cumsum(Isynt))[0]



plt.yscale('log')
plt.hist(I, bins = 200, color = 'b', alpha = 0.5)
plt.hist(Isynt, bins = 200, color = 'r', alpha = 0.5)
plt.show()


