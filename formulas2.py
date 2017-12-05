'''


import math


mu_p = 2.1  #''random.uniform(0.0, initmax)
mu_Q = 1.5  #''random.uniform(0.0, initmax)
mu_N = 1.1  #''random.uniform(0.0, initmax)

sigma_p = 0.1  #''random.uniform(0.0, initmax)
sigma_Q = 2.1  #''random.uniform(0.0, initmax)
sigma_N = 4.1  #''random.uniform(0.0, initmax)

sigma_pQ = 2.1  #''random.uniform(-1.0 * initmax, initmax)
sigma_pN = 0.1  #''random.uniform(-1.0 * initmax, initmax)
sigma_QN = 1.4


c = [[6.6, 7.1, 7.5, 7.6, 7.5, 8.2, 8.5],
     [7.3, 7.4, 6.3, 8.3, 5.8, 6.2, 5.5, 7.1, 6.5, 7.2, 9.3, 7.9, 7.1, 6.7],
     [6.8, 6.6, 6.2, 8.3, 5.8, 7.4],
     [9.4, 8.7, 8.4, 7.7, 8.5, 7.1, 6.9, 8.2, 8.5, 8.1, 5.3, 8.6, 7.4, 9.1]]


for c_i in c[0:1]:

    sum_ci  = sum([math.log(c) for c in c_i])
    sum_ci2 = sum([(math.log(c))**2 for c in c_i])  


    N_i = math.log(len(c_i))
    pi  = math.pi

    expN = math.exp(N_i)
    inv_expN = 1.0 - math.exp(N_i)

    sigma_Q2 = sigma_Q**2
    sigma_N2 = sigma_N**2
    sigma_p2 = sigma_p**2

    sigma_QN2 = sigma_QN**2
    sigma_pN2 = sigma_pN**2
    sigma_pQ2 = sigma_pQ**2

    nev = sigma_Q2*sigma_N2 - sigma_QN2


    K   = sigma_p2 * sigma_Q2 * sigma_N2 + 2 * sigma_pQ*sigma_QN*sigma_pN - sigma_pN2*sigma_Q2 - sigma_QN2 * sigma_p2 - sigma_pQ2*sigma_N2 




    A_i = inv_expN / nev * sigma_N2 + expN/K * ( nev - 2*(sigma_pN*sigma_QN - sigma_pQ * sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2 )

    B_i = inv_expN/nev * (-2.0*mu_Q*sigma_N2 - 2.0 * sigma_QN*(N_i - mu_N)) + expN/K * (-2.0 * (sum_ci/expN - mu_p)  * nev  + 2.0 * (sum_ci/expN - mu_p + mu_Q) * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) - 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2) * (N_i - mu_N)   - 2.0 * mu_Q * (sigma_p2*sigma_N2 - sigma_pN2)+ 2.0 * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) )                    

    C_i = inv_expN/nev * (sigma_N2*mu_Q**2 + 2.0 * sigma_QN*mu_Q*(N_i - mu_N) + sigma_Q2 * (N_i - mu_N)**2)  + expN/K * (  nev* (sum_ci2/expN + mu_p**2 - 2.0*mu_p*sum_ci )  - 2.0 * mu_Q * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) * (sum_ci/expN - mu_p) + 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2)*(N_i - mu_N)*(sum_ci/expN - mu_p) + mu_Q**2*(sigma_p**2*sigma_N**2 - sigma_pN2) - 2.0 * mu_Q * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) + (sigma_p2*sigma_Q2 - sigma_pQ2)*(N_i - mu_N)**2 )

    D_i = (2.0*pi)**(1.0 + 0.5 * expN) * (nev) ** (inv_expN/2.0) * abs(K)**(expN/2.0)


    A_i = A_i / 2.0
    B_i = B_i / 2.0
    C_i = C_i / 2.0



    print K
    print A_i
    print B_i
    print C_i
    print D_i



'''


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def lik(parameters):
    m = parameters[0]
    b = parameters[1]
    sigma = parameters[2]
    for i in np.arange(0, len(x)):
        y_exp = m * x + b
    L = (len(x)/2 * np.log(2 * np.pi) + len(x)/2 * np.log(sigma ** 2) + 1 /
         (2 * sigma ** 2) * sum((y - y_exp) ** 2))
    return L

x = np.array([1,2,3,4,5])
y = np.array([2,5,8,11,14])
lik_model = minimize(lik, np.array([1,1,1]), method='L-BFGS-B')
plt.scatter(x,y)
plt.plot(x, lik_model['x'][0] * x + lik_model['x'][1])
plt.show()












