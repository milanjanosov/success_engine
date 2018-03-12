

import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import scipy.stats as stats







''' PROBLEM:

    - x     ->     9 parameters
    - i     ->     1, ... 100k, individuals
    - L_i   ->     L_i(A_i(x), B_i(x), C_i(x), D_i(x))
    - L     ->     sum_i L_i 
            -->    this it to minimize



    # http://www.robertasinatra.com/pdf/Science_quantifying_Supplementary.pdf
    
'''






def logL(x, y, y_exp, sigma):

    LogL = 0

    LogL += (len(x)/2 * np.log(2 * np.pi) + len(x)/2 * np.log(sigma ** 2) + 1 / (2 * sigma ** 2) * sum((y - y_exp) ** 2))

    return LogL 





def get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2 ):

    return sigma_p2 * sigma_Q2 * sigma_N2 + 2 * sigma_pQ * sigma_QN * sigma_pN - sigma_pN2 * sigma_Q2 - sigma_QN2 * sigma_p2 - sigma_pQ2*sigma_N2 


def get_A_i(expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2 ):

    return ( inv_expN / nev * sigma_N2 + expN/K * ( nev - 2*(sigma_pN*sigma_QN - sigma_pQ * sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2 ) ) / 2.0


def get_B_i(sum_ci, sum_ci2, N_i, mu_N, mu_p, mu_Q, N, expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):

    return ( inv_expN/nev * (-2.0*mu_Q*sigma_N2 - 2.0 * sigma_QN*(N_i - mu_N)) + expN/K * (-2.0 * (sum_ci/expN - mu_p)  * nev  + 2.0 * (sum_ci/expN - mu_p + mu_Q) * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) - 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2) * (N_i - mu_N)   - 2.0 * mu_Q * (sigma_p2*sigma_N2 - sigma_pN2)+ 2.0 * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) ) ) / 2.0  


def get_C_i(sigma_p, sigma_Q, sigma_N, sum_ci, sum_ci2, N_i, mu_N, mu_p, mu_Q, N, expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):

    return ( inv_expN/nev * (sigma_N2*mu_Q**2 + 2.0 * sigma_QN*mu_Q*(N_i - mu_N) + sigma_Q2 * (N_i - mu_N)**2)  + expN/K * (  nev* (sum_ci2/expN + mu_p**2 - 2.0*mu_p*sum_ci )  - 2.0 * mu_Q * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) * (sum_ci/expN - mu_p) + 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2)*(N_i - mu_N)*(sum_ci/expN - mu_p) + mu_Q**2*(sigma_p**2*sigma_N**2 - sigma_pN2) - 2.0 * mu_Q * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) + (sigma_p2*sigma_Q2 - sigma_pQ2)*(N_i - mu_N)**2 ) ) / 2.0


def get_D_i(pi, expN, nev, inv_expN, K):

    return (2.0*pi)**(1.0 + 0.5 * expN) * (nev) ** (inv_expN/2.0) * abs(K)**(expN/2.0)


def get_L_i(pi, A_i, B_i, C_i, D_i):

    return - 1.0 * (    (B_i**2 / ( 4*A_i ) ) - C_i + math.log(1.0/D_i * math.sqrt(pi/A_i))     )






def logL2(sum_c, sum_c2, N, mu_p, mu_Q, mu_N):


    sigma_p = 1.1
    sigma_Q = 1.3
    sigma_N = 1.5

    sigma_pQ = 1.01
    sigma_pN = 1.015
    sigma_QN = 1.05

    sigma_Q2 = sigma_Q**2
    sigma_N2 = sigma_N**2
    sigma_p2 = sigma_p**2

    sigma_QN2 = sigma_QN**2
    sigma_pN2 = sigma_pN**2
    sigma_pQ2 = sigma_pQ**2
    
    nev = sigma_Q2*sigma_N2 - sigma_QN2



    logL = 0

    for index, sum_ci in enumerate(sum_c):

 
        sum_ci2  = sum_c2[index]
        N_i      = N[index]
        pi       = math.pi
        expN     = math.exp(N_i)
        inv_expN = 1.0 - math.exp(N_i)


        K = get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
        
        A_i = get_A_i(expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2 )
        B_i = get_B_i(sum_ci, sum_ci2, N_i, mu_N, mu_p, mu_Q, N, expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
        C_i = get_C_i(sigma_p, sigma_Q, sigma_N, sum_ci, sum_ci2, N_i, mu_N, mu_p, mu_Q, N, expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
        D_i = get_D_i(pi, expN, nev, inv_expN, K)

 
        logL -= get_L_i(pi, A_i, B_i, C_i, D_i)



    return logL


   
def lik(parameters, *args):

    sum_c   = args[0][0]
    sum_c2  = args[0][1]
    N       = args[0][2]


    
    mu_p = parameters[0]
    mu_Q = parameters[1]
    mu_N = parameters[2]




    #y_exp = [mu_p * sum_c[i] + mu_Q for i in np.arange(0, len(sum_c))]
    #L = logL(sum_c, sum_c2, y_exp, mu_N)

    L = logL2(sum_c, sum_c2, N, mu_p, mu_Q, mu_N)


    return L




def optimize():



    ''' load the career data: '''
    ''' sum_c and sum_c2 contain the sum of the impacts and the square sum of the impacts, while N contains the productivities '''
    
    files = os.listdir('careerstrajectorie')

    sum_c   = []
    sum_c2  = []
    N       = []

    for filename in files[0:100]:

        c_i = [float(line.strip().split('\t')[1]) for line in open('careerstrajectorie/' + filename)]
        if len(c_i) > 4:
            sum_c   .append(sum([math.log(c) for c in c_i]))
            sum_c2  .append(sum([(math.log(c))**2 for c in c_i]))
            N       .append(math.log(len(c_i)))
       

    sum_c  = np.array(sum_c)
    sum_c2 = np.array(sum_c2)


    ''' run the minimizer '''
    limit = 10000000000000000.0
    bnds  = [(0.0, limit), (0.0, limit), (0.0, limit)]

    lik_model = minimize(lik, np.array([1.0,1.0,1.0]), args=([sum_c, sum_c2, N] , ), method='SLSQP', bounds = bnds)


    ''' plot results like the one for the regression '''

    print lik_model
    plt.scatter(sum_c,sum_c2)
    plt.plot(sum_c, lik_model['x'][0] * sum_c + lik_model['x'][1])
    plt.show()

    



if __name__ == '__main__':         


    optimize()




