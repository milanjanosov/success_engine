import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import sys





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


def get_logD_i(pi, expN, nev, inv_expN, K):

    return - 1 * (  (1.0 + 0.5 * expN) * math.log(2.0*pi)    +    (inv_expN/2.0) * math.log(nev)    +   expN/2.0 * math.log(abs(K))     )  



def get_L_i(pi, A_i, B_i, C_i, logD_i):

    #return - 1.0 * (    (B_i**2 / ( 4*A_i ) ) - C_i + math.log(1.0/D_i * math.sqrt(pi/A_i))     )
    return - 1.0 * (    (B_i**2 / ( 4*A_i ) ) - C_i +  logD_i  +   math.log( math.sqrt(pi/A_i))     )



def get_sumI():


    I = []
    N = []
    sumI = []
    files = os.listdir('career_data')      
    for filename in files:
        c_i = [float(line.strip().split('\t')[1]) for line in open('career_data/' + filename)]
        
        if len(c_i) != 0:
            N.append(len(c_i))
            for c in c_i:   
                I.append(math.log(c))

            sumI.append(sum([math.log(c) for c in c_i]))        


    return sumI, N





def lik(mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN):




    sigma_Q2 = sigma_Q**2
    sigma_N2 = sigma_N**2
    sigma_p2 = sigma_p**2

    sigma_QN2 = sigma_QN**2
    sigma_pN2 = sigma_pN**2
    sigma_pQ2 = sigma_pQ**2
    
    nev     = sigma_Q2*sigma_N2 - sigma_QN2
    sumI, N = get_sumI() 

 

    logL = 0

    for index, sum_ci in enumerate(sumI):


        sum_ci2  = sum_ci**2
        N_i      = math.log(N[index])
        pi       = math.pi
        expN     = math.exp(N_i)
        inv_expN = 1.0 - math.exp(N_i)

        K = get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
        
        A_i = get_A_i(expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2 )
        B_i = get_B_i(sum_ci, sum_ci2, N_i, mu_N, mu_p, mu_Q, N, expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
        C_i = get_C_i(sigma_p, sigma_Q, sigma_N, sum_ci, sum_ci2, N_i, mu_N, mu_p, mu_Q, N, expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
        logD_i = get_D_i(pi, expN, nev, inv_expN, K)


        logL -= get_L_i(pi, A_i, B_i, C_i, logD_i)



    return logL




print lik(1.1, 1.01, 0.4,1.4, 1.6, 1.8, 0.02, 0.01, 0.01)




