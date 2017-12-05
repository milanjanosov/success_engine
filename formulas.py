import numpy as np
import math
from scipy.optimize import minimize
import scipy.stats as stats
import random




def get_nev(sigma_Q2, sigma_N2, sigma_QN2):   
    return  sigma_Q2*sigma_N2 - sigma_QN2


def get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):   
    return sigma_p2 * sigma_Q2 * sigma_N2 + 2 * sigma_pQ*sigma_QN*sigma_pN - sigma_pN2*sigma_Q2 - sigma_QN2 * sigma_p2 - sigma_pQ2*sigma_N2


def get_A(K, expN, inv_expN, nev, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):
    return (inv_expN / nev * sigma_N2 + expN/K * ( nev - 2*(sigma_pN*sigma_QN - sigma_pQ * sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2 ))/2.0


def get_B(sum_ci, mu_p, mu_Q, mu_N, N_i, K, expN, inv_expN, nev, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):    
    return (inv_expN/nev * (-2.0*mu_Q*sigma_N2 - 2.0 * sigma_QN*(N_i - mu_N)) + expN/K * (-2.0 * (sum_ci/expN - mu_p)  * nev  + 2.0 * (sum_ci/expN - mu_p + mu_Q) * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) - 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2) * (N_i - mu_N)   - 2.0 * mu_Q * (sigma_p2*sigma_N2 - sigma_pN2)+ 2.0 * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) ) )/2.0                   


def get_C(sum_ci2, sum_ci, mu_p, mu_Q, mu_N, N_i, K, expN, inv_expN, nev, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):   
    return (inv_expN/nev * (sigma_N2*mu_Q**2 + 2.0 * sigma_QN*mu_Q*(N_i - mu_N) + sigma_Q2 * (N_i - mu_N)**2)  + expN/K * (  nev* (sum_ci2/expN + mu_p**2 - 2.0*mu_p*sum_ci )  - 2.0 * mu_Q * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) * (sum_ci/expN - mu_p) + 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2)*(N_i - mu_N)*(sum_ci/expN - mu_p) + mu_Q**2*(sigma_p2*sigma_N2 - sigma_pN2) - 2.0 * mu_Q * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) + (sigma_p2*sigma_Q2 - sigma_pQ2)*(N_i - mu_N)**2 ))/2.0


def get_D(pi, expN, nev, inv_expN, K):
    return  (2.0*pi)**(1.0 + 0.5 * expN) * (nev) ** (inv_expN/2.0) * abs(K)**(expN/2.0)



def LogL_i(parameters):


    logL_i = 0

    mu_p = parameters[0]
    mu_Q = parameters[1]
    mu_N = parameters[2]

    sigma_p = parameters[3]
    sigma_Q = parameters[4]
    sigma_N = parameters[5]

    sigma_pQ = parameters[6]
    sigma_pN = parameters[7]
    sigma_QN = parameters[8]

    sigma_Q2 = sigma_Q**2
    sigma_N2 = sigma_N**2
    sigma_p2 = sigma_p**2

    sigma_QN2 = sigma_QN**2
    sigma_pN2 = sigma_pN**2
    sigma_pQ2 = sigma_pQ**2

    nev = sigma_Q2*sigma_N2 - sigma_QN2
    pi  = math.pi


    c = [[6.6, 7.1, 7.5, 7.6, 7.5, 8.2, 8.5],
         [7.3, 7.4, 6.3, 8.3, 5.8, 6.2, 5.5, 7.1, 6.5, 7.2, 9.3, 7.9, 7.1, 6.7],
         [6.8, 6.6, 6.2, 8.3, 5.8, 7.4],
         [9.4, 8.7, 8.4, 7.7, 8.5, 7.1, 6.9, 8.2, 8.5, 8.1, 5.3, 8.6, 7.4, 9.1]]


    for c_i in c:

        sum_ci  = sum([math.log(cc) for cc in c_i])
        sum_ci2 = sum([(math.log(cc))**2 for cc in c_i])  

        N_i = math.log(len(c_i))
        expN = math.exp(N_i)
        inv_expN = 1.0 - math.exp(N_i)

   

        '''
        K   = sigma_p2 * sigma_Q2 * sigma_N2 + 2 * sigma_pQ*sigma_QN*sigma_pN - sigma_pN2*sigma_Q2 - sigma_QN2 * sigma_p2 - sigma_pQ2*sigma_N2 

        A_i =   (math.sqrt( inv_expN / nev * sigma_N2 + expN/K * ( nev - 2*(sigma_pN*sigma_QN - sigma_pQ * sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2 ) ) )**2

  

        B_i = inv_expN/nev * (-2.0*mu_Q*sigma_N2 - 2.0 * sigma_QN*(N_i - mu_N)) + expN/K * (-2.0 * (sum_ci/expN - mu_p)  * nev  + 2.0 * (sum_ci/expN - mu_p + mu_Q) * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) - 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2) * (N_i - mu_N)   - 2.0 * mu_Q * (sigma_p2*sigma_N2 - sigma_pN2)+ 2.0 * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) )                    

        C_i = inv_expN/nev * (sigma_N2*mu_Q**2 + 2.0 * sigma_QN*mu_Q*(N_i - mu_N) + sigma_Q2 * (N_i - mu_N)**2)  + expN/K * (  nev* (sum_ci2/expN + mu_p**2 - 2.0*mu_p*sum_ci )  - 2.0 * mu_Q * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) * (sum_ci/expN - mu_p) + 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2)*(N_i - mu_N)*(sum_ci/expN - mu_p) + mu_Q**2*(sigma_p2*sigma_N2 - sigma_pN2) - 2.0 * mu_Q * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) + (sigma_p2*sigma_Q2 - sigma_pQ2)*(N_i - mu_N)**2 )

        D_i = (2.0*pi)**(1.0 + 0.5 * expN) * (nev) ** (inv_expN/2.0) * abs(K)**(expN/2.0)



       

        A_i = A_i / 2.0
        B_i = B_i / 2.0
        C_i = C_i / 2.0
        '''
           
        logL_i += - 1.0 * ( (B_i**2 / ( 4*A_i ) ) - C_i  + math.log(1.0/D_i * math.sqrt(pi/A_i))  )
                   



    return logL_i



params      = ['mu_p',      'mu_Q',      'mu_N',      'sigma_p',      'sigma_Q',      'sigma_N'    ,  'sigma_pQ',      'sigma_pN',      'sigma_QN']
params_dict = {'mu_p' : [], 'mu_Q' : [], 'mu_N' : [], 'sigma_p' : [], 'sigma_Q' : [], 'sigma_N' : [], 'sigma_pQ' : [], 'sigma_pN' : [], 'sigma_QN' : []}


N = 5

f_max = 0
x_max = []

initmax = 100

for i in range(N):




    mu_p = random.uniform(0.0, initmax)
    mu_Q = random.uniform(0.0, initmax )
    mu_N = random.uniform(0.0, initmax )

    sigma_p = random.uniform(0.0, initmax)
    sigma_Q = random.uniform(0.0, initmax)
    sigma_N = random.uniform(0.0, initmax)

    sigma_pQ = random.uniform(-initmax, initmax)
    sigma_pN = random.uniform(-initmax, initmax)
    sigma_QN = random.uniform(-initmax, initmax)




    initial_guess = [mu_p, mu_Q, mu_N, sigma_p, sigma_Q, sigma_N, sigma_pQ, sigma_pN, sigma_QN]

    model = minimize(LogL_i, np.array(initial_guess), method='BFGS', options = {'maxiter' : 50, 'eps' : 0.00010})


    print  model, '\n', mu_p, model.x[0], '\n'




