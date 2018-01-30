import numpy as np
import math
from scipy.optimize import minimize
import scipy.stats as stats
import random
import os
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory
import sys


''' -----------------------------------------------'''
'''            DEFINE VARIABLE FUNCTIONS           '''
''' -----------------------------------------------'''

# a helper variable
def get_nev(sigma_Q2, sigma_N2, sigma_QN2):   
    return  sigma_Q2*sigma_N2 - sigma_QN2


# calculate the K determinant
def get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):   

    return sigma_p2 * sigma_Q2 * sigma_N2 + 2 * sigma_pQ*sigma_QN*sigma_pN - sigma_pN2*sigma_Q2 - sigma_QN2 * sigma_p2 - sigma_pQ2*sigma_N2


# calc the A_is
def get_A(expN, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):

    K   = get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
    nev = get_nev(sigma_Q2, sigma_N2, sigma_QN2)
    inv_expN = 1.0 - expN

    return (inv_expN / nev * sigma_N2 + expN/K * ( nev - 2*(sigma_pN*sigma_QN - sigma_pQ * sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2 ))/2.0


# calc the B_is
def get_B(sum_ci, mu_p, mu_Q, mu_N, N_i, expN, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):    

    K   = get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
    nev = get_nev(sigma_Q2, sigma_N2, sigma_QN2)
    inv_expN = 1.0 - expN

    return (inv_expN/nev * (-2.0*mu_Q*sigma_N2 - 2.0 * sigma_QN*(N_i - mu_N)) + expN/K * (-2.0 * (sum_ci/expN - mu_p)  * nev  + 2.0 * (sum_ci/expN - mu_p + mu_Q) * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) - 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2) * (N_i - mu_N)   - 2.0 * mu_Q * (sigma_p2*sigma_N2 - sigma_pN2)+ 2.0 * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) ) )/2.0                   


# calc the C_is
def get_C(sum_ci2, sum_ci, mu_p, mu_Q, mu_N, N_i, expN, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):   
 
    K   = get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
    nev = get_nev(sigma_Q2, sigma_N2, sigma_QN2)
    inv_expN = 1.0 - expN
 
    return (inv_expN/nev * (sigma_N2*mu_Q**2 + 2.0 * sigma_QN*mu_Q*(N_i - mu_N) + sigma_Q2 * (N_i - mu_N)**2)  + expN/K * (  nev* (sum_ci2/expN + mu_p**2 - 2.0*mu_p*sum_ci )  - 2.0 * mu_Q * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) * (sum_ci/expN - mu_p) + 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2)*(N_i - mu_N)*(sum_ci/expN - mu_p) + mu_Q**2*(sigma_p2*sigma_N2 - sigma_pN2) - 2.0 * mu_Q * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) + (sigma_p2*sigma_Q2 - sigma_pQ2)*(N_i - mu_N)**2 ))/2.0


#calc the C_is
def get_D(expN, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2):

    K   = get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
    nev = get_nev(sigma_Q2, sigma_N2, sigma_QN2)
    inv_expN = 1.0 - expN

    return  (2.0*math.pi)**(1.0 + 0.5 * expN) * (nev) ** (inv_expN/2.0) * abs(K)**(expN/2.0)



 
 
''' -----------------------------------------------'''
'''             THE LIKELIHOOD FUNCTION            '''
''' -----------------------------------------------''' 
 
# the likelihood function
def logL(A_i, B_i, C_i, D_i):

    return (B_i**2 / ( 4*A_i ) ) - C_i  + math.log(1.0/D_i * math.sqrt(math.pi/A_i))  



def LogL_i(parameters, args):

    logL_i = 0

    N        = args[0]
    sums_ci  = args[1]
    sums_ci2 = args[2]
    N_is     = args[3]
    expNs    = args[4]
    

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


    for i in range(N):

        sum_ci  = sums_ci[i]
        sum_ci2 = sums_ci2[i]
        N_i     = N_is[i]
        expN    = expNs[i]
   
        A_i = get_A(expN, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
        B_i = get_B(sum_ci, mu_p, mu_Q, mu_N, N_i, expN, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
        C_i = get_C(sum_ci2, sum_ci, mu_p, mu_Q, mu_N, N_i, expN, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
        
        try:
            D_i = float(get_D(expN, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2))
            logL_i -= logL(A_i, B_i, C_i, D_i)
        except:
            pass          



    return logL_i




''' ----------------------------------------------'''
'''                    MINIMIZING                 '''
''' ----------------------------------------------''' 
 

# parameters
NNN = 25

f_max = 0
x_max = []

initmax = float(sys.argv[1])


f = open('MLE/MLE_params_director_' + str(initmax) + '_' + str(NNN) + '.dat', 'w') 
header = ['mu_p', 'mu_Q', 'mu_N', 'sigma_p', 'sigma_Q', 'sigma_N', 'sigma_pQ', 'sigma_pN', 'sigma_QN', ]
header = [h + '_init' for h in header] + header 
f.write('initmax' + '\t' + '\t'.join(header) + '\n')


for i in range(NNN):

    print i, '/', NNN
    # variables to minimize
    mu_p = random.uniform(0.0, initmax)
    mu_Q = random.uniform(0.0, initmax )
    mu_N = random.uniform(0.0, initmax )

    sigma_p = random.uniform(0.0, initmax)
    sigma_Q = random.uniform(0.0, initmax)
    sigma_N = random.uniform(0.0, initmax)

    sigma_pQ = random.uniform(-initmax, initmax)
    sigma_pN = random.uniform(-initmax, initmax)
    sigma_QN = random.uniform(-initmax, initmax)


    # careers
    career = 'director'
    files = os.listdir('Data/Film/film-' + career + '-simple-careers')

    print 'files: ', len(files)
    c = []
    for filename in files:
        individuals_career = SimpleCareerTrajectory(filename, 'Data/Film/film-' + career + '-simple-careers/' + filename, 1, [], False, 0, 0, 9999)  
        c_i = individuals_career.getImpactValues()
        if len(c_i) > 14:

            c.append(c_i)

    print 'Careers: ', len(c)


    # stat data
    N = len(c)    
    sums_ci  = [sum([math.log(cc) for cc in c_i])       for c_i in c ]
    sums_ci2 = [sum([(math.log(cc))**2 for cc in c_i])  for c_i in c ]  
    N_is     = [math.log(len(c_i))                      for c_i in c ] 
    expNs    = [len(c_i)                           for c_i in c]
  


    # init guess
    initial_guess = [mu_p, mu_Q, mu_N, sigma_p, sigma_Q, sigma_N, sigma_pQ, sigma_pN, sigma_QN]


    # boundari conditions
    bnds = (( 0.0, initmax ),      # mu_p
            ( 0.0, initmax ),      # mu_Q
            ( 0.0, initmax ),      # mu_N
            ( 0.0, initmax ),      # sigma_p
            ( 0.0, initmax ),      # sigma_Q
            ( 0.0, initmax ),      # sigma_N
            ( -initmax, initmax ), # sigma_pQ
            ( -initmax, initmax ), # sigma_pN
            ( -initmax, initmax )) # sigma_QN


    # the minimizer
    model = minimize(LogL_i,  np.array(initial_guess), 
                args    = ([N, sums_ci, sums_ci2, N_is, expNs], ), 
                method  = 'SLSQP', 
                bounds  = bnds ,
                options = {'maxiter' : 100, 'eps' : 0.010})


    inits   = '\t'.join([str(ini) for ini in initial_guess])
    results = '\t'.join([str(xx)  for xx  in model.x])
 
   
    f.write(str(initmax) + '\t' + inits + '\t' + results + '\n')


f.close()




