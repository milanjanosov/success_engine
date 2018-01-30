import math
import random
import numpy as np
import sys
import os
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory

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
def logL_i(A_i, B_i, C_i, D_i):

    return (B_i**2 / ( 4*A_i ) ) - C_i  + math.log(1.0/D_i * math.sqrt(math.pi/A_i))  



def LogL(parameters, N, sums_ci, sums_ci2, N_is, expNs):

    logL = 0

    mu_p = parameters[0]
    mu_Q = parameters[1]
    mu_N = parameters[2]

    sigma_p = 1.1  #parameters[3]
    sigma_Q = 1.2  #parameters[4]
    sigma_N = 1.13 #parameters[5]

    sigma_pQ = 0.3 #parameters[6]
    sigma_pN = 0.2 #parameters[7]
    sigma_QN = 0.3 #parameters[8]

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
        D_i = get_D(expN, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)




        logL += logL_i(A_i, B_i, C_i, D_i)
                  



    return logL




''' ----------------------------------------------'''
'''                    MINIMIZING                 '''
''' ----------------------------------------------''' 
 

# parameters


f_max = 0
x_max = []

initmax = 100



# variables to minimize
#mu_p = random.uniform(0.0, initmax)
#mu_Q = random.uniform(0.0, initmax )
#mu_N = random.uniform(0.0, initmax )



# dummmy careers



'''c = [[6.6, 7.1, 7.5, 7.6, 7.5, 8.2, 8.5],
     [7.3, 7.4, 6.3, 8.3, 5.8, 6.2, 5.5, 7.1, 6.5, 7.2, 9.3, 7.9, 7.1, 6.7],
     [6.8, 6.6, 6.2, 8.3, 5.8, 7.4],
     [9.4, 8.7, 8.4, 7.7, 8.5, 7.1, 6.9, 8.2, 8.5, 8.1, 5.3, 8.6, 7.4, 9.1]]
'''

career = 'director'
files = os.listdir('Data/Film/film-' + career + '-simple-careers')
c = []
for filename in files:
    individuals_career = SimpleCareerTrajectory(filename, 'Data/Film/film-' + career + '-simple-careers/' + filename, 1, [], False, 0, 0, 9999)  
    c_i = individuals_career.getImpactValues()
#    print c_i
    if len(c_i) > 4:
        c.append(c_i)

print len(c)



# stat data
N = len(c)    
sums_ci  = [sum([math.log(cc) for cc in c_i])       for c_i in c ]
sums_ci2 = [sum([(math.log(cc))**2 for cc in c_i])  for c_i in c ]  
N_is     = [math.log(len(c_i))                      for c_i in c ] 
expNs    = [math.exp(N_i)                           for N_i in N_is]


maxr = float(sys.argv[1])
step = float(sys.argv[2])

r = np.arange(0.0, maxr, step)

maxLogL = 0
max_x = (0,0,0)

n   = len(r)  
ijk = 0

folder = 'bruteforce_output'
if not os.path.exists(folder): 
    os.makedirs(folder)  

f = open('bruteforce_output/formula_data_' + str(maxr) + '_' + str(step)  + '.dat', 'w')

for mu_p in r:
    #print ijk, '/', n
    ijk += 1        
    for mu_Q in r:
        for mu_N in r:

            parameters = [mu_p, mu_Q, mu_N]


            LogLvalue = LogL(parameters, N, sums_ci, sums_ci2, N_is, expNs)
            f.write(str(mu_p) + '\t' + str(mu_Q) + '\t' + str(mu_N) + '\t' + str(LogLvalue) + '\n')
            if LogLvalue > maxLogL:
                maxLogL = LogLvalue
                max_x = (mu_p, mu_Q, mu_N)

f.close()





  
