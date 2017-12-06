import numpy as np
import math
from scipy.optimize import minimize
import scipy.stats as stats
import random



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
'''                DEFINE CONSTRAINTS              '''
''' -----------------------------------------------'''

# the constraint A_i > 0 for all i
'''def get_cons_a(expN, sigma_p, sigma_Q, sigma_N, sigma_pQ, sigma_QN, sigma_pN ):


    expN    = args[0]
    
    print args
    sigma_p  = args[1]
    sigma_Q  = args[2]
    sigma_N  = args[3]
    sigma_pQ = args[4]
    sigma_QN = args[5]
    sigma_pN = args[6]
    



    sigma_Q2 = sigma_Q**2
    sigma_N2 = sigma_N**2
    sigma_p2 = sigma_p**2

    sigma_QN2 = sigma_QN**2
    sigma_pN2 = sigma_pN**2
    sigma_pQ2 = sigma_pQ**2


    K   = get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2) 
    nev = get_nev(sigma_Q2, sigma_N2, sigma_QN2)
    inv_expN = 1.0 - expN

    return ((inv_expN / nev * sigma_N2 + expN/K * ( nev - 2*(sigma_pN*sigma_QN - sigma_pQ * sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2 ))/2.0)
'''

'''
# the constraint for D_i > 0 for all i
def get_cons_d(args):


    expN    = args[0]
    sigma_p  = args[1]
    sigma_Q  = args[2]
    sigma_N  = args[3]
    sigma_pQ = args[4]
    sigma_QN = args[5]
    sigma_pN = args[6]


    min_A = 1

    sigma_Q2 = sigma_Q**2
    sigma_N2 = sigma_N**2
    sigma_p2 = sigma_p**2

    sigma_QN2 = sigma_QN**2
    sigma_pN2 = sigma_pN**2
    sigma_pQ2 = sigma_pQ**2


    K   = get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2) 
    nev = get_nev(sigma_Q2, sigma_N2, sigma_QN2)
    inv_expN = 1.0 - expN

    return (2.0*math.pi)**(1.0 + 0.5 * expN) * (nev) ** (inv_expN/2.0) * abs(K)**(expN/2.0)
'''







def get_cons_aa(*args):



    expNs    = args[1][0]
    sigma_p  = args[1][1]
    sigma_Q  = args[1][2]
    sigma_N  = args[1][3]
    sigma_pQ = args[1][4]
    sigma_QN = args[1][5]
    sigma_pN = args[1][6]

    min_A = 1

    sigma_Q2 = sigma_Q**2
    sigma_N2 = sigma_N**2
    sigma_p2 = sigma_p**2

    sigma_QN2 = sigma_QN**2
    sigma_pN2 = sigma_pN**2
    sigma_pQ2 = sigma_pQ**2


    K   = get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
    
    for expN in expNs:    
        nev = get_nev(sigma_Q2, sigma_N2, sigma_QN2)
        inv_expN = 1.0 - expN
        A_i = (inv_expN / nev * sigma_N2 + expN/K * ( nev - 2*(sigma_pN*sigma_QN - sigma_pQ * sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2 ))/2.0
        if A_i < min_A:
            min_A = A_i

    print 'c', min_A

    return min_A

 
 
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
        D_i = get_D(expN, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)

        #print A_i, D_i


        logL_i -= logL(A_i, B_i, C_i, D_i)
                  



    return logL_i




''' ----------------------------------------------'''
'''                    MINIMIZING                 '''
''' ----------------------------------------------''' 
 

# parameters
N = 5

f_max = 0
x_max = []

initmax = 100

for i in range(N):

    # variables to minimize
    mu_p = 1#random.uniform(0.0, initmax)
    mu_Q = 2#random.uniform(0.0, initmax )
    mu_N = 3#random.uniform(0.0, initmax )

    sigma_p = 0.1#random.uniform(0.0, initmax)
    sigma_Q = 0.2#random.uniform(0.0, initmax)
    sigma_N = 0.3#random.uniform(0.0, initmax)

    sigma_pQ = 0.01#random.uniform(-initmax, initmax)
    sigma_pN = 0.02#random.uniform(-initmax, initmax)
    sigma_QN = 0.03#random.uniform(-initmax, initmax)


    # dummmy careers
    c = [[6.6, 7.1, 7.5, 7.6, 7.5, 8.2, 8.5],
         [7.3, 7.4, 6.3, 8.3, 5.8, 6.2, 5.5, 7.1, 6.5, 7.2, 9.3, 7.9, 7.1, 6.7],
         [6.8, 6.6, 6.2, 8.3, 5.8, 7.4],
         [9.4, 8.7, 8.4, 7.7, 8.5, 7.1, 6.9, 8.2, 8.5, 8.1, 5.3, 8.6, 7.4, 9.1]]


    # stat data
    N = len(c)    
    sums_ci  = [sum([math.log(cc) for cc in c_i])       for c_i in c ]
    sums_ci2 = [sum([(math.log(cc))**2 for cc in c_i])  for c_i in c ]  
    N_is     = [math.log(len(c_i))                      for c_i in c ] 
    expNs    = [len(c_i)                           for c_i in c]
  
    print expNs

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


    # constraints
    #cns = ({'type' : 'ineq',  'fun' : get_cons_aa})  #, 'args': ([expN, sigma_p, sigma_Q, sigma_N, sigma_pQ, sigma_pN, sigma_QN],)})   
    cns = ({'type' : 'ineq',  'fun' : get_cons_aa, 'args': ([expNs, sigma_p, sigma_Q, sigma_N, sigma_pQ, sigma_pN, sigma_QN],)})
           #{'type' : 'ineq',  'fun' : get_cons_d})
   

    # the minimizer
    model = minimize(LogL_i,  np.array(initial_guess), 
                args    = ([N, sums_ci, sums_ci2, N_is, expNs], ), 
                method  = 'SLSQP', 
                bounds  = bnds ,
                constraints = cns,
                options = {'maxiter' : 100, 'eps' : 0.010})


    # the results
    print  model, '\n', mu_p, model.x[0], '\n'




