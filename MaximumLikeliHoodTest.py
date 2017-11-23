
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory
from scipy.optimize import minimize
import scipy.stats as stats
import time
import random






def fit2():


    def lik(parameters):

        m = parameters[0]
        b = parameters[1]
        sigma = parameters[2]
        c = parameters[3]
        for i in np.arange(0, len(x)):
            y_exp = m * x + b
        L = c + (len(x)/2 * np.log(2 * np.pi) + len(x)/2 * np.log(sigma ** 2) + 1 /
             (2 * sigma ** 2) * sum(( y_exp) ** 2))
        return L



    bnds = ((-10000.0, 2.0), (-10000.0, 10000.0) ,(-10000.0, 10000.0), (0.0, 2.0))




    x = np.array([1,2,3,4,5])
    xy = np.array([2,5,8,11,14])
    lik_model = minimize(lik, np.array([1.0,1.0,1.0 ,1.0]), method='TNC', bounds=bnds)
     
    print lik_model

    #plt.scatter(x,y)
    #plt.plot(x, lik_model['x'][0] * x + lik_model['x'][1])
    #plt.show()








    


   
def opt(N):


    def LogL_i(parameters):


        #careers = [[3.0,  4.5, 7.0,  11.3, 8.65, 23.1, 3.3, 5.9, 12.4, 3.3],
        #          [8.0,  3.5, 31.0,  1.3, 40.65, 13.1, 5.3, 10.9, 17.4, 40.4],
        #          [12.0, 8.5, 9.0,   1.3, 2.65, 33.1, 1.3, 5.9, 22.4, 3.4]]

        logL_i = 0

        #for c_i in careers:


        files = os.listdir('Data/Film/film-director-simple-careers')
        norm_factors   = {}


        ijk = 0

        for filename in files:

            individuals_career = SimpleCareerTrajectory(filename, 'Data/Film/film-director-simple-careers/' + filename, 0, [], False, 0, 0, 9999)  
            c_i = individuals_career.getImpactValues()


            if len(c_i) > 1:


                sum_ci = sum([math.log(c) for c in c_i])

                mu_p = parameters[0]
                mu_Q = parameters[1]
                mu_N = parameters[2]

                sigma_p = parameters[3]
                sigma_Q = parameters[4]
                sigma_N = parameters[5]

                sigma_pQ = parameters[6]
                sigma_pN = parameters[7]
                sigma_QN = parameters[8]

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


                K = sigma_p2*sigma_Q2*sigma_N2 + 2*sigma_pQ*sigma_QN*sigma_pN - sigma_pN2*sigma_Q2 - sigma_QN2 * sigma_p2 - sigma_pQ2*sigma_N2 

                A_i = inv_expN / nev * sigma_N2 + expN/K * ( nev - 2*(sigma_pN*sigma_QN - sigma_pQ * sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2 )

                B_i = inv_expN/nev * (-2.0*mu_Q*sigma_N2 - 2.0 * sigma_QN*(N_i - mu_N)) + expN/K * (-2.0 * (sum_ci/expN - mu_p)  * nev  + 2.0 * (sum_ci/expN - mu_p + mu_Q) * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) - 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2) * (N_i - mu_N)   - 2.0 * mu_Q * (sigma_p2*sigma_N2 - sigma_pN2)+ 2.0 * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) )                    
                 
                C_i = inv_expN/nev * (sigma_N2*mu_Q**2 + 2.0 * sigma_QN*mu_Q*(N_i - mu_N) + sigma_Q2 * (N_i - mu_N)**2)  + expN/K * (  nev* (sum_ci**2/expN + mu_p**2 - 2.0*mu_p*sum_ci )  - 2.0 * mu_Q * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) * (sum_ci/expN - mu_p) + 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2)*(N_i - mu_N)*(sum_ci/expN - mu_p) + mu_Q**2*(sigma_p**2*sigma_N**2 - sigma_pN2) - 2.0 * mu_Q * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) + (sigma_p2*sigma_Q2 - sigma_pQ2)*(N_i - mu_N)**2 )

                D_i = (2.0*pi)**(1.0 + 0.5 * expN) * (nev) ** (inv_expN/2.0) * abs(K)**(expN/2.0)


                A_i = A_i/2.0
                B_i = B_i / 2.0
                C_i = C_i / 2.0

     

                try:
                    logL_i +=  (B_i**2/(4*A_i)) - C_i + math.log(1.0/D_i * math.sqrt(pi/A_i))
                except:
                    return -np.inf
     
            
        return logL_i






    params      = ['mu_p',      'mu_Q',      'mu_N',      'sigma_p',      'sigma_Q',      'sigma_N',      'sigma_pQ',      'sigma_pN',      'sigma_QN']
    params_dict = {'mu_p' : [], 'mu_Q' : [], 'mu_N' : [], 'sigma_p' : [], 'sigma_Q' : [], 'sigma_N' : [], 'sigma_pQ' : [], 'sigma_pN' : [], 'sigma_QN' : []}
    

    for i in range(N):
        
        mu_p = 1.2 + random.random()
        mu_Q = 2.4 + random.random()
        mu_N = 12.3 + random.random()

        sigma_p = 0.95 + random.random()/10.0
        sigma_Q = 0.9  + random.random()/10.0
        sigma_N = 0.96 + random.random()/10.0

        sigma_pQ = 0.02 + random.random()/10.0
        sigma_pN = 0.04 + random.random()/10.0
        sigma_QN = 0.07 + random.random()/10.0

        initial_guess = [mu_p, mu_Q, mu_N, sigma_p, sigma_Q, sigma_N, sigma_pQ, sigma_pN, sigma_QN]


        bnds = (( 0.0, 10.0  ),    # mu_p
                ( 0.0, 10.0  ),    # mu_Q
                ( 1.0, 100.0 ),    # mu_N
                ( 0.0, 1.0   ),    # sigma_p
                ( 0.0, 1.0   ),    # sigma_Q
                ( 0.0, 1.0   ),    # sigma_N
                (-1.0, 1.0   ),    # sigma_pQ
                (-1.0, 1.0   ),    # sigma_pN
                (-1.0, 1.0   ))    # sigma_QN



        x = np.array([1,2,3,4,5])
        xy = np.array([2,5,8,11,14])
        model = minimize(LogL_i, np.array(initial_guess), method='L-BFGS-B', bounds=bnds)

        for i in range(9):
            params_dict[params[i]].append(model.x[i])
        
    return params_dict



params = opt(10)

for k, v in sorted(params.items()):
    print k, ':\t', np.mean(v), ' +/- ' , np.std(v)




'''
def LogL_i(c_i):

    c_i = [3.0, 4.5, 7.0, 11.3, 8.65, 23.1, 3.3, 5.9, 12.4, 53.4]
    sum_ci = sum([math.log(c) for c in c_i])

    mu_p = 0.2
    mu_Q = 0.4
    mu_N = 3.3

    sigma_p = 0.95
    sigma_Q = 0.9
    sigma_N = 0.8

    sigma_pQ = 0.12
    sigma_pN = 0.13
    sigma_QN = 0.09

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




    #K = (sigma_p**2 * sigma_Q**2 * sigma_N**2)  +  (2* sigma_pQ * sigma_QN * sigma_pN) - (sigma_pN**2 * sigma_Q2 + sigma_QN2 * sigma_p2 + sigma_pQ**2*sigma_N2 )

    #A_i = (1- expN)/(sigma_Q2 * sigma_N2 - sigma_QN2) * sigma_N2 + expN/K * ( sigma_Q2*sigma_N2 - sigma_QN2  - 2*(sigma_pN*sigma_QN - sigma_pQ*sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2     )

    #B_i = inv_expN / nev * (-2.0*mu_Q*sigma_N2 - 2.0 * sigma_QN*(N_i - mu_N)) + expN/K * (-2.0 * (sum_ci/expN - mu_p) * nev  + 2.0 * (sum_ci/expN - mu_p + mu_Q) * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) - 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2) * (N_i - mu_N)  - 2.0 * mu_Q * (sigma_p2*sigma_N2 - sigma_pN2)  + 2.0* (sigma_pN*sigma_pQ - sigma_p2*sigma_QN) * (N_i - mu_N)  )
     
    #C_i = inv_expN/nev * (  sigma_N2*mu_Q**2 + 2.0*sigma_QN*mu_Q*(N_i-mu_N) + sigma_Q2 * (N_i - mu_N)**2 ) + expN/K * (    nev*(sum_ci**2/expN + mu_p**2 - 2.0*mu_p * sum_ci) - 2.0 * mu_Q *(sigma_pN*sigma_QN - sigma_pQ*sigma_N2) * (sum_ci/expN - mu_p)  + 2.0*(sigma_pQ*sigma_QN - sigma_pN*sigma_Q2) * (N_i - mu_N) * (sum_ci/expN - mu_p) + mu_Q**2 * (sigma_p2*sigma_N2 - sigma_pN2)  - 2.0 * mu_Q * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN) * (N_i - mu_N) + (sigma_p2*sigma_Q2 - sigma_pQ2) * (N_i - mu_N)**2)

    #D_i = (2.0*math.pi)**(1.0+1.0/2.0 * expN)* (sigma_Q2*sigma_N2 - sigma_QN2)**(1.0/2-1.0/2 * expN) * abs(K)**(expN/2.0)

    #logL_i2 = (B_i**2/(4.0*A_i) - C_i) + math.log(1.0/D_i * math.sqrt(pi/A_i))




    K = sigma_p2*sigma_Q2*sigma_N2 + 2*sigma_pQ*sigma_QN*sigma_pN - sigma_pN2*sigma_Q2 - sigma_QN2 * sigma_p2 - sigma_pQ2*sigma_N2 


    A_i = inv_expN / nev * sigma_N2 + expN/K * ( nev - 2*(sigma_pN*sigma_QN - sigma_pQ * sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2 )


    B_i = inv_expN/nev * (-2.0*mu_Q*sigma_N2 - 2.0 * sigma_QN*(N_i - mu_N)) + expN/K * (-2.0 * (sum_ci/expN - mu_p)  * nev  + 2.0 * (sum_ci/expN - mu_p + mu_Q) * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) - 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2) * (N_i - mu_N)   - 2.0 * mu_Q * (sigma_p2*sigma_N2 - sigma_pN2)+ 2.0 * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) )                    
     

    C_i = inv_expN/nev * (sigma_N2*mu_Q**2 + 2.0 * sigma_QN*mu_Q*(N_i - mu_N) + sigma_Q2 * (N_i - mu_N)**2)  + expN/K * (  nev* (sum_ci**2/expN + mu_p**2 - 2.0*mu_p*sum_ci )  - 2.0 * mu_Q * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) * (sum_ci/expN - mu_p) + 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2)*(N_i - mu_N)*(sum_ci/expN - mu_p) + mu_Q**2*(sigma_p**2*sigma_N**2 - sigma_pN2) - 2.0 * mu_Q * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) + (sigma_p2*sigma_Q2 - sigma_pQ2)*(N_i - mu_N)**2 )


    D_i = (2.0*pi)**(1.0 + 0.5 * expN) * (nev) ** (inv_expN/2.0) * abs(K)**(expN/2.0)

    A_i = A_i/2.0
    B_i = B_i / 2.0
    C_i = C_i / 2.0



    logL_i =  (B_i**2/(4*A_i)) - C_i + math.log(1.0/D_i * math.sqrt(pi/A_i))






def CareerMLE():



    files = os.listdir('Data/Film/film-director-simple-careers2')
    norm_factors   = {}


    SumLogL = 0

    for filename in files:

        individuals_career = SimpleCareerTrajectory(filename, 'Data/Film/film-director-simple-careers2/' + filename, 0, [], False, 0, 0, 9999)  
        print individuals_career.getImpactValues()








def MLEexample():
 
    x = np.array([1,2,3,4,5])
    y = np.array([2,5,8,11,14])
    lik_model = minimize(lik, np.array([1,1,1]), method='L-BFGS-B')



MLEexample()


'''






