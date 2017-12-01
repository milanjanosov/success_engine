
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory
from scipy.optimize import minimize
import scipy.stats as stats
import time
import random
import sys



    


   
def opt(N, initmax, career):


    def LogL_i(parameters):


        logL_i = 0


        files = os.listdir('Data/Film/film-' + career + '-simple-careers')
        norm_factors   = {}


        ijk = 0

        for filename in files[0:100]:

            individuals_career = SimpleCareerTrajectory(filename, 'Data/Film/film-' + career + '-simple-careers/' + filename, 0, [], False, 0, 0, 9999)  
            c_i = individuals_career.getImpactValues()


            if len(c_i) > 4:

                sum_ci  = sum([math.log(c) for c in c_i])
                sum_ci2 = sum([(math.log(c))**2 for c in c_i])  
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


                K   = sigma_p2 * sigma_Q2 * sigma_N2 + 2 * sigma_pQ*sigma_QN*sigma_pN - sigma_pN2*sigma_Q2 - sigma_QN2 * sigma_p2 - sigma_pQ2*sigma_N2 

                A_i = inv_expN / nev * sigma_N2 + expN/K * ( nev - 2*(sigma_pN*sigma_QN - sigma_pQ * sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2 )

                B_i = inv_expN/nev * (-2.0*mu_Q*sigma_N2 - 2.0 * sigma_QN*(N_i - mu_N)) + expN/K * (-2.0 * (sum_ci/expN - mu_p)  * nev  + 2.0 * (sum_ci/expN - mu_p + mu_Q) * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) - 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2) * (N_i - mu_N)   - 2.0 * mu_Q * (sigma_p2*sigma_N2 - sigma_pN2)+ 2.0 * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) )                    
                 
                C_i = inv_expN/nev * (sigma_N2*mu_Q**2 + 2.0 * sigma_QN*mu_Q*(N_i - mu_N) + sigma_Q2 * (N_i - mu_N)**2)  + expN/K * (  nev* (sum_ci2/expN + mu_p**2 - 2.0*mu_p*sum_ci )  - 2.0 * mu_Q * (sigma_pN*sigma_QN - sigma_pQ*sigma_N2) * (sum_ci/expN - mu_p) + 2.0 * (sigma_pQ*sigma_QN - sigma_pN*sigma_Q2)*(N_i - mu_N)*(sum_ci/expN - mu_p) + mu_Q**2*(sigma_p**2*sigma_N**2 - sigma_pN2) - 2.0 * mu_Q * (sigma_pN*sigma_pQ - sigma_p2*sigma_QN)*(N_i - mu_N) + (sigma_p2*sigma_Q2 - sigma_pQ2)*(N_i - mu_N)**2 )

                D_i = (2.0*pi)**(1.0 + 0.5 * expN) * (nev) ** (inv_expN/2.0) * abs(K)**(expN/2.0)


                A_i = A_i / 2.0
                B_i = B_i / 2.0
                C_i = C_i / 2.0


                try:
                    logL_i += - 1.0 * (    (B_i**2 / ( 4*A_i ) ) - C_i     + math.log(1.0/D_i * math.sqrt(pi/A_i))     )
                except:
                    pass                    
                    #return np.inf


    
        return logL_i






    params      = ['mu_p',      'mu_Q',      'mu_N',      'sigma_p',      'sigma_Q',      'sigma_N'    ,  'sigma_pQ',      'sigma_pN',      'sigma_QN']
    params_dict = {'mu_p' : [], 'mu_Q' : [], 'mu_N' : [], 'sigma_p' : [], 'sigma_Q' : [], 'sigma_N' : [], 'sigma_pQ' : [], 'sigma_pN' : [], 'sigma_QN' : []}
    
    
    eps = initmax / 1.00000 

   
    for i in range(N):
        


        
        mu_p = random.uniform(0.0, initmax)
        mu_Q = random.uniform(0.0, initmax )
        mu_N = random.uniform(0.0, initmax )
        
        sigma_p = random.uniform(0.0, initmax)#initv+random.uniform(-100,1)#initv*(0.95 + random.random()) #random_init(initv)
        sigma_Q = random.uniform(0.0, initmax)#initv+random.uniform(-100,1)#initv*(1.0  + random.random())#random_init(initv)
        sigma_N = random.uniform(0.0, initmax)#initv+random.uniform(-100,1)#initv*(1.15 + random.random())#random_init(initv)
        
        sigma_pQ = random.uniform(-initmax, initmax)#initv+random.uniform(-100,1000)#initv*(0.02 + random.random()/10.0)#random_init(0)
        sigma_pN = random.uniform(-initmax, initmax)#initv+random.uniform(-100,1000)#initv*(0.03 + random.random()/10.0)#random_init(0)
        sigma_QN = random.uniform(-initmax, initmax)#initv+random.uniform(-100,1000)#initv*(0.04 + random.random()/10.0)#random_init(0)


        initial_guess = [mu_p, mu_Q, mu_N, sigma_p, sigma_Q, sigma_N, sigma_pQ, sigma_pN, sigma_QN]

        
        '''bnds = (( -1000.0, 1000000000.0 ),   # mu_p
                ( -1000.0, 1000000000.0 ),      # mu_Q
                ( 0.0, 100.0 ),                 # mu_N
                ( 0.0, 100.0 ),                 # sigma_p
                ( 0.0, 100.0 ),                 # sigma_Q
                ( 0.0, 100.0 ),                 # sigma_N
                ( -100.0, 100.0 ),              # sigma_pQ
                ( -100.0, 100.0 ),              # sigma_pN
                ( -100.0, 100.0 ))              # sigma_QN
        '''
                    

        model = minimize(LogL_i, np.array(initial_guess), method='L-BFGS-B', options = {'eps' : eps}) #L-BFGS-B
        print i, '\t', initmax, ', director'

        
        
        for i in range(len(initial_guess)):
            if model.success:
                params_dict[params[i]].append(model.x[i])
        
    return params_dict
    


def random_init(q):

    return q + random.uniform(-1.0*q, 10.0)



if __name__ == '__main__':         

    repeat_num = 1000

    params = ['mu_p', 'mu_p_err', 'mu_Q',  'mu_Q_err', 'mu_N', 'mu_N_err', 'sigma_p', 'sigma_p_err', 'sigma_Q', 'sigma_Q_err', 'sigma_N', 'sigma_N_err', 'sigma_pQ', 'sigma_pQ_err','sigma_pN', 'sigma_pN_err', 'sigma_QN', 'sigma_QN_err']

    career = sys.argv[1] 

    f = open('MLE/MLE_params_' + career +'_' + str(repeat_num) + '.dat', 'w') # ' + str(time.time()) + '

    for initmax in [1.0,2.0,3.0, 5.0,7.0,10.0,15.0,20.0,25.0,30.0,40.0,50.0,75.0,100.0,150.0,200.0,250.0,300.0,500.0,1000.0,2000.0,3000.0,5000.0,10000.0,25000.0]:

        param_results = opt(repeat_num, initmax, career)


        vals = {}

        for k, v in sorted(param_results.items()):
            print k, ':\t', np.mean(v), ' +/- ' , np.std(v)/math.sqrt(len(v)), '\t', np.std(v),  '\t',  len(v)
            vals[k] = np.mean(v)
            vals[k+'_err'] = np.std(v)/math.sqrt(len(v))




        f.write('initmax' + '\t' + '\t'.join(params) + '\n')

        f.write(str(initmax) + '\t' +'\t'.join(str(vals[p]) for p in params      )   )
    

    f.close()

        





