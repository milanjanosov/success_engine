"""This file contains some test fitness functions."""

import gzip
import math
import os
import numpy as np

################################################################################
######################## MODEL SPECIFIC FITNESS FUNCTIONS ######################
################################################################################


ROOT   = '../../../../Data/'
LIMIT  = 80
limit  = '-limit-' + str(LIMIT)








def fitness_template(onesingleargument):
    """All model fitness functions should look like this:

    def fitness_mymodel(onesingleargument):

        Arguments:
        ----------

        onesingleargument - anything you wish to use to evaluate fitnesses
                            for all phenotypes. Here below pvalues list is used,
                            where all elements are lists of pvalues from last
                            generation.

        Usage:
        ------

        Use any code to define a fitness for each phenotype.

        Return a dictionary that contains all phenotype (p) values as 
        integer keys and multi-objective fitnesses as dict of float values.

    """
    pass





''' =========================================================== '''
''' ----------------------------------------------------------- '''
''' =========================================================== '''



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

    return - 1.0 * (    (B_i**2 / ( 4*A_i ) ) - C_i +  logD_i  +   math.log( math.sqrt(pi/A_i))     )



def get_sumI(careerf):

    I     = []
    N     = []
    sumI  = []
    files = [careerf + '/' + fn for fn in os.listdir(careerf)]
   
    for filename in files:

        c_i = [float(line.strip().split('\t')[2]) for line in open(filename)]
        N.append(len(c_i)) 
        sumI.append(sum([math.log(c) for c in c_i]))        



    return sumI, N







def liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN):

    sigma_Q2 = sigma_Q**2
    sigma_N2 = sigma_N**2
    sigma_p2 = sigma_p**2

    sigma_QN2 = sigma_QN**2
    sigma_pN2 = sigma_pN**2
    sigma_pQ2 = sigma_pQ**2
    
    nev     = sigma_Q2*sigma_N2 - sigma_QN2


    logL = 0

    try:

        for index, sum_ci in enumerate(sumI):


            sum_ci2  = sum_ci**2
            N_i      = math.log(N[index])
            pi       = math.pi
            expN     = math.exp(N_i)
            inv_expN = 1.0 - math.exp(N_i)

            K = get_K(sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
            
            A_i = get_A_i(expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2 )
            B_i =  get_B_i(sum_ci, sum_ci2, N_i, mu_N, mu_p, mu_Q, N, expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
            C_i = get_C_i(sigma_p, sigma_Q, sigma_N, sum_ci, sum_ci2, N_i, mu_N, mu_p, mu_Q, N, expN, inv_expN, nev, K, sigma_p2, sigma_Q2, sigma_N2, sigma_pQ, sigma_QN, sigma_pN, sigma_pN2, sigma_QN2, sigma_pQ2)
            logD_i = get_logD_i(pi, expN, nev, inv_expN, K)

         
            logL += get_L_i(pi, A_i, B_i, C_i, logD_i)



        return logL

    except ValueError:
        return 0






#######################################################################################################################################
def fitness_mlesuccess_director(pvalues):

    tipus     = 'director'
    fitnesses = {}
    careerf   = ROOT + tipus + '/film-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    
    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]
        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses


#######################################################################################################################################
def fitness_mlesuccess_art_director(pvalues):
    
    tipus     = 'art-director'
    fitnesses = {}
    careerf   = ROOT + tipus + '/film-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses


#######################################################################################################################################
def fitness_mlesuccess_producer(pvalues):

    tipus     = 'producer'
    fitnesses = {}
    careerf   = ROOT + tipus + '/film-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses


#######################################################################################################################################
def fitness_mlesuccess_composer(pvalues):

    tipus     = 'composer'
    fitnesses = {}
    careerf   = ROOT + tipus + '/film-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses


#######################################################################################################################################
def fitness_mlesuccess_writer(pvalues):

    tipus     = 'writer'
    fitnesses = {}
    careerf   = ROOT + tipus + '/film-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses





#######################################################################################################################################
def fitness_mlesuccess_rock(pvalues):

    tipus     = 'rock'
    fitnesses = {}
    careerf   = ROOT + tipus + '/music-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses





#######################################################################################################################################
def fitness_mlesuccess_pop(pvalues):

    tipus     = 'pop'
    fitnesses = {}
    careerf   = ROOT + tipus + '/music-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses



#######################################################################################################################################
def fitness_mlesuccess_hiphop(pvalues):

    tipus     = 'hiphop'
    fitnesses = {}
    careerf   = ROOT + tipus + '/music-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses





#######################################################################################################################################
def fitness_mlesuccess_electro(pvalues):

    tipus     = 'electro'
    fitnesses = {}
    careerf   = ROOT + tipus + '/music-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses



#######################################################################################################################################
def fitness_mlesuccess_folk(pvalues):

    tipus     = 'folk'
    fitnesses = {}
    careerf   = ROOT + tipus + '/music-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses



#######################################################################################################################################
def fitness_mlesuccess_funk(pvalues):

    tipus     = 'funk'
    fitnesses = {}
    careerf   = ROOT + tipus + '/music-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses




#######################################################################################################################################
def fitness_mlesuccess_jazz(pvalues):

    tipus     = 'jazz'
    fitnesses = {}
    careerf   = ROOT + tipus + '/music-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses



#######################################################################################################################################
def fitness_mlesuccess_classical(pvalues):

    tipus     = 'classical'
    fitnesses = {}
    careerf   = ROOT + tipus + '/music-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses


#######################################################################################################################################
def fitness_mlesuccess_authors(pvalues):

    tipus     = 'authors'
    fitnesses = {}
    careerf   = ROOT + tipus + '/book-' + tipus + '-simple-careers' + limit
    sumI, N   = get_sumI(careerf) 

    for i in xrange(len(pvalues)):
        mu_N     = pvalues[i][0]
        mu_p     = pvalues[i][1]
        mu_Q     = pvalues[i][2]
        sigma_N  = pvalues[i][3]

        sigma_Q  = pvalues[i][4]
        sigma_p  = pvalues[i][5]
        sigma_pQ = pvalues[i][6]
        sigma_pN = pvalues[i][7]
        sigma_QN = pvalues[i][8]
        
        fitnesses[i] = {'mlesuccess_' + tipus: liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses




''' =========================================================== '''
''' ----------------------------------------------------------- '''
''' =========================================================== '''



def fitness_linear(pvalues):
    """Fitness function of a simple N-dim linear function.

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get fitnesses for all phenotypes
    for i in xrange(len(pvalues)):
        fitnesses[i] = {"linear": 1}
        for x in pvalues[i]:
            fitnesses[i]["linear"] *= (abs(x)-100)/100.0
    # return fitness dictionary
    print fitnesses

    return fitnesses


def fitness_test(pvalues):
    """Test fitness function.

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get fitnesses for all phenotypes
    for i in xrange(len(pvalues)):
        x = pvalues[i][0]
        y = pvalues[i][1]
        fitnesses[i] = {"test": -(abs(x)-100 + 3*abs(y)-300)}
    # return fitness dictionary
    return fitnesses


def fitness_rosenbrock(pvalues):
    """Fitness function for the N-dim Rosenbrock function.

    http://en.wikipedia.org/wiki/Test_functions_for_optimization

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get fitnesses for all phenotypes
    for i in xrange(len(pvalues)):
        fitnesses[i] = {"rosenbrock": 0}
        for j in xrange(1, len(pvalues[i])):
            x = pvalues[i][j-1]
            y = pvalues[i][j]
            fitnesses[i]["rosenbrock"] -= 100 * (y-x**2)**2 + (x-1)**2
    # return fitness dictionary
    return fitnesses


def fitness_schaffer2(pvalues):
    """Fitness function for the 2D Schaffer2 function.

    http://en.wikipedia.org/wiki/Test_functions_for_optimization

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get fitnesses for all phenotypes
    for i in xrange(len(pvalues)):
        x = pvalues[i][0]
        y = pvalues[i][1]
        fitnesses[i] = {"schaffer2": -1 * (0.5 + ((math.sin(x**2 - y**2))**2 - 0.5)/((1 + 0.001*(x**2 + y**2))**2))}
    # return fitness dictionary
    return fitnesses


def fitness_sphere(pvalues):
    """Fitness function of a simple N-dim quadratic function.

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get fitnesses for all phenotypes
    for i in xrange(len(pvalues)):
        fitnesses[i] = {"sphere": 0}
        for x in pvalues[i]:
            fitnesses[i]["sphere"] -= x**2
    # return fitness dictionary
    return fitnesses


def fitness_flat(pvalues):
    """A completely flat (zero) fitness function for testing purposes.

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get fitnesses for all phenotypes
    for i in xrange(len(pvalues)):
        fitnesses[i] = {"flat": 0}
    # return fitness dictionary
    return fitnesses



