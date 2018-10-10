"""This file contains some test fitness functions."""

import gzip
import math
import os
import numpy as np

################################################################################
######################## MODEL SPECIFIC FITNESS FUNCTIONS ######################
################################################################################


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


def fitness_ackley(pvalues):
    """Fitness function for the 2D Ackley's function.

    http://en.wikipedia.org/wiki/Test_functions_for_optimization

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get fitnesses for all phenotypes
    for i in xrange(len(pvalues)):
        x = pvalues[i][0] / 20.0 # convert range from [-100,100] to [-5,5]
        y = pvalues[i][1] / 20.0 # convert range from [-100,100] to [-5,5]
        fitnesses[i] = {"ackley": -1 * (-20 * math.exp( - 0.2 * math.sqrt(0.5*(x**2 + y**2))) - math.exp(0.5*(math.cos(2*math.pi*x) + math.cos(2*math.pi*y))) + math.e + 20)}
    # return fitness dictionary
    print fitnessses
    return fitnesses


def fitness_cross(pvalues):
    """Fitness function for the cross-in-tray function.

    http://en.wikipedia.org/wiki/Test_functions_for_optimization

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get fitnesses for all phenotypes
    for i in xrange(len(pvalues)):
        x = pvalues[i][0] / 10.0 # convert range from [-100,100] to [-10,10]
        y = pvalues[i][1] / 10.0 # convert range from [-100,100] to [-10,10]
        fitnesses[i] = {"cross_in_tray": -1 * ( -0.0001 * math.pow(abs(math.sin(x)*math.sin(y)*math.exp(abs(100-math.sqrt(x**2+y**2)/math.pi)+1)), 0.1))}
    # return fitness dictionary
    return fitnesses


def fitness_eggholder(pvalues):
    """Fitness function for the 2D Eggholder function.

    http://en.wikipedia.org/wiki/Test_functions_for_optimization

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get fitnesses for all phenotypes
    for i in xrange(len(pvalues)):
        x = pvalues[i][0] * 5.12 # convert range from [-100,100] to [-512,512]
        y = pvalues[i][1] * 5.12 # convert range from [-100,100] to [-512,512]
        fitnesses[i] = {"eggholder": -1 * ((y + 47) * math.sin(math.sqrt(abs(y+x/2+47))) + x*math.sin(math.sqrt(abs(x-(y+47)))))}
    # return fitness dictionary
    print fitnesses

    return fitnesses





''' =========================================================== '''
''' ----------------------------------------------------------- '''
''' =========================================================== '''



import numpy as np





def get_mle_values(x, y):

    return -1 * ((-y + 7) * math.sin(math.sqrt(abs(y+x/2+47))) + x*math.sin(math.sqrt(abs(x-(y+47)))))





def lik(m, b, sigma):

    xdata = np.array([1,2,3,4,5])
    ydata = np.array([2,5,8,11,14])

    for i in np.arange(0, len(xdata)):
        y_exp = m * xdata + b

    return -1*(len(xdata)/2 * np.log(2 * np.pi) + len(xdata)/2 * np.log(sigma ** 2) + 1 / (2 * sigma ** 2) * sum((ydata - y_exp) ** 2))


def fitness_mle(pvalues):

    # excepted: 3.00000002e+00, -1.00000005e+00,  2.28448873e-08

    # initialize empty fitness dictionary
    fitnesses = {}


    print 'MAX', lik(10, 10, 10)


    # get fitnesses for all phenotypes
    for i in xrange(len(pvalues)):
        x = pvalues[i][0] 
        y = pvalues[i][1] 
        z = pvalues[i][2] 
        x2 = pvalues[i][3] 
        y2 = pvalues[i][4] 
        z2 = pvalues[i][5] 
        x3 = pvalues[i][6] 
        y3 = pvalues[i][7] 
        z3 = pvalues[i][8]    


     
        fitnesses[i] = {"mle": lik(x, y, z)}
    
    print lik(x, y, z)
    print fitnesses
    # return fitness dictionary
    

    return fitnesses




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

    #return - 1.0 * (    (B_i**2 / ( 4*A_i ) ) - C_i + math.log(1.0/D_i * math.sqrt(pi/A_i))     )
    return - 1.0 * (    (B_i**2 / ( 4*A_i ) ) - C_i +  logD_i  +   math.log( math.sqrt(pi/A_i))     )



def get_sumI(careerf):


    I = []
    N = []
    sumI = []


    files = os.listdir(careerf)      
    for filename in files:
#	print filename
        c_i = [float(line.strip().split('\t')[3]) for line in gzip.open(careerf + '/' + filename) if 'rating' not in line and line.strip().split('\t')[3] != 'None' and float(line.strip().split('\t')[3]) > 0]
#        print c_i
        if len(c_i) > 4:
            N.append(len(c_i))
            for c in c_i:   
                I.append(math.log(c))

            sumI.append(sum([math.log(c) for c in c_i]))        


    return sumI, N





def get_sumI_music(careerf):


    I = []
    N = []
    sumI = []


    files = os.listdir(careerf)      
    for filename in files:

        c_i = [float(line.strip().split('\t')[2]) for line in gzip.open(careerf + '/' + filename) if 'year' not in line and  len(line.strip().split('\t')) > 2 and line.strip().split('\t')[2] != 'None'  and   line.strip().split('\t')[2]  != 'unknown' and float(line.strip().split('\t')[2]) > 0]
#        print c_i
        if len(c_i) > 4:
            N.append(len(c_i))
            for c in c_i:   
                I.append(math.log(c))

            sumI.append(sum([math.log(c) for c in c_i]))        


    return sumI, N








def get_sumI_book(careerf):


    I = []
    N = []
    sumI = []


    files = os.listdir(careerf)      
    for filename in files:

        c_i = [float(line.strip().split('\t')[2].replace(',','')) for line in gzip.open(careerf + '/' + filename) if 'year' not in line and  len(line.strip().split('\t')) > 2 and line.strip().split('\t')[2] != 'None' and    line.strip().split('\t')[2]  != 'unknown' and float(line.strip().split('\t')[2].replace(',','')) > 0]
#        print c_i
        if len(c_i) > 4:
            N.append(len(c_i))
            for c in c_i:   
                I.append(math.log(c))

            sumI.append(sum([math.log(c) for c in c_i]))        


    return sumI, N




def get_sumI_science(careerf):


    I = []
    N = []
    sumI = []


    files = os.listdir(careerf)      
    for filename in files:

        c_i = [float(line.strip().split('\t')[2].replace(',','')) for line in open(careerf + '/' + filename) if 'year' not in line and  len(line.strip().split('\t')) > 2 and line.strip().split('\t')[2] != 'None' and    line.strip().split('\t')[2]  != 'unknown' and float(line.strip().split('\t')[2].replace(',','')) > 0]
#        print c_i
        if len(c_i) > 4:
            N.append(len(c_i))
            for c in c_i:   
                I.append(math.log(c))

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
def fitness_sci_mathematics(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Science/science-mathematics-simple-careers/'
    sumI, N   = get_sumI_science(careerf) 

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
        
        fitnesses[i] = {"sci_mathematics": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses







#######################################################################################################################################
def fitness_mlesuccess_dir(pvalues):

    fitnesses = {}
#    careerf   = '../../../../DataSample/Film/film-director-simple-careers/'
    careerf   = '../../../../Data/Film/film-director-simple-careers/'
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
        
        fitnesses[i] = {"mlesuccess_dir": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses


#######################################################################################################################################
def fitness_mlesuccess_art(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Film/film-art-director-simple-careers/'
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
        
        fitnesses[i] = {"mlesuccess_art": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses


#######################################################################################################################################
def fitness_mlesuccess_prod(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Film/film-producer-simple-careers/'
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
        
        fitnesses[i] = {"mlesuccess_prod": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses


#######################################################################################################################################
def fitness_mlesuccess_comp(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Film/film-composer-simple-careers/'
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
        
        fitnesses[i] = {"mlesuccess_comp": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses


#######################################################################################################################################
def fitness_mlesuccess_wr(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Film/film-writer-simple-careers/'
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
        
        fitnesses[i] = {"mlesuccess_wr": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses





#######################################################################################################################################
def fitness_mlesuccess_rock(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Music/music-rock-simple-careers/'
    sumI, N   = get_sumI_music(careerf) 

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
        
        fitnesses[i] = {"mlesuccess_rock": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses





#######################################################################################################################################
def fitness_mlesuccess_pop(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Music/music-pop-simple-careers/'
    sumI, N   = get_sumI_music(careerf) 

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
        
        fitnesses[i] = {"mlesuccess_pop": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses



#######################################################################################################################################
def fitness_mlesuccess_hiphop(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Music/music-hiphop-simple-careers/'
    sumI, N   = get_sumI_music(careerf) 

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
        
        fitnesses[i] = {"mlesuccess_hiphop": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses





#######################################################################################################################################
def fitness_mlesuccess_electro(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Music/music-electro-simple-careers/'
    sumI, N   = get_sumI_music(careerf) 

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
        
        fitnesses[i] = {"mlesuccess_electro": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses



#######################################################################################################################################
def fitness_mlesuccess_folk(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Music/music-folk-simple-careers/'
    sumI, N   = get_sumI_music(careerf) 

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
        
        fitnesses[i] = {"mlesuccess_folk": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses



#######################################################################################################################################
def fitness_mlesuccess_funk(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Music/music-funk-simple-careers/'
    sumI, N   = get_sumI_music(careerf) 

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
        
        fitnesses[i] = {"mlesuccess_funk": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses




#######################################################################################################################################
def fitness_mlesuccess_jazz(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Music/music-jazz-simple-careers/'
    sumI, N   = get_sumI_music(careerf) 

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
        
        fitnesses[i] = {"mlesuccess_jazz": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses



#######################################################################################################################################
def fitness_mlesuccess_class(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Music/music-classical-simple-careers/'
    sumI, N   = get_sumI_music(careerf) 

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
        
        fitnesses[i] = {"mlesuccess_class": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

    return fitnesses


#######################################################################################################################################
def fitness_mlesuccess_books(pvalues):

    fitnesses = {}
    careerf   = '../../../../Data/Book/book-authors-simple-careers/'
    sumI, N   = get_sumI_book(careerf) 

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
        
        fitnesses[i] = {"mlesuccess_books": liksuccess(sumI, N, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)}

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



