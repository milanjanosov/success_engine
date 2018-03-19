import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

def gaussian(x, mu, sig):
    return 1./(math.sqrt(2.*math.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def lik(parameters):

    mu    = parameters[0]
    sigma = parameters[1]    
    n     = len(x)  
    L     = n/2.0 * np.log(2 * np.pi) + n/2.0 * math.log(sigma **2 ) + 1/(2*sigma**2) * sum([(x_ - mu)**2 for x_ in x ])

    return L






I = []
N = []
files = os.listdir('career_data')      
for filename in files:
    c_i = [float(line.strip().split('\t')[1]) for line in open('career_data/' + filename)]
    N.append(len(c_i))
    for c in c_i:   
        I.append(math.log(c))



y, x, bin = plt.hist(I, color='r',bins = 20 ,alpha = 0.5, normed = True)    
x = (x[1:] + x[:-1])/2   


mu0    = 10
sigma0 = 3


x = I #np.random.normal(loc=mu0, scale=sigma0, size=len(I))
#y = I
#y = gaussian(x, mu0, sigma0)



lik_model = minimize(lik, np.array([2.1,2.1]), method='L-BFGS-B')


mu    = lik_model['x'][0]
sigma = lik_model['x'][1]

print lik_model

plt.hist(I, bins = 30, alpha = 0.5, color = 'r')#, 'o', label = 'fit')
plt.hist(x, bins = 30, alpha = 0.5, color = 'b')#, 'o', label = 'data')

plt.show()
