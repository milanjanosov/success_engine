import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


def func(x, a, x0, sigma):

    return a*np.exp(-(x-x0)**2/(2*sigma**2))



def func2D(X, mu_N, mu_p, mu_Q, sigma_N, sigma_p, sigma_Q ):

    N, Q,p  = X
 

    return 1/(math.sqrt(2*math.pi) * sigma_p*sigma_Q) * np.exp(   - ( ( Q - mu_Q )**2/(2*sigma_Q**2) + ( p - mu_p )**2/(2*sigma_p**2) + ( N - mu_N )**2/(2*sigma_N**2)     ) )


'''
mu_N 2.1 	2.13961975032
mu_Q 5.2 	5.23015861122
mu_p 2.5 	2.498606304
sigma_N 1.0 	1.03779120062
sigma_Q 1.0 	1.04111404623
sigma_p 1.0 	0.991256968701
sigma_NQ 0.05 	0.219626934701
sigma_pQ 0.02 	0.0131432351906
sigma_pN 0.02 	0.0204936942337

'''


files = os.listdir('career_data')

#fout = open('impacts.dat', 'w')


I = []
N = []


        
for filename in files:

    c_i = [math.log(float(line.strip().split('\t')[1])) for line in open('career_data/' + filename)]
    if len(c_i) > 0:
        N.append(math.log(len(c_i)))
    I += c_i



plt.yscale('log')
ydata, bins, bars = plt.hist(I, bins = 1000)
plt.close()

xdata = (bins[1:] + bins[:-1])/2    

plt.plot(xdata, ydata, 'bo')

popt, pcov = curve_fit(func2D, (xdata, xdata, xdata), ydata, maxfev = 10000)

print popt
plt.plot(xdata, func2D((xdata, xdata, xdata), *popt), 'r-')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()



