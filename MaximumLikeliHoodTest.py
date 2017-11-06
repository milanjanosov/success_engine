

from scipy.stats import beta
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

x = np.linspace(0, 5, 100, endpoint=False)
y = multivariate_normal.pdf(x, mean=2.5, cov=0.1); y



#plt.plot(x, y)


#plt.show() 


sigma_p = 1
sigma_Q = 1
sigma_N = 1

sigma_pQ = 0.1
sigma_pN = 0.1
sigma_QN = 0.1

N_i = math.log(10)


expN = math.exp(N_i)
sigma_Q2 = sigma_Q**2
sigma_N2 = sigma_N**2
sigma_p2 = sigma_p**2

sigma_QN2 = sigma_QN**2
sigma_pN2 = sigma_pN**2




K = (sigma_p**2 * sigma_Q2 * sigma_N2)  +  (2* sigma_pQ * sigma_QN * sigma_pN) - (sigma_pN**2 * sigma_Q2 - sigma_QN2 * sigma_p2 + sigma_pQ**2*sigma_N2 )



A_i = (1- expN)/(sigma_Q2 * sigma_N2 - sigma_QN2) * sigma_N2 + expN/K * ( sigma_Q2*sigma_N2 - sigma_QN2  - 2*(sigma_pN*sigma_QN - sigma_pQ*sigma_N2) + sigma_p2 * sigma_N2 - sigma_pN2     )

A_i = A_i / 2


#B_i =
#C_i


D_i = 2.0*math.pi**(1.0+1.0/2 * expN) * (sigma_Q2*sigma_N2 - sigma_QN2)**(1.0/2+1.0/2 * expN) * abs(K)**(expN/2)


print K
print A_i
print D_i
