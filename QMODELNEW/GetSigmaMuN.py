import os
import sys
import numpy as np
import math


folder     = 'Data_linrescaled'
subfolders = os.listdir(folder)


for subfolder in subfolders:


    logNs = []

    files = os.listdir(folder + '/' + subfolder)
    
    for fn in files[0:10]:
        
        with open(folder + '/' + subfolder + '/' + fn) as myfile:
    
            n = len(myfile.read().strip().split('\n'))


        logNs.append(math.log(n))



    print subfolder, np.mean(logNs), np.std(logNs)
