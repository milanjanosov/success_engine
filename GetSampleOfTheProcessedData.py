import os
import sys
from shutil import move
import random





p = 0.05

folders = os.listdir('ProcessedData')
for folder in folders:

    
    
    if not os.path.exists('ProcessedDataSample/' + folder):
        os.makedirs('ProcessedDataSample/' + folder)

    files = os.listdir('ProcessedData/' + folder) 
    for filename in files:
        #if 'book' in filename:
        #    p  = p/10.0
        #else:
        #    p = 0.05


        f = open('ProcessedDataSample/' + folder + '/' + filename, 'w')
        for line in open('ProcessedData/' + folder + '/' + filename):            
            if random.random() < p:
                f.write(line)                   
        f.close()    
         
  


p = 0.05          
    
folders = os.listdir('ProcessedDataNormalized')
for folder in folders:
    
    if not os.path.exists('ProcessedDataNormalizedSample/' + folder):
        os.makedirs('ProcessedDataNormalizedSample/' + folder)

    files = os.listdir('ProcessedDataNormalized/' + folder)  
    for filename in files:

        #if 'book' in filename:
        #    p = p/10.0
        #else:
        #    p = 0.05

        f = open('ProcessedDataNormalizedSample/' + folder + '/' + filename, 'w')     
        for line in open('ProcessedDataNormalized/' + folder + '/' + filename):            
            if random.random() < p:
                f.write(line)                   
        f.close()    
                  



p = 0.05

folders = os.listdir('ProcessedDataNormalizedRandomized')
for folder in folders:
    
    if not os.path.exists('ProcessedDataNormalizedRandomizedSample/' + folder):
        os.makedirs('ProcessedDataNormalizedRandomizedSample/' + folder)

    files = os.listdir('ProcessedDataNormalizedRandomized/' + folder)  
    for filename in files:


        #if 'book' in filename:
        #    p  = p/10.0
        #else:
        #    p = 0.05

        f = open('ProcessedDataNormalizedRandomizedSample/' + folder + '/' + filename, 'w')     
        for line in open('ProcessedDataNormalizedRandomized/' + folder + '/' + filename):            
            if random.random() < p:
                f.write(line)                   
        f.close()    
       
