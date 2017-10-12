import os
import sys
from shutil import move
import random


   
folders = os.listdir('ProcessedData')
for folder in folders:
    
    if not os.path.exists('ProcessedDataSample/' + folder):
        os.makedirs('ProcessedDataSample/' + folder)

    files = os.listdir('ProcessedData/' + folder) 
    for filename in files:

        f = open('ProcessedDataSample/' + folder + '/' + filename, 'w')
        for line in open('ProcessedData/' + folder + '/' + filename):            
            if random.random() < 0.05:
                f.write(line)                   
        f.close()    
         
  
          
    
folders = os.listdir('ProcessedDataNormalized')
for folder in folders:
    
    if not os.path.exists('ProcessedDataNormalizedSample/' + folder):
        os.makedirs('ProcessedDataNormalizedSample/' + folder)

    files = os.listdir('ProcessedDataNormalized/' + folder)  
    for filename in files:

        f = open('ProcessedDataNormalizedSample/' + folder + '/' + filename, 'w')     
        for line in open('ProcessedDataNormalized/' + folder + '/' + filename):            
            if random.random() < 0.005:
                f.write(line)                   
        f.close()    
                  

folders = os.listdir('ProcessedDataNormalizedRandomized')
for folder in folders:
    
    if not os.path.exists('ProcessedDataNormalizedRandomizedSample/' + folder):
        os.makedirs('ProcessedDataNormalizedRandomizedSample/' + folder)

    files = os.listdir('ProcessedDataNormalizedRandomized/' + folder)  
    for filename in files:

        f = open('ProcessedDataNormalizedRandomizedSample/' + folder + '/' + filename, 'w')     
        for line in open('ProcessedDataNormalizedRandomized/' + folder + '/' + filename):            
            if random.random() < 0.005:
                f.write(line)                   
        f.close()    
        
