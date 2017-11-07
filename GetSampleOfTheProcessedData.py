import os
import sys
from shutil import move
import random





p = 0.05

folders = os.listdir('ProcessedData')
for folder in folders:


    subfolders = os.listdir('ProcessedData/'+folder)
    
    
    for subfolder in subfolders:

        folder_o   = 'ProcessedData/' + folder
        filefolder_o = folder_o + '/' + subfolder
        
        folder_t   = 'ProcessedDataSample/' + folder + '_Sample'
        filefolder_t = folder_t + '/' + subfolder
        
        if not os.path.exists(filefolder_t):
            os.makedirs(filefolder_t)


        #print filefolder_o, filefolder_t
    
    
        files = os.listdir(filefolder_o) 
        for filename in files:
            if 'book' in filename:
                p  = p/10.0
            else:
                p = 0.05
            
            f = open(filefolder_t + '/' + filename, 'w')
            for line in open(filefolder_o + '/' + filename):            
                if random.random() < p:
                    f.write(line)                   
            f.close()    
         
  
       
       
