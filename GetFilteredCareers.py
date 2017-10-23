
''' TODO '''
'''

- lemasolni a data fileokat N15re
- RQ model -> full
    - impact distr, drop bax, plot all distros
    - normalization
    - correlation plots
    - original and normalized

- RQ model -> top
    - normalized and randomized
    - time distr
    - N*/N plot
    - R-rule plot



- processer - plot all the stuff, and the plotter plots only the interesting examples in a fancy way



'''



import os
import gzip
from shutil import copyfile

N = 15

fields = os.listdir('Data')



for field in fields:
    
    folders = [f for f in os.listdir('Data/'+field) if '.' not in f]

    for folder in folders:
    
        files = os.listdir('Data/' + field + '/' + folder)
        
                    
        outdir = 'Data_' + str(N) + '/' + field + '/' + folder
        if not os.path.exists(outdir):
            os.makedirs(outdir)
         
        
        for filename in files:
            
            infile = 'Data/' + field + '/' + folder + '/' + filename
            if len([line  for line in gzip.open(infile)]) >= N:                          
                copyfile(infile, outdir + '/' + filename)
