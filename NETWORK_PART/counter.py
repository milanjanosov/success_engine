import os
import gzip
files = os.listdir('simple-careers/film-director-simple-careers/')
NNN = 0
for ind, fn in enumerate(files):
   # if ind == 100: break
    print ind
    length = 0
    for line in gzip.open('simple-careers/film-director-simple-careers/' + fn):
        try:
            fields = 
            i = float(line.strip().split('\t')[3])
            length += 1
        except:
            pass
        
    if length > 9:
        NNN += 1
        
        
print NNN
