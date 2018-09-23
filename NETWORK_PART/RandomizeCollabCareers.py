import sys
import os
import random
import numpy as np
from multiprocessing import Process




def run_randomizer_thread(r, careerfolder, all_names):

    folderout  = careerfolder + '_R_nullmodels/' + str(r)
    if not os.path.exists(folderout):
        os.makedirs(folderout)


    for ind, fn in enumerate(files):

        if ind % 1000 == 0: print r, '\t', ind, '/', nnn

        fileout = open(folderout + '/' + fn.replace('collab', 'collab_R'), 'w')

        for line in open(careerfolder + '/' + fn):

            movie, year, impact, names = line.strip().split('\t')
            
            names      = names.split(',')
            rand_names = all_names[0:len(names)]
            all_names  = all_names[len(names):]

            fileout.write(movie + '\t' + year + '\t' + impact + '\t' + ','.join(rand_names) + '\n')






field        = 'film'
profession   = 'director'
R            = 50
collabroot   = 'collab-careers'
careerfolder = collabroot + '/' + field + '-' + profession + '-collab-careers-QQ'
files        = os.listdir(careerfolder)
all_names    = []
nnn          = len(files)




for ind, fn in enumerate(files):

    if ind % 1000 == 0: print ind, '/', nnn

    for line in open(careerfolder + '/' + fn):
        all_names += line.strip().split('\t')[3].split(',')
 
random.shuffle(all_names)



num_threads = R
for r in range(R):

    Pros = []
                    
    for i in range(0,num_threads):  
        p = Process(target = run_randomizer_thread, args=(r, careerfolder, all_names, ))
        Pros.append(p)
        p.start()
         
    for t in Pros:
        t.join()

    
        



