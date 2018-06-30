import os
import random
import numpy as np


Qfolder = 'ProcessedData/ProcessedDataNormalized_no/11_log_Q_wout_means/film_log_Q_wout_mean_rating_count_director.dat'
pfolder = 'ProcessedData/ProcessedDataNormalized_no/9_p_without_avg/film_p_without_mean_rating_count_director.dat'


users_QNs = {}
ps        = []




synth_max = []

fasz = {}






for i in range(100):


    print 'Run:  ', i 


    ps = [float(line.strip()) for line in open(pfolder)]
    random.shuffle(ps)

    for ind, line in enumerate(open(Qfolder)):

        #if ind == 1000: break

        name, N, Q, Qa = line.strip().split('\t')   
        N = int(N)
        Q = float(Q)

        users_QNs[name] = (N, Q)






    for name, (N, Q) in users_QNs.items():

        career = []

        for i in range(N):
            career.append(ps[0])
            del ps[0]

        synth_max.append((N, max(career)))






    for (N, I) in synth_max:

        if N not in fasz:
            fasz[N] = [I]
        else:
            fasz[N].append(I)





fout = open('QTESTMODEL.dat', 'w')
for N, Iavg in fasz.items():
    fout.write(str( N) + '\t' + str( np.mean(Iavg)) + '\n')

fout.close




