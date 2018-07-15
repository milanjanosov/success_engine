import os
import random
import numpy as np
from multiprocessing import Process



def getPercentileBinnedDistribution(x, y, nbins):

    x, y = zip(*sorted(zip(x, y), key=lambda tup: tup[0]))
    elements_per_bin = int(len(x)/float(nbins))

    xx  = [np.mean(x[i*elements_per_bin:(i+1)*elements_per_bin]) for i in range(nbins)]
    yy  = [np.mean(y[i*elements_per_bin:(i+1)*elements_per_bin]) for i in range(nbins)]
    std = [np.std(y[i*elements_per_bin:(i+1)*elements_per_bin])  for i in range(nbins)]

    return xx, yy, std


def divideUnequal(list1, list2):
    counter=0
    step=0
    divided=[]
    for count in list1:
        count = int(count)
        step= counter+ count
        sublist= list2[counter: step]
        counter= step
        divided.append(sublist)
    return divided



def generate_career_data(args):


    Qfolder = args[0]
    pfolder = args[1]
    label   = args[2]


    users_QNs = {}
    ps        = []
    synth_max = []



    NNN = 100
    outfolder = 'QTESTRESULTS/' + label + '_' + str(NNN)



    ps = {}

    for line in open(pfolder):

        try:
            name, p = line.strip().split('\t')

            if name not in ps:
                ps[name] = [float(p)]
            else:
                ps[name].append(float(p))
        except:

            pass
        
    

    career_len = []
    allps = []

    for name, p in ps.items():
        career_len.append(len(p))
        allps += p
    


    random.shuffle(allps)


    split_ps = divideUnequal(career_len, allps)  



    
    for i in range(NNN):


        

        #ps = [float(line.strip()) for line in open(pfolder)]
        print 'Run:  ', label, '\t', i+1, '/', NNN, '\t', len(ps)
        #random.shuffle(ps)

        for ind, line in enumerate(open(Qfolder)):

            #if ind == 100: break

    
            #if ind == 1000: break

            fields = line.strip().split('\t')   

            if len(fields) == 4:

                name, N, Q, Qa = fields
                N = int(N)
                Q = float(Q)

                users_QNs[name] = (N, Q)


        for ind, (name, (N, Q)) in enumerate(users_QNs.items()):
            career = []
    

           # print name, sorted(ps[name])
           # print name, sorted(split_ps[ind]), '\n'


            #for p in ps[name]:
            for p in split_ps[ind]:

                career.append(Q*p)

            #for i in range(N):
            #    career.append(Q*allps[0])
            #    del allps[0]


            synth_max.append((N, max(career)))

    




    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    gout = open(outfolder + '/QCareerData_' + label + '.dat', 'w')
    n = []
    q = []
    for (N, Iavg) in synth_max:
        gout.write(str( N) + '\t' + str( np.mean(Iavg)) + '\n')
        n.append(N)
        q.append(np.mean(Iavg))
    gout.close

    




    #for line in open(hhhout):
    #    print line.strip()
     #   print tuple([float(fff) for fff in line.strip().split('\t')]) 



   
    '''hhhout = outfolder + '/QCareerData_' + label + '.dat'
    print hhhout
    for line in open( outfolder + '/QCareerData_' + label + '.dat'):
        print 'FASZ'
    '''
  

    #n, q   = zip(*[ tuple([float(fff) for fff in line.strip().split('\t')]) for line in open(hhhout) ])




    for nbins in [7, 8, 10, 12, 15, 20]:

        print label, '  binning:  ', nbins

        fout = open(outfolder + '/' + label + '_qmodel_' + label + '_' + str(nbins), 'w')

        xb_Qgen, pb_Qgen, pberr_Qgen = getPercentileBinnedDistribution(np.asarray(n),  np.asarray(q), nbins)

        for i in range(len(xb_Qgen)):
            fout.write( str(xb_Qgen[i]) + '\t' + str(pb_Qgen[i]) + '\t' + str(pberr_Qgen[i]) + '\n' )
        fout.close()
    
    
    

    




input_fields = [('film',       'director'),
                ('film',       'art-director'),
                ('film',       'producer'),   
                ('film',       'writer'),   
                ('film',       'composer'),   
                ('music',      'pop'),
                ('music',      'electro'),
                ('music',      'classical'),
                ('music',      'folk'),
                ('music',      'funk'),
                ('music',      'jazz'),
                ('music',      'hiphop'),                   		
                ('music',      'rock'),  
                ('book',       'authors') ]




for (field, label) in input_fields[0:13]:



    cnt = 'rating'
    if 'music' in field:
        cnt = 'play'


    Qfolder = 'ProcessedData/ProcessedDataNormalized_no_3/11_log_Q_wout_means/' + field + '_log_Q_wout_mean_' + cnt + '_count_' + label + '.dat'
    pfolder = 'ProcessedData/ProcessedDataNormalized_no_3/9_p_without_avg/'     + field + '_p_without_mean_'  + cnt + '_count_' + label + '.dat'


    Pros = []
    

    p = Process(target = generate_career_data, args=([Qfolder, pfolder, label], ))
    Pros.append(p)
    p.start()
       
for t in Pros:
    t.join()




