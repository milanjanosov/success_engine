import sys
import os
import random
import numpy as np
from multiprocessing import Process
from copy import deepcopy





def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
        







field        = 'film'
ctype        = 'director'
collabroot   = 'collab-careers'
outfolder    = 'collab-cumulative-careers/' + field + '_' + ctype + '-collab-cumulative-careers-ALL'
Qdir         = set([line.strip() for line in open('users_types/Q_' + field + '_' + ctype + '_namelist.dat')])
if not os.path.exists(outfolder):
    os.makedirs(outfolder)


names_movies = {}


for profession in ['art-director', 'director', 'producer', 'writer', 'composer']:


    careerfolder = collabroot + '/' + field + '-' + profession + '-ALL-collab-careers'
    files        = os.listdir(careerfolder)
    nnn          = len(files)

    for ind, fn in enumerate(files[0:10]):
        


        name = fn.split('_')[0]

        #if 'nm0160614' == name:

        print 'PART 1   -  ', ind, '/', nnn

        #if name in Qdir:

        #print name
    
        for line in open(careerfolder + '/' + fn):
        
            fields = line.strip().split('\t',3)      

            if len(fields) == 4:

                movie, year, impact, cast = fields
             
                #year = year + random.random()/10.0
                cast = cast.split('\t') + [name]
                for c in cast:

                    #if c in Qdir:

                    if c not in names_movies:
                        names_movies[c] = [(year, movie)]
                    else:
                        names_movies[c].append((year, movie))



#print len(names_movies['nm0160614'])

            


def process_name_stuff(args):


    names        = args[0]
    names_movies = args[1]
    outfolder    = args[2] 
    thread_id    = args[3]  
    num_thread   = args[4]  


    names_cum_movies = {}
    movies_years     = {}
    nnn              = len(names)


    for ind, name in enumerate(names):

        movies = names_movies[name]

        print 'PART 2   -  ',  ind, '/', nnn, '\t', thread_id, '/', num_thread

        #if 'nm0160614' == name:

        movies                 = sorted(movies, key=lambda tup: tup[0])
        current_movies         = []
        names_cum_movies[name] = {}

        #print movies

        for year, movie in movies: 
            current_movies.append(movie)
            cccc = current_movies
            movies_years[movie] = year
            names_cum_movies[name][movie] = deepcopy(cccc)

            #print name, year, movie, len(cccc)




    for name, cummovies in names_cum_movies.items():
        
        fout = open(outfolder + '/' + name + '_cumulative_movies_career.dat', 'w')
        for movie, prevmovies in cummovies.items():
            fout.write(movie + '\t' + movies_years[movie] + '\t' + ','.join(prevmovies) + '\n')

        fout.close()








all_names   = names_movies.keys()
num_threads = 40 
name_chunks = chunkIt(all_names, num_threads)

Pros = []

for i in range(0,num_threads):  
    p = Process(target = process_name_stuff, args=([name_chunks[i], names_movies, outfolder, i+1, num_threads], ))
    Pros.append(p)
    p.start()
   
for t in Pros:
    t.join()










#  9.05 - 21000

#  7270 file  216.6kb

