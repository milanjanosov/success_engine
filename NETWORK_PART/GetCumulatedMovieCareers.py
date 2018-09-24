import sys
import os
import random
import numpy as np
from multiprocessing import Process
from copy import deepcopy

field        = 'film'
ctype        = 'director'
R            = 10
collabroot   = 'collab-careers'
outfolder    = 'collab-cumulative-careers/' + field + '_' + ctype + '-collab-cumulative-careers-QQ'
Qdir         = set([line.strip() for line in open('users_types/Q_' + field + '_' + ctype + '_namelist.dat')])
if not os.path.exists(outfolder):
    os.makedirs(outfolder)



names_movies = {}


for profession in ['art-director', 'director', 'producer', 'writer', 'composer']:


    careerfolder = collabroot + '/' + field + '-' + profession + '-ALL-collab-careers'
    files        = os.listdir(careerfolder)
    nnn          = len(files)

    for ind, fn in enumerate(files):
        


        name = fn.split('_')[0]

        #if 'nm0296144' == name:

        print ind, '/', nnn, name in Qdir

        if name in Qdir:

        #print name
    
            for line in open(careerfolder + '/' + fn):
            
                fields = line.strip().split('\t',3)      

                if len(fields) == 4:

                    movie, year, impact, cast = fields
                 
                    #year = year + random.random()/10.0
                    cast = cast.split('\t') + [name]
                    for c in cast:

                        if c in Qdir:

                            if c not in names_movies:
                                names_movies[c] = [(year, movie)]
                            else:
                                names_movies[c].append((year, movie))


print names_movies['nm0296144']

            

names_cum_movies = {}
movies_years     = {}
nnn              = len(names_movies)


for ind, (name, movies) in enumerate(names_movies.items()):

   # print ind, '/', nnn

   # if 'nm0296144' == name:

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


