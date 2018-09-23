import os
import gzip
import sys


# get the movie casts
movies_cast = {}

for line in open('ALL_movies_casts.dat'):

    movie = tuple(line.strip().split('\t', 1))
    if len(movie) > 1:
        movie, cast = movie
    else:
        cast = ''
    movies_cast[movie] = cast



# add the casts to the careers

ctype       = sys.argv[1]
field       = 'film'
individuals = [line.strip() for line in open('users_types/Q_film_' + ctype + '_namelist.dat')]
nnn         = len(individuals)
folderout   = 'collab-careers'
folderout   = folderout + '/' + field + '-' + ctype + '-collab-careers/'

if not os.path.exists(folderout):
    os.makedirs(folderout)


for ind, individual in enumerate(individuals):

    print ind, '/', nnn

    fout = open(folderout + individual + '_collab_career.dat', 'w')

    for line in gzip.open('simple-careers/film-' + ctype + '-simple-careers/' + individual + '_' + ctype + '_simple_career.gz'):

        if 'movie_id' not in line:

            if field == 'film':

                line  = line.strip().split('\t') 
                movie = line[0]
                line  = movie + '\t' + line[1] + '\t' + line[3] 

                if movie in movies_cast:
                    cast = movies_cast[movie]
                else:
                    cast = ''

                fout.write(line + '\t' + cast + '\n')

    fout.close() 
