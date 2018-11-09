import os
import sys
import gzip





'''     GET THE DIRECTORS CUMULATIVE CAREERS    '''
''' including all the movies they contriuted to '''


def get_year(year):

    if len(year) == 4: return int(year)
    else: return int(year.split('-')[0])


def get_career_start():

    folder = 'simple-careers/film-director-simple-careers/'
    files  = os.listdir(folder)
    fout   = open('ALL_directors_starting_year.dat', 'w')
    
    for ind, fn in enumerate(files):

        print ind

        years     = ([get_year(line.strip().split('\t')[1]) for line in gzip.open(folder + fn ) if 'movie_id' not in line and len(line.strip().split('\t')[1]) > 1])
        director  = fn.split('_')[0]
    
        if len(years) > 0:
            startyear = min(years)
    	    fout.write( director + '\t' + str(startyear) + '\n')

    fout.close()
        


def get_directors_all_contributed_movies():


    directors    = set([f.split('_')[0] for f in os.listdir('simple-careers/film-director-simple-careers')])
    movies_years = {} 

    for line in open('ALL_movies_years.dat'):

        year, movie  = line.strip().split('\t')
        movies_years[movie] = year   
        

    directors_movies = {}

    for jind, line in enumerate(open('ALL_movies_casts.dat')):

        fields = line.strip().split('\t')
        movie  = fields[0]
        cast   = sorted(fields[1:])
        cast_s = set(cast)

        director = 'nm0000184'

        if director in cast_s:

            if director not in directors_movies:
                directors_movies[director] = []
    
            directors_movies[director].append((movie, get_year(movies_years[movie])))


    folderout = 'NEWTemporal/directors_movies_years/'
    if not os.path.exists(folderout):
        os.makedirs(folderout)


    for director, movies in directors_movies.items():

        fout   = open(folderout + director + '.dat', 'w')
        movies = sorted(movies, key=lambda tup: tup[1])

        for movie, year in movies:
            fout.write( str(year) + '\t' + movie + '\n')





get_career_start()
#get_directors_all_contributed_movies()

