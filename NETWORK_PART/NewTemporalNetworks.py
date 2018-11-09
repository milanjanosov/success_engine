import os
import sys
import gzip





'''     GET THE DIRECTORS CUMULATIVE CAREERS    '''
''' including all the movies they contriuted to '''


'''

types:
    - they directed
    - they occurred on the cast somewhere
    - only Q director ties considered
    - all the director ties considered



    - NOW: all the occurrances, and all the directors because this has the most amount of info 
           and is still computationally feasible

'''


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


    directors        = set([f.split('_')[0] for f in os.listdir('simple-careers/film-director-simple-careers')])
    movies_years     = {} 
    start_years      = {}
    directors_movies = {}

    for line in open('ALL_movies_years.dat'):
        year, movie  = line.strip().split('\t')
        movies_years[movie] = year   

    for line in open('ALL_directors_starting_year.dat'):
        name, year = line.strip().split('\t')
        start_years[name] = int(year)

    directors_s = directors.intersection(set(start_years.keys()))
    directors   = list(directors.intersection(set(start_years.keys())))


    for jind, line in enumerate(open('ALL_movies_casts.dat')):
    
       # if jind == 1000: break
  
        fields  = line.strip().split('\t')
        movie   = fields[0]
        cast    = sorted(fields[1:])
        cast_s  = set(cast)
        cast_d  = set(cast_s.intersection(directors_s))
        year    = get_year(movies_years[movie])

        print jind, '\t703216\t', len(cast_s) 

        if len(cast_d) > 0:

            cast_dl = list(cast_d)

            for director in cast_dl:
    
                if director not in directors_movies:
                    directors_movies[director] = []
        
                if year >= start_years[director]:

                    directors_movies[director].append((movie, year))
        
    


    folderout = 'NEWTemporal/1_directors_movies_years/'
    if not os.path.exists(folderout):
        os.makedirs(folderout)


    for director, movies in directors_movies.items():

        fout   = open(folderout + director + '.dat', 'w')
        movies = sorted(movies, key=lambda tup: tup[1])

        for movie, year in movies:
            fout.write( str(year) + '\t' + movie + '\n')

    

def create_cumulative_careers():

    folderin  = 'NEWTemporal/1_directors_movies_years/' 
    folderout = 'NEWTemporal/2_directors_cumulative_careers/'   
    files     = os.listdir(folderin)

    if not os.path.exists(folderout): os.makedirs(folderout)


    for fn in files:

        director      = fn.replace('.dat', '')
        yearly_movies = {}        

        for line in open(folderin + fn):

            year, movie = line.strip().split('\t')
            year        = int(year)

            if year not in yearly_movies:
                yearly_movies[year] = set([movie])
            else:       
                yearly_movies[year].add(movie)

        years = yearly_movies.keys()

        if len(years) > 0:
       
            for year in range(min(years)+1, max(years)+1):

                if year in yearly_movies:
                    yearly_movies[year] = yearly_movies[year].union(yearly_movies[year-1])
                else:
                    yearly_movies[year] = yearly_movies[year-1] 


            fout = open(folderout + fn, 'w')

            for year in range(min(years), max(years)+1):
                fout.write( str(year) + '\t' + '\t'.join(list(yearly_movies[year])) + '\n')

            fout.close()







#get_career_start()
#get_directors_all_contributed_movies()
create_cumulative_careers()
