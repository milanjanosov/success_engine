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
    nnn       = len(files)

    if not os.path.exists(folderout): os.makedirs(folderout)


    for ind, fn in enumerate(files):

        print ind, '/', nnn

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





'''     GET THE DIRECTORS COLLABORATION CAREERS    '''


def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_networks():


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
    directors   = list(directors.intersection(set( [f.split('.')[0] for f in os.listdir('NEWTemporal/2_directors_cumulative_careers/') ]  )))
    edges_cnt   = {}
    edges_jacc  = {}


    director_movies = {}

    mmm = len(directors)

   # dirrrrs = ['nm0000184', 'nm0000245']

    for ind, dddd in enumerate(directors):

        #if dddd in dirrrrs:
        print 'Parse careers   ', ind, '/', mmm

        for line in open('NEWTemporal/2_directors_cumulative_careers/' + dddd + '.dat'):
            
            fields = line.strip().split('\t')
            year   = int(fields[0])
            movies = set(fields[1:])
    
            if dddd not in director_movies: director_movies[dddd] = {}
            director_movies[dddd][year] = movies



    for jind, line in enumerate(open('ALL_movies_casts.dat')):
    
       # if jind == 1000: break
  
        fields  = line.strip().split('\t')
        movie   = fields[0]
        cast    = sorted(fields[1:])
        cast_s  = set(cast)
        cast_d  = set(cast_s.intersection(directors_s))
        cast_l  = list(set(cast_s.intersection(directors_s)))
        year    = get_year(movies_years[movie])


        print jind, '/703216'
   


        for ind, director1 in enumerate(cast_l):

            #if director1 in dirrrrs:

            start1 = start_years[director1]

            for director2 in cast_l[ind+1:]:

                #if director2 in dirrrrs:

                start2 = start_years[director2]

                if year >= start1 and year >= start2:

                    if year not in edges_cnt:  edges_cnt[year]  = {}
                    if year not in edges_jacc: edges_jacc[year] = {}

    

                    movies1  = director_movies[director1][year]
                    movies2  = director_movies[director2][year]
                    jaccardv = jaccard(movies1, movies2)
                    count    = len(movies1.intersection(movies2))
                    edge     = '--'.join(sorted([director1, director2]))

                    for yyyy in range(year,2018):

                        edges_cnt[yyyy][edge]  = count
                        edges_jacc[yyyy][edge] = jaccardv

                    #print edge, year, count, jaccardv



    folderout = 'NEWTemporal/3_edgelists/'
    if not os.path.exists(folderout): os.makedirs(folderout)

    gout = open('NEWTemporal/3_edgelists/networks_size_over_time.dat', 'w')
    
    for year, edges in edges_jacc.items():

        gout.write(str(year) + '\t' + str(len(edges)) + '\n')

        edgefolder = folderout + str(year) + '/'
        if not os.path.exists(edgefolder): os.makedirs(edgefolder)

        fout = open(edgefolder + 'edge_list_jaccard.dat', 'w')
        for edge, weight in edges.items():
            fout.write(edge + '\t' + str(weight) + '\n')
        fout.close()

    gout.close()


    for year, edges in edges_cnt.items():

        edgefolder = folderout + str(year) + '/'
        if not os.path.exists(edgefolder): os.makedirs(edgefolder)

        fout = open(edgefolder + 'edge_list_count.dat', 'w')
        for edge, weight in edges.items():
            fout.write(edge + '\t' + str(weight) + '\n')
        fout.close()




#get_career_start()
#get_directors_all_contributed_movies()
#create_cumulative_careers()

get_networks()







