import os
import sys
import gzip
from igraph import Graph
import numpy as np
import pandas as pd
from multiprocessing import Process


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

        print 'Read movies     ', jind, '\t703216\t', len(cast_s) 

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

        print 'Create cumulative careers     ', ind, '/', nnn

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


    directors        = set([f.split('.')[0] for f in os.listdir('NEWTemporal/2_directors_cumulative_careers')])
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

  
    edges_cnt   = {}
    edges_jacc  = {}


    director_movies = {}

    mmm = len(directors)

   # dirrrrs = ['nm0000184', 'nm0000245']

    for ind, dddd in enumerate(directors):

        #if dddd in dirrrrs:
        print 'Parse careers     ', ind, '/', mmm

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


        print  'Create edgelists     ', jind+1, '/703216'
   


        for ind, director1 in enumerate(cast_l):

            #if director1 in dirrrrs:

            start1 = start_years[director1]

            for director2 in cast_l[ind+1:]:

                #if director2 in dirrrrs:

                start2 = start_years[director2]

                if year >= start1 and year >= start2:

                    

    

                    movies1  = director_movies[director1][year]
                    movies2  = director_movies[director2][year]
                    jaccardv = jaccard(movies1, movies2)
                    count    = len(movies1.intersection(movies2))
                    edge     = '\t'.join(sorted([director1, director2]))

                    for yyyy in range(year,2018):

                        if count > 0:

                            if yyyy not in edges_cnt:  edges_cnt[yyyy]  = {}
                            if yyyy not in edges_jacc: edges_jacc[yyyy] = {}

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

        fout = open(edgefolder + '/'+str(year)+'_edge_list_jaccard.dat', 'w')
        for edge, weight in edges.items():
            fout.write(edge + '\t' + str(weight) + '\n')
        fout.close()

    gout.close()


    for year, edges in edges_cnt.items():

        edgefolder = folderout + str(year) + '/'
        if not os.path.exists(edgefolder): os.makedirs(edgefolder)

        fout = open(edgefolder + '/'+str(year)+'_edge_list_count.dat', 'w')
        for edge, weight in edges.items():
            fout.write(edge + '\t' + str(weight) + '\n')
        fout.close()





''' COMPUTE NETWORK CENTRALITIES '''


def add_df_meas(meas, tipus):

    df = pd.DataFrame(meas.items(), columns = ['name', tipus])
    df.index = df.name
    df = df.drop(columns = ['name'])    
    
    return df


def get_centraliti_Values(G_ig, year, fileout):


    N = len(G_ig.vs)

    print '\n', year, '\tGet IG degrees...'
    G_ig.vs['degree_ig'] =  G_ig.degree()

    print year, '\tGet IG clustering...'
    G_ig.vs['clustering_ig'] = G_ig.transitivity_local_undirected( weights = None)
       
    print year, '\tGet IG betweenness...'
    G_ig.vs['betweenness_ig']  = G_ig.betweenness( weights = None)

    print year, '\tGet IG closeness...'
    G_ig.vs['closeness_ig']  = G_ig.closeness( weights = None, normalized = False )

    print year, '\tGet IG pageranks...'
    G_ig.vs['pagerank_ig'] = G_ig.pagerank( weights = None)   
            
    print year, '\tGet IG constraint...'
    G_ig.vs['constraint_ig']  = G_ig.constraint( weights = None )


    degrees_ig       = {}
    clusterings_ig   = {}
    closenesses_ig   = {}
    pageranks_ig     = {}
    constraints_ig   = {}
    betweennesses_ig = {}


    for v in G_ig.vs():

        Bnormalizer =  (N*N-3*N+2) / 2.0
        if np.isnan(v['clustering_ig']):
            v['clustering_ig'] = 0.0

        degrees_ig[v['name']]       = v['degree_ig']#/float(N-1)
        pageranks_ig[v['name']]     = v['pagerank_ig'] 
        constraints_ig[v['name']]   = v['constraint_ig']
        closenesses_ig[v['name']]   = v['closeness_ig']
        betweennesses_ig[v['name']] = v['betweenness_ig']/Bnormalizer
        clusterings_ig[v['name']]   = v['clustering_ig']


    degrees_ig       = add_df_meas(degrees_ig,       'degree_ig')
    clusterings_ig   = add_df_meas(clusterings_ig,   'clustering_ig')
    betweennesses_ig = add_df_meas(betweennesses_ig, 'betweennesse_ig')
    pageranks_ig     = add_df_meas(pageranks_ig,     'pagerank_ig')
    constraints_ig   = add_df_meas(constraints_ig,   'constraint_ig')
    closenesses_ig   = add_df_meas(closenesses_ig,   'closenesse_ig')


    df_ig = degrees_ig.merge(clusterings_ig, left_index=True,  right_index=True)
    df_ig = df_ig.merge(pageranks_ig,        left_index=True,  right_index=True)
    df_ig = df_ig.merge(betweennesses_ig,    left_index=True,  right_index=True)
    df_ig = df_ig.merge(closenesses_ig,      left_index=True,  right_index=True)
    df_ig = df_ig.merge(constraints_ig,      left_index=True,  right_index=True)


    df_ig.to_csv(fileout, na_rep='nan')




def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
        


def calc_centr(years):

    for year in years:
    
        infile  = 'NEWTemporal/3_edgelists/' + str(year) + '/'+str(year)+'_edge_list_jaccard.dat'
        G_ig    = Graph.Read_Ncol(infile, weights = True, directed=False)
        outfile = 'NEWTemporal/3_edgelists/' + str(year) + '/'+str(year)+'_centralities_jaccard.dat'
       
        get_centraliti_Values(G_ig, year, outfile)



def get_network_centralities():

    edgefolder = 'NEWTemporal/3_edgelists/'
    years      = sorted([int(y) for y in os.listdir(edgefolder) if '.d' not in y])
    nthreads   = 3
    years_seg  = chunkIt(years, nthreads)
    Pros       = []

   
    for a in years_seg: 
        if len(a) == 30: a.append(1927)

    years_segments = []
    for i in range(len(years_seg[0])):
        years_segments.append( [years_seg[0][i], years_seg[1][::-1][i], years_seg[2][i]] )

    for years in years_segments: 
        p = Process(target = calc_centr, args=(years, ))
        Pros.append(p)
        p.start()
         
    for t in Pros:
        t.join()
     

    


''' THE NETWORK POSITIONS OVER TIME '''
 
def get_centrality_careers(top_directors):

    edgefolder    = 'NEWTemporal/3_edgelists/'
    years         = sorted([y for y in os.listdir(edgefolder) if '.d' not in y])
    directors_nw  = {}  
    start_years   = {}

    #years    = ['1927', '1928', '1929', '1930']

    #topnames = set(top_directors.keys())

    directors        = set([f.split('_')[0] for f in os.listdir('simple-careers/film-director-simple-careers')])
    for line in open('ALL_directors_starting_year.dat'):
        name, year = line.strip().split('\t')
        start_years[name] = int(year)

    directors_s = directors.intersection(set(start_years.keys()))
    directors   = set(list(directors.intersection(set( [f.split('.')[0] for f in os.listdir('NEWTemporal/2_directors_cumulative_careers/') ]  ))))





    for year in years:

        print year

        for ind, line in enumerate(open(edgefolder + year + '/' + year + '_centralities_jaccard.dat')):
            #if ind == 3: break

            if 'degree_' not in line:

                name,degree_ig,clustering_ig,pagerank_ig,betweennesse_ig,closenesse_ig,constraint_ig = line.strip().split(',')

                if name in directors:


                    measures = [degree_ig,clustering_ig,pagerank_ig,betweennesse_ig,closenesse_ig,constraint_ig]
                    names    = ['degree', 'clustering', 'pagerank', 'betweenness', 'closeness', 'constraint']
                    measures = [float(m) for m in measures]


                    if name not in directors_nw: directors_nw[name] = {}
                    if year not in directors_nw: directors_nw[name][year] = {}

                    for jind, meas in enumerate(measures):
                        directors_nw[name][year][names[jind]] = meas
                    

    folderout = 'NEWTemporal/4_directors_centralities/'
    if not os.path.exists(folderout): os.makedirs(folderout)

    for director, yearly_centralities in directors_nw.items():

        fout = open(folderout + director + '.dat', 'w')

        for year in years:

            if year in yearly_centralities:
            
                centralities = yearly_centralities[year]
                fout.write( year + '\t' + '\t'.join([str(centralities[name]) for name in names]) + '\n')
        
        fout.close()






top_directors = {   'nm0000184' : 'Lucas',
                    'nm0000233' : ' Tarantino',
                    'nm0000229' : ' Spielberg',
                    'nm0000040' : ' Kubrick',
                    'nm0634240' : ' Nolan',
                    'nm0000033' : ' Hitchcock',
                    'nm0000122' : ' Charlie Chaplin',
                    'nm0000631' : ' Ridley Scott',
                    'nm0001053' : ' E Coen',
                    'nm0000142' : ' Eastwood',
                    'nm0001392' : ' P Jackson',
                    'nm0000591' : ' Polanski',
                    'nm0000154' : ' Gibson',
                    'nm0001232' : ' Forman',
                    'nm0001628' : ' Pollack'}







#get_career_start()
#get_directors_all_contributed_movies()
#create_cumulative_careers()

#get_networks()
get_network_centralities()

#
get_centrality_careers(top_directors)

##   source /opt/virtualenv-python2.7/bin/activate







