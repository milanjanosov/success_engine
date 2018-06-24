import os
import time
import sys
from shutil import copyfile
import gzip
import os
from multiprocessing import Process
import time
import pandas as pd
import random
from igraph import Graph
from shutil import copyfile
import gzip




''' ================================================= '''
'''       GET STATS ABOUT THE NUMBER OF USERS         '''
''' ================================================= '''

def get_users_numbers_stats():


    files = os.listdir('users_types')
    outf  = 'results/'

    if not os.path.exists(outf): os.makedirs(outf)
    
    fout = open(outf + 'user_numbers_stats.dat', 'w')
    fout.write('========================\nNumber of users with Q params\n')


    for fn in files:
        with open('users_types/' + fn) as mf:
            fout.write( fn.split('_')[0] + '\t' + str(len(mf.read().strip().split('\n'))) + '\n')


    fout.write('\n\n========================\nAll the users\n')
    for fn in [fff for fff in os.listdir('simple-careers') if 'simple-careers' in fff]:
        fout.write(fn.split('-')[1] + '\t' + str(len(os.listdir('simple-careers/' + fn))) + '\n')


    fout.close()


def get_merged_user_lists():


    # PUT EVERYONE WITH Q INTO THIS FILE '''
    # QEVERYone gets created

    Qusers = []
    files  = os.listdir('users_types')   
    outf   = 'results/'

    for fn in files:
        Qusers += [line.strip() for line in open('users_types/' + fn)]
        
    Qusers = set(Qusers)

    fout = open('users_types/QEVERYone', 'w')
    for q in Qusers:
        fout.write(q + '\n')
    fout.close()


    # EVERY USER INTO THIS LIST '''
    all_users = []
    for fn in [fff for fff in os.listdir('simple-careers') if 'simple-careers' in fff]:
        all_users += [u.split('_')[0] for u in os.listdir('simple-careers/' + fn)]

    all_users = set(all_users)

    fout = open('users_types/EVERYone', 'w')

    for a in all_users:
        fout.write(a + '\n')
    fout.close()
    
    
    gout = open(outf + 'user_numbers_stats.dat', 'a')
    gout.write('\n\n========================\nCombined numbers\n')
    gout.write('QEVERYone\t' + str(len(Qusers)) + '\n')
    gout.write('\n\n========================\nCombined numbers\n')
    gout.write('EVERYone\t'  + str(len(all_users)) + '\n')
    gout.close()





''' ================================================= '''
'''              GET EVERYONES FIRST MOVIE            '''
''' ================================================= '''




def get_everyones_first():


    ctype  = 'director'

    Qdir   = set([line.strip() for line in open('users_types/Q' + ctype + '_namelist.dat')])
    QEVER  = set([line.strip() for line in open('users_types/QEVERYone')])


    names_first_years = {}

 
    ''' ctype first years '''

    ffiles = ['simple-careers/film-' + ctype + '-simple-careers/' + fn for fn in   os.listdir('simple-careers/film-' + ctype + '-simple-careers')]
    nnn    = len(ffiles)

    fout   = open('users_types/FirstYears_Q' + ctype + '.dat', 'w')


    for ind, fn in enumerate(ffiles):

        name = fn.split('/')[-1].split('_')[0]

        
        if ind % 1000 == 0:
            print 'Dir :',  ind, '/', nnn

        if name in Qdir:

            years = []

            for line in gzip.open(fn):

                
                if 'year' not in line:
                    year = line.strip().split('\t')[1]             
                    if len(year) > 0:
                        if len(year) > 4:
                            year = min([float(yy) for yy in year.split('-')])
                            years.append(float(year))
                        else:
                            years.append(float(year))
        
            fout.write(name + '\t' + str(min(years)) + '\n')


            names_first_years[name] = [min(years)]

    fout.close()
    



    ''' ctype first years '''
    types  = list(set(['producer', 'director', 'composer', 'art-director', 'writer']).difference(set([ctype])))
    ffiles = []



    for tipus in types:
        ffiles += ['simple-careers/film-' + tipus + '-simple-careers/' + fn for fn in   os.listdir('simple-careers/film-' + tipus + '-simple-careers')]

    nnn = len(ffiles)


   
    for ind, fn in enumerate(ffiles):

        name = fn.split('/')[-1].split('_')[0]

        if ind % 1000 == 0:
            print 'EVER  ', ind, '/', nnn

        if name in QEVER:

            years = []

            for line in gzip.open(fn):

                
                if 'year' not in line:
                    year = line.strip().split('\t')[1]             
                    if len(year) > 0:
                        if len(year) > 4:
                            year = min([float(yy) for yy in year.split('-')])
                            years.append(float(year))
                        else:
                            years.append(float(year))
        
   
            if len(years) > 0:  
                if name not in names_first_years:
                    names_first_years[name] = [min(years)]
                else:
                    names_first_years[name].append(min(years))




    gout = open('users_types/FirstYears_QEVER' + ctype + '.dat', 'w')
    for name, years in names_first_years.items():
        gout.write(name + '\t' + str(min(years)) + '\n')
    gout.close()













''' ================================================= '''
'''    REMAP COLLAB NETWORKS BASED ON WHO HAS > 4     '''
''' ================================================= '''



def get_sample():

    # just data sample of 1000 guys to make life easier

    root  = 'collab-careers/film-director-collab-careers/'
    files = [(fn, os.stat(root + fn).st_size) for fn in os.listdir(root)]
    files.sort(key=lambda tup: tup[1], reverse = True)

    roots = 'collab-careers_sample/film-director-collab-careers_sample/'
    if not os.path.exists(roots): os.makedirs(roots)

    for fn in files[0:100]:
        copyfile(root + fn[0], roots + fn[0])



def remapping_collab_careers(sample):

    ctype  = 'director'

    sam = ''
    if sample: sam = '_sample'


    file_Qdir_EVER   = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers'    + sam + '/'      ### this is already done, by def
    file_Qdir_QEVER  = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers-QE' + sam + '/'      
    file_Qdir_Qdir   = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers-QQ' + sam + '/'      




    if not os.path.exists(file_Qdir_EVER): os.makedirs(file_Qdir_EVER)
    if not os.path.exists(file_Qdir_QEVER): os.makedirs(file_Qdir_QEVER)
    if not os.path.exists(file_Qdir_Qdir) : os.makedirs(file_Qdir_Qdir)


    Qdir  = set([line.strip() for line in open('users_types/Q' + ctype + '_namelist.dat')])
    QEVER = set([line.strip() for line in open('users_types/QEVERYone')])

    print len(Qdir), len(QEVER)


    if len( os.listdir(file_Qdir_EVER)) == 0:
        get_sample()


    ffiles = os.listdir(file_Qdir_EVER)
    nnn    = len(ffiles)

    for ind, fn in enumerate(ffiles):
        
    
        if ind % 1000 == 0:
            print 'Remap collab nws\t', ind, '/', nnn
       # if ind == 100: break


        director = fn.split('_')[0]


        if director in QEVER: fQEout = open(file_Qdir_QEVER + fn.replace('.dat', '') + '_QE.dat', 'w')
        if director in Qdir:  fQQout = open(file_Qdir_Qdir  + fn.replace('.dat', '') + '_QQ.dat',  'w')
        


        for line in open(file_Qdir_EVER + fn):
            fields = line.strip().split('\t')

            if len(fields) == 4:

                cast    = fields[3].split(',')
                cast_QE = ''
                cast_QQ = ''

                if director in QEVER:
                    cast_QE = ','.join([ccc for ccc in cast if ccc in QEVER and ccc != director])

                if director in Qdir:
                    cast_QQ = ','.join([ccc for ccc in cast if ccc in Qdir  and ccc != director])

    

                if len(cast_QE) > 0:
                    fQEout.write(fields[0] + '\t' + fields[1] + '\t' + fields[2] + '\t' + cast_QE + '\n')
                    
                if len(cast_QQ) > 0:
                    fQQout.write(fields[0] + '\t' + fields[1] + '\t' + fields[2] + '\t' + cast_QQ + '\n')
      
     
        if director in QEVER: fQEout.close()
        if director in Qdir: fQQout.close()
    
        



''' ================================================= '''
'''    PREPROC	 THE NETWORKS FOR THE IGRAPH STUFF      '''
''' ================================================= '''


'''
- film-director-collab-careers_QDIR   -->  
- film-director-collab-careers_QEVER  -->
- film-director-collab-careers_EVER   -->
'''

#source /opt/virtualenv-python2.7/bin/activate

def process_yearly_nw(args):


    yearLIMITs  = args[0]
    thread_id   = args[1]
    num_threads = args[2]
    sam         = args[3] 
    ctype       = args[4]
    tipus       = args[5]



    ''' ATTENTION : THE user_first[name] >= YEAR CONDITION IS A STRONG ONE
        REASON: GEORGE LUCAS WAS FEATURED ON A MOVIE FROM 1902, BC HIS COMPANY RE-RELEASED THE THING
        PROBABL ON THE LARGE SCALE DOESNT MATTER, BUT THIS IS A BUT WEIRD TO HAVE AS 'EARLY CAREER COLLAB
        SO I AM DROPPING THOSE
    '''
    


    ''' parse year stuff, only for QQ right now '''
    user_first = {}  
    for line in open('users_types/FirstYears_Qdirector.dat'):
        fields = line.strip().split('\t')
        user_first[fields[0]] = float(fields[1]) 
    

    for yearLIMIT in yearLIMITs:

        edges     = {}
        edge_dist = {}
        nodes     = set()


        outfolder = 'networks' + sam + '/' + ctype + tipus + '_' + str(yearLIMIT)
   


        if not os.path.exists(outfolder): os.makedirs(outfolder)


        root  = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers' + tipus + sam + '/'
        files = os.listdir(root)
        gout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_rating_'  + str(yearLIMIT) + '.dat', 'w')
        hout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_'    + str(yearLIMIT) + '.dat', 'w')
        n     = len(files)


        for ind, fn in enumerate(files):
                  
            director = fn.split('_')[0]
               
            #if ind == 1000: break

          #  if ind % 1000 == 0: 
          #      print thread_id, '/', num_threads, '\t', yearLIMIT, '\t', ind, '/', n

            for line in open(root + fn):

                fields = line.strip().split('\t') 
       
                if len(fields) == 4:

                    movie, year, rating, cast = fields
                    if len(year) > 0:
                        year = str(int(min([float(y) for y in year.split('-')])))

            
                        #if 2 == 2:
                        if  year is not None and year != 'None' and len(str(int(year))) == 4 and rating != 'None':# and year is not None:
                        #try:
            
                            year = float(year)
                            rating = float(rating)                        


                            if year <= yearLIMIT and rating > 0.0 and year >= user_first[director]:                        

                                # casts need to be handled as full graphs 
                                cast =  [ccc for ccc in list(set(cast.split(',') + [director])) if 'cast' not in ccc and user_first[ccc] >= year]
            


                                for c1 in cast:

                                    if c1 not in nodes:
                                        nodes.add(c1)

                                    for c2 in cast:
                                        if c1 != c2:

                                            edge = '\t'.join(sorted([c1, c2]))


                                            if edge not in edges:
                                                edges[edge]     = 1
                                                edge_dist[edge] = [rating, [movie]]
                                            else:
                                                edges[edge]        += 1
                                                edge_dist[edge][0] += rating
                                                edge_dist[edge][1].append(movie)


                                            

                                            if 'nm0000184' == director:
                                                print edge

                        #except:
                        #    pass

                

        dataout   = open('networks' + sam  + '/networks_statistics' + tipus + '.dat', 'a') 
        dataout.write(ctype + tipus + '\t' + str(yearLIMIT) + '\t'  + str(len(nodes)) + '\t' + str(len(edges)) + '\n')
        dataout.close()


        f = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_gephi_edges' + str(yearLIMIT) + '.dat', 'w')
        f.write('Source'+'\t'+'Target'+'\t'+'Weight'+'\t'+'Type'+'\n')      


        for e, v in edges.items():
            if 'nm0000184' in e:
                print e, v
            gout.write(e + '\t' + str(edge_dist[e][0]) + '\t' + '--'.join(edge_dist[e][1]) + '\n')
            hout.write(e + '\t' + str(v)            + '\n')              


        f.close()
        

        ''' PROB ADDING THE DICT OF BEST PRODUCTS HERE AND WRITE THEM AS NODE ATTRIBUTES '''
        #g = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_gephi_nodes.dat', 'w')
        #g.write('ID' + '\t' + 'Label' +'\t'+ 'House'+ '\n')    
        #for n, h in list(nodes):
        #    g.write(n + '\t' + n + '\t' + h + '\n')
        #g.close()



        gout.close()
        hout.close()



  

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
        






def create_full_nws(sample):



    ### THIS CODE HERE CREATES LGL EDGELIST FILE OUT OF THE COLLAB CARREER FILES !!!!
    ### LGL-S CREATED HERE

    ## 'QQ' : Qdirector - Qdirextor
    ## 'QE' : Qdirector - Qeveryone
    ## ''   :  


    ctype     = 'director'
    sam       = ''
    neighbrs  = {}

    tipusok   = ['-QQ']#['-QQ']#, '-QE', '']


    if sample: sam = '_sample'

    if sample and not os.path.exists('networks_sample'): 
        os.makedirs('networks_sample')
    if not os.path.exists('networks'): 
        os.makedirs('networks')





    for tipus in tipusok: 


        dataout   = open('networks' + sam + '/networks_statistics_' + tipus + '.dat', 'w') 
        dataout.write('network\tyear\tnodes\tedges\n')
        dataout.close()



        yearLIMITs = range(1900, 2018)#[1990, 2000, 2010, 2020]
        random.shuffle(yearLIMITs)

        num_threads = 40
        files_chunks = chunkIt(yearLIMITs, num_threads)
        Pros = []
                    
            
        for i in range(0,num_threads):  
            p = Process(target = process_yearly_nw, args=([files_chunks[i], i+1, num_threads, sam, ctype, tipus], ))
            Pros.append(p)
            p.start()
           
        for t in Pros:
            t.join()



     






''' ======================================================== '''
'''    GENERATE THE IGRAPH STUFF OUTPUT IS NODE FEATURES     '''
''' ======================================================== '''


def yearly_graph_data(args):

    yearLIMITs  = args[0]
    thread_id   = args[1]
    num_threads = args[2]
    sam         = args[3] 
    ctype       = args[4]
    tipus       = args[5]
    sample      = args[6]


    for yearLIMIT in yearLIMITs:


        edges = {}
        nodes = set()


        outfolder = 'networks' + sam + '/' + ctype + tipus + '_' + str(yearLIMIT)


        #if ind % 1000 == 0:
        #    print thread_id, '/', num_threads, '\t igraph.... \t', yearLIMIT

        if not os.path.exists(outfolder): os.makedirs(outfolder)


        root  = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers' + tipus + sam + '/'
        files = os.listdir(root)

        #filename = outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_'      + str(yearLIMIT) + '.lgl'
        gilename = outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_rating_' + str(yearLIMIT) + '.dat'
        hilename = outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_'   + str(yearLIMIT) + '.dat'

        G        = Graph.Read_Ncol(hilename, weights = True, directed = False) 

        edge_ratings = {}

        for line in open(gilename):
            source, target, rating, movies = line.strip().split('\t')

            rating = float(rating)
            edge   =  '\t'.join([source, target])

            edge_ratings[edge] = rating


        ratings = []

        for ind, g in enumerate(G.es()):

            target = G.vs[g.target]['name']
            source = G.vs[g.source]['name']
            edge   = '\t'.join(sorted([source, target]))
            
            ratings.append( edge_ratings[edge])


        G.es['ratings'] = ratings


        for g in G.es():
            target = G.vs[g.target]['name']
            source = G.vs[g.source]['name']
           



        t1 = time.time()
        degree  = G.strength(                   weights=None)
        print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'degree   ', time.time() - t1


        t1 = time.time()
        strength     = G.strength(                      weights='weight')
        print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'strengthes   ', time.time() - t1


        t1 = time.time()
        ratings     = G.strength(                      weights='ratings')
        print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'ratings   ', time.time() - t1


        t1 = time.time()
        betweenness  = G.betweenness(                   weights='weight')
        print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'betweennesses   ', time.time() - t1


        t1 = time.time()
        clustering    = G.transitivity_local_undirected( weights='weight')
        print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'clustering   ', time.time() - t1


        t1 = time.time()
        pagerank      = G.pagerank(                      weights='weight')
        print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'pagerank   ', time.time() - t1


        t1 = time.time()
        eigenvector   = G.eigenvector_centrality(        weights='weight')
        print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'eigenvector   ', time.time() - t1
        



        node_centralities = {}
        

        for i in range(len(G.vs)):

            node = G.vs[i]['name']


            node_centralities[node] = { 'degree'      : degree[i],
                                        'strength'    : strength[i], 
                                        'ratings'     : ratings[i], 
                                        'betweenness' : betweenness[i], 
                                        'clustering'  : clustering[i], 
                                        'pagerank'    : pagerank[i], 
                                        'eigenvector' : eigenvector[i]}



        df_centr = pd.DataFrame.from_dict(node_centralities, orient = 'index')
        df_centr.to_csv(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_NODE_CENTRALITIES_' + str(yearLIMIT) + '.dat', sep = '\t', index = True)







def create_igraphnw(sample):


    ctype     = 'director'
    sam       = ''
    neighbrs  = {}

    tipusok   = ['-QQ']#['-QQ']#, '-QE', '']

    print tipusok

    if sample: sam = '_sample'

    for tipus in tipusok: 

        yearLIMITs = range(1900, 2018)#[1990, 2000, 2010, 2020]
        random.shuffle(yearLIMITs)


        num_threads = 40
        files_chunks = chunkIt(yearLIMITs, num_threads)
        Pros = []
                    
            
        for i in range(0,num_threads):  
            p = Process(target = yearly_graph_data, args=([files_chunks[i], i+1, num_threads, sam, ctype, tipus, sample], ))
            Pros.append(p)
            p.start()
           
        for t in Pros:
            t.join()



     
         




  



    
if __name__ == '__main__':         


    sample = False
      

    if sys.argv[1] == 'basic_stat':
        get_users_numbers_stats()
    

    elif sys.argv[1] == 'merge_users':
        get_merged_user_lists()     
     

    elif sys.argv[1] == 'remap_collab_careers':
        if sys.argv[2] == 'sample':
            sample = True  
        remapping_collab_careers(sample)


    elif sys.argv[1] == 'get_first_dates':
        get_everyones_first()


    elif sys.argv[1] == 'get_full_network':
        if sys.argv[2] == 'sample':
            sample = True  
        create_full_nws(sample)
    
        
    elif sys.argv[1] == 'create_igraphnw':
        if sys.argv[2] == 'sample':
            sample = True          
        create_igraphnw(sample)



### counts the user numbers
### get_users_numbers_stats()
### get_merged_user_lists()


#remapping_collab_careers()




## source /opt/virtualenv-python2.7/bin/activate










