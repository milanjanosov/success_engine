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
import math






''' ================================================= '''
'''    PREPROC	 THE NETWORKS FOR THE IGRAPH STUFF      '''
''' ================================================= '''


'''
- film-director-collab-careers_QDIR   -->  
- film-director-collab-careers_QEVER  -->
- film-director-collab-careers_EVER   -->
'''

#source /opt/virtualenv-python2.7/bin/activate


def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def adamic_adar(a, b):
    c = a.intersection(b)  

    if len(c) > 1: return 1.0 / (math.log(len(c))) 
    else: return 0    


def process_yearly_nw(args):


    yearLIMITs  = args[0]
    thread_id   = args[1]
    num_threads = args[2]
    sam         = args[3] 
    ctype       = args[4]
    tipus       = args[5]
    root        = args[6]
    files       = args[7]

   

    individuals_movie_seq = args[8]
    ids_names             = args[9]

    R = args[10]


    ''' parse year stuff, only for QQ right now '''
    user_first = {}  
    #for line in open('users_types/FirstYears_Qdirector.dat'):
    #    fields = line.strip().split('\t')
    #    user_first[fields[0]] = float(fields[1]) 
    
    '''NOW QE <---- parse year stuff, only for QQ right now '''
    for line in open('users_types/FirstYears_QEVERdirector.dat'):
        fields = line.strip().split('\t')
        user_first[fields[0]] = float(fields[1]) 
    

    Qdir  = set([line.strip() for line in open('users_types/Q_' + 'film' + '_' + ctype + '_namelist.dat')])



    for yearLIMIT in yearLIMITs:


        nodes      = set()
        outfolder  = 'networks' + sam + '/' + ctype + '/' + ctype + tipus + '_' + str(yearLIMIT)
        edges_jacc = {}
        edges_aa   = {}
        edges_cnt  = {}

        print  thread_id, yearLIMIT

        if not os.path.exists(outfolder): os.makedirs(outfolder)

  

        n     = len(files)
        nodes = set()

        for ind, fn in enumerate(files):
                  
            director = fn.split('_')[0]

            

            #  TEST Pa	ROS: nm0160614	nm0580726
            if 'nm0000184' == director :#and yearLIMIT == 2017:   
            #  if ind == 1000: break


                if ind % 1000 == 0: 
                    print thread_id, '/', num_threads, '\t', yearLIMIT, '\t', ind, '/', n

                for line in open(root + fn):

                    #print line

                    fields = line.strip().split('\t') 
           
                    if len(fields) == 4:


                        movie, year, rating, cast = fields
                        if len(year) > 0:
                            year = str(int(min([float(y) for y in year.split('-')])))

                
                            if  year is not None and year != 'None' and len(str(int(year))) == 4 and rating != 'None':# and year is not None:
              
                
                                year   = float(year)
                                rating = float(rating)                        
        

                                if year <= yearLIMIT and rating > 0.0 and year >= user_first[director]:                        

                                    # casts need to be handled as full graphs 
            
                                    cast = [ccc for ccc in list(set(cast.split(',') + [director])) if 'cast' not in ccc and user_first[ccc] <= year]
                        

                                    for c1 in cast:

                                        for c2 in cast:
                                            if c1 != c2:


                                                print c1, c2


                                                edge = '\t'.join(sorted([c1, c2]))

                                                #if 'nm0160614\tnm0580726' == edge:
        
                                                nodes.add(c1)
                                                nodes.add(c2)

                                                if c2 in Qdir:

                                                    movies1 = set(individuals_movie_seq[c1][movie])
                                                    movies2 = set(individuals_movie_seq[c2][movie])
                                                    edges_jacc[edge] = str(jaccard(movies1, movies2))
                                                    edges_aa[edge]   = str(adamic_adar(movies1, movies2))
                         
                                                    if edge not in edges_cnt:
                                                        edges_cnt[edge]  = set()     
                                                    else:
                                                        edges_cnt[edge].add(movie)                            
                                          
     


        hout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_cnt_'    + str(yearLIMIT) + '_R_nullmodels' + '_' + str(R) + '.dat', 'w')
        for e in edges_jacc.keys():
            hout.write(e + '\t' + str(len(edges_cnt[e])) + '\n')               
        hout.close()
      
        gout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_jaccard_'    + str(yearLIMIT) + '_R_nullmodels' + '_' + str(R) +  '.dat', 'w')
        for e in edges_jacc.keys():
            gout.write(e + '\t' + edges_jacc[e] + '\n')               
        gout.close()

        iout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_aa_'    + str(yearLIMIT) + '_R_nullmodels' + '_' + str(R) +  '.dat', 'w')
        for e in edges_jacc.keys():
            iout.write(e + '\t' + edges_aa[e] + '\n')               
        iout.close()

        iout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_jaccard_gephi' + str(yearLIMIT) + '_R_nullmodels' + '_' + str(R) +  '.dat', 'w')
        iout.write('Source\tTarget\tWeight\tType\n')
        for e in edges_jacc.keys():
            if edges_jacc[e] > 0:
                iout.write(e + '\t' + edges_jacc[e] + '\tundirected\n')               
        iout.close()

        iout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_aa_gephi' + str(yearLIMIT) + '_R_nullmodels' + '_' + str(R) +  '.dat', 'w')
        iout.write('Source\tTarget\tWeight\tType\n')
        for e in edges_jacc.keys():
            if edges_aa[e] > 0:
                iout.write(e + '\t' + edges_aa[e] + '\tundirected\n')               
        iout.close()



        iout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_node_list_gephi' + str(yearLIMIT) + '_R_nullmodels' + '_' + str(R) +  '.dat', 'w')
        iout.write('ID\tLabel\n')
        for n in list(nodes):
            iout.write(n + '\t' + ids_names[n] + '\n')               
        iout.close()

        



  

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
    field     = 'film'
    sam       = ''
    neighbrs  = {}
    tipusok   = ['-QQ']#, '-QE', '']
    ids_names = {}
    R         = 1
 
    for line in open('../../../IMDb/IMDB_NAMES_IDs.dat'):
        fields = line.strip().split('\t')
        ids_names[fields[0]] = fields[1]


    if sample: sam = '_sample'

    if sample and not os.path.exists('networks_sample'): 
        os.makedirs('networks_sample')
    if not os.path.exists('networks'): 
        os.makedirs('networks')



    for tipus in tipusok: 


        root   = 'collab-careers/' + field + '-' + ctype + '-collab-careers-QQ_R_nullmodels/' + str(1)   
        root2  = 'collab-cumulative-careers/' + field + '_' + ctype + '-collab-cumulative-careers-QQ'  


        files2 = os.listdir(root2)  
        files  = os.listdir(root)
        nnn    = len(files)




        individuals_movie_seq = {}
        for ind, fn in enumerate(files2):
            
            if ind % 100 == 0: print ind, '/', nnn

            #if ind == 500: break
            name = fn.split('_')[0]    
            individuals_movie_seq[name] = {} 


            for line in open(root2 + '/' + fn):


                try:

                    movie, year, prevmovs = line.strip().split('\t')
                    prevmovs = prevmovs.split(',')

                    individuals_movie_seq[name][movie] = prevmovs
                except:
                    pass



        yearLIMITs = range(1900, 2018)#[1990, 2000, 2010, 2020]
        random.shuffle(yearLIMITs)

        num_threads = 40
        files_chunks = chunkIt(yearLIMITs, num_threads)
        Pros = []
                    
            
        for i in range(0,num_threads):  
            p = Process(target = process_yearly_nw, args=([files_chunks[i], i+1, num_threads, sam, ctype, tipus, root, files, individuals_movie_seq, ids_names, R], ))
            Pros.append(p)
            p.start()
                 
        for t in Pros:
            t.join()
        
      
       


  
            




''' ======================================================== '''
'''    GENERATE THE IGRAPH STUFF OUTPUT IS NODE FEATURES     '''
''' ======================================================== '''



def get_network_measures(G, outfolder, weighttype, thread_id, ctype, tipus, num_threads, yearLIMIT, iweight = None):



    t1 = time.time()
    degree  = G.degree(  )
    print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'degree,         ',          weighttype,   round(time.time() - t1,2), ' seconds'

    t1 = time.time()
    strength  = G.strength( weights = iweight )
    print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'strength,       ',        weighttype,   round(time.time() - t1,2), ' seconds'


    t1 = time.time()
    betweenness  = G.betweenness( weights = iweight )
    print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'betweennesses,  ',  weighttype,  round(time.time() - t1,2), ' seconds'


    t1 = time.time()
    closenesses    = G.closeness( weights = iweight )
    print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'closeness,      ',    weighttype,  round(time.time() - t1,2), ' seconds'


    t1 = time.time()
    clustering     = G.transitivity_local_undirected( weights = iweight )
    print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'clustering,     ',   weighttype,  round(time.time() - t1,2), ' seconds'


    t1 = time.time()
    pagerank      = G.pagerank( weights = iweight )
    print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'pagerank,       ',     weighttype,  round(time.time() - t1,2), ' seconds'


    t1 = time.time()
    eigenvector   = G.eigenvector_centrality( weights = iweight )
    print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'eigenvector,    ',  weighttype,  round(time.time() - t1,2), ' seconds'
    

    t1 = time.time()
    constraint   = G.constraint( weights = iweight )
    print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'constraint,     ',   weighttype,  round(time.time() - t1,2), ' seconds'
    

    node_centralities = {}
    

    for i in range(len(G.vs)):

        node = G.vs[i]['name']

        node_centralities[node] = { 'degree'        : degree[i],
                                    'strength'      : strength[i], 
                                    'betweenness'   : betweenness[i], 
                                    'closeness'     : closenesses[i],
                                    'clustering'    : clustering[i],
                                    'pagerank'      : pagerank[i], 
                                    'eigenvector'   : eigenvector[i],
                                    'constraint'    : constraint[i]
                                   }
 




    if iweight is None:
        isweighted = 'unweighted'
    else:
        isweighted = 'weighted'

    df_centr = pd.DataFrame.from_dict(node_centralities, orient = 'index')
    df_centr.to_csv(outfolder + '/Q' + ctype + '_' + ctype + '_' + tipus + '_NODE_CENTRALITIES_' + weighttype + '_' + str(yearLIMIT) + '_' + isweighted + '.dat', sep = '\t', index = True)




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

        root     = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers' + tipus + sam + '/'
        files    = os.listdir(root)
        infolder = 'networks' + sam + '/' + ctype + '/' + ctype +  tipus + '_' + str(yearLIMIT)


        outfolder_jacc = infolder + '/' + ctype + tipus + '_' + str(yearLIMIT) + '_jacc' 
        outfolder_aa   = infolder + '/' + ctype + tipus + '_' + str(yearLIMIT) + '_aa' 
        outfolder_cnt  = infolder + '/' + ctype + tipus + '_' + str(yearLIMIT) + '_cnt' 

        if not os.path.exists(outfolder_jacc): os.makedirs(outfolder_jacc)
        if not os.path.exists(outfolder_aa)  : os.makedirs(outfolder_aa)
        if not os.path.exists(outfolder_cnt) : os.makedirs(outfolder_cnt)

        filename_jacc = infolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_jaccard_' + str(yearLIMIT) + '.dat'       
        filename_aa   = infolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_aa_'      + str(yearLIMIT) + '.dat'       
        filename_cnt  = infolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_cnt_'     + str(yearLIMIT) + '.dat'       
 

        G_jacc = Graph.Read_Ncol(filename_jacc, weights = True, directed = False)
        G_aa   = Graph.Read_Ncol(filename_aa,   weights = True, directed = False)
        G_cnt  = Graph.Read_Ncol(filename_cnt,  weights = True, directed = False)
 

        #get_network_measures(G_jacc, outfolder_jacc, 'jaccard', thread_id, ctype, tipus, num_threads, yearLIMIT, iweight = None)
        get_network_measures(G_jacc, outfolder_jacc, 'jaccard', thread_id, ctype, tipus, num_threads, yearLIMIT, iweight = 'weight')

        #get_network_measures(G_aa,   outfolder_aa,   'aa',      thread_id, ctype, tipus, num_threads, yearLIMIT, iweight = None)
        #get_network_measures(G_aa,   outfolder_aa,   'aa',      thread_id, ctype, tipus, num_threads, yearLIMIT, iweight = 'weight')

        #get_network_measures(G_cnt,  outfolder_cnt,  'cnt',     thread_id, ctype, tipus, num_threads, yearLIMIT, iweight = None)
        #get_network_measures(G_cnt,  outfolder_cnt,  'cnt',     thread_id, ctype, tipus, num_threads, yearLIMIT, iweight = 'weight')

 



def create_igraphnw(sample):



    '''SET UP TO QE ! ! !!  '''

    ''' '''

    ctype     = 'director'
    sam       = ''
    neighbrs  = {}
    tipusok   = ['-QQ']#['-QQ']#, '-QE', '']


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
      

    if sys.argv[1] == 'get_nws':
        if sys.argv[2] == 'sample':
            sample = True  
        create_full_nws(sample)
    
        
    elif sys.argv[1] == 'get_centr':
        if sys.argv[2] == 'sample':
            sample = True          
        create_igraphnw(sample)



### counts the user numbers
### get_users_numbers_stats()
### get_merged_user_lists()


#remapping_collab_careers()




## source /opt/virtualenv-python2.7/bin/activate










