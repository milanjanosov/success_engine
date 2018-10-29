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



''' ==================    TODO    ----------------- 

    - randomized collab network, and the after-filtered ones also randomized
    - see the size of the 
    - % of nodes and edges using the DF and NC backboning on the 2017 network
    - do the same backboning and see how many of nodes get dropped randomly, in the sense that the backboning keeps the node their in round1, 2, 3, 4 but drops in 5, gets back in 6, 7...
        --- a plot on the % of nodes that coma and go, or the distribution of the number of 'moves' - 1 is to drop out, 1 is to come back, comparing to the non-backboned case
    - create slides about how i make these networks, add stats



 -----------------------======================== '''



''' ================================================= '''
'''              GET EVERYONES FIRST MOVIE            '''
''' ================================================= '''




def get_everyones_first():


    ctype  = 'director'
    field  = 'film'

    Qdir  = set([line.strip() for line in open('users_types/Q_' + field + '_' + ctype + '_namelist.dat')])
    QEVER = set([line.strip() for line in open('users_types/Q_' + field + '_EVERYONE_namelist.dat')])


    names_first_years = {}

 
    ''' ctype first years '''

    ffiles = ['simple-careers/film-' + ctype + '-simple-careers/' + fn for fn in   os.listdir('simple-careers/film-' + ctype + '-simple-careers')]
    nnn    = len(ffiles)

    fout   = open('users_types/FirstYears_ALL' + ctype + '.dat', 'w')


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
    field  = 'film'

    sam = ''
    if sample: sam = '_sample'


    file_Qdir_EVER   = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers'    + sam + '/'      ### this is already done, by def
    file_Qdir_QEVER  = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers-QE' + sam + '/'      
    file_Qdir_Qdir   = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers-QQ' + sam + '/'      


    if not os.path.exists(file_Qdir_EVER): os.makedirs(file_Qdir_EVER)
    if not os.path.exists(file_Qdir_QEVER): os.makedirs(file_Qdir_QEVER)
    if not os.path.exists(file_Qdir_Qdir) : os.makedirs(file_Qdir_Qdir)


    Qdir  = set([line.strip() for line in open('users_types/Q_' + field + '_' + ctype + '_namelist.dat')])
    QEVER = set([line.strip() for line in open('users_types/Q_' + field + '_EVERYONE_namelist.dat')])

    print 'Q' + ctype + ': ', len(Qdir)
    print 'Qeveryone: '     , len(QEVER)

    if len( os.listdir(file_Qdir_EVER)) == 0:
        get_sample()


    ffiles = os.listdir(file_Qdir_EVER)
    nnn    = len(ffiles)
    
    print nnn   

    for ind, fn in enumerate(ffiles):
        
    
        if ind % 1000 == 0:
            print 'Remap collab nws\t', ind, '/', nnn
        #if ind == 100: break

    

        director = fn.split('_')[0]


        if director in QEVER: fQEout = open(file_Qdir_QEVER + fn.replace('.dat', '') + '_QE.dat', 'w')
        if director in Qdir:  fQQout = open(file_Qdir_Qdir  + fn.replace('.dat', '') + '_QQ.dat',  'w')
        

        for line in open(file_Qdir_EVER + fn):
            fields = line.strip().split('\t')

   
            if len(fields) > 3:

                cast    = fields[3:]

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




    '''    ---------------------------------------    '''
    '''   FirstYears_QEVERdirector  CLEANING NEEDED   '''
    '''    ---------------------------------------    '''




    ''' parse year stuff, only for QQ right now '''
    user_first = {}  
    #for line in open('users_types/FirstYears_Qdirector.dat'):
    #    fields = line.strip().split('\t')
    #    user_first[fields[0]] = float(fields[1]) 
    
    '''NOW QE <---- parse year stuff, only for QQ right now '''
   # for line in open('users_types/FirstYears_QEVERdirector.dat'):
   #     fields = line.strip().split('\t')
   #     user_first[fields[0]] = float(fields[1]) 
    

    #Qdir  = set([line.strip() for line in open('users_types/Q_' + 'film' + '_' + ctype + '_namelist.dat')])



    for yearLIMIT in yearLIMITs:


        nodes      = set()
        outfolder  = 'networks' + sam + '/' + ctype + '/' + ctype + tipus + '_' + str(yearLIMIT)
        edges_jacc = {}
        #edges_aa   = {}
        #edges_cnt  = {}

        #print  thread_id, yearLIMIT

        if not os.path.exists(outfolder): os.makedirs(outfolder)

  

        n     = len(files)
        nodes = set()

        for ind, fn in enumerate(files):
                  
            director = fn.split('_')[0]

       
            

            #  TEST Pa	ROS: nm0160614	nm0580726
            #if 'nm0000184' == director :#and yearLIMIT == 2017:   
            if 2 == 2:

                #if ind == 10000: break


               # if ind % 100 == 0: 
                #    print thread_id, '/', num_threads, '\t', yearLIMIT, '\t', ind, '/', n


                for line in open(root + fn):

                    #print line

                    fields = line.strip().split('\t') 
           
                    print 'FFF   ', fields, '\n'
                 

                    if len(fields) == 4:


                        movie, year, rating, cast = fields
                        if len(year) > 0:
                            year = str(int(min([float(y) for y in year.split('-')])))

                
                            if  year is not None and year != 'None' and len(str(int(year))) == 4 and rating != 'None':# and year is not None:
              
                
                                year   = float(year)
                                rating = float(rating)                        
        
                      
                                if year <= yearLIMIT and rating > 0.0: # and year >= user_first[director]:                        

                                    # casts need to be handled as full graphs 
            

                                  
                                    cast = [ccc for ccc in list(set(cast.split(',') + [director])) if 'cast' not in ccc] # and user_first[ccc] <= year]
                        

                                    for c1 in cast:

                                        for c2 in cast:
                                            if c1 != c2:


                                                #print c1, c2


                                                edge = '\t'.join(sorted([c1, c2]))

                                                #if 'nm0160614\tnm0580726' == edge:
        
                                                nodes.add(c1)
                                                nodes.add(c2)

                                                #if c2 in Qdir:

                                                movies1 = set(individuals_movie_seq[c1][movie])
                                                movies2 = set(individuals_movie_seq[c2][movie])

                                                print 'EE    ', edge, jaccard(movies1, movies2)

                                                edges_jacc[edge] = str(jaccard(movies1, movies2))
                                                #edges_aa[edge]   = str(adamic_adar(movies1, movies2))
                     
                                                #if edge not in edges_cnt:
                                                #    edges_cnt[edge]  = set()     
                                                #else:
                                                #    edges_cnt[edge].add(movie)                            
                                          
     


        #hout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_cnt_'    + str(yearLIMIT) + '.dat', 'w')
        #for e in edges_jacc.keys():
        #    hout.write(e + '\t' + str(len(edges_cnt[e])) + '\n')               
        #hout.close()
      
        gout  = open(outfolder + '/ALL' + ctype + '_' + ctype + tipus + '_edges_list_jaccard_'    + str(yearLIMIT) + '.dat', 'w')
        for e in edges_jacc.keys():
            gout.write(e + '\t' + edges_jacc[e] + '\n')               
        gout.close()

        #iout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_aa_'    + str(yearLIMIT) + '.dat', 'w')
        #for e in edges_jacc.keys():
        #    iout.write(e + '\t' + edges_aa[e] + '\n')               
        #iout.close()

        iout  = open(outfolder + '/ALL' + ctype + '_' + ctype + tipus + '_edges_list_jaccard_gephi' + str(yearLIMIT) + '.dat', 'w')
        iout.write('Source\tTarget\tWeight\tType\n')
        for e in edges_jacc.keys():
            if edges_jacc[e] > 0:
                iout.write(e + '\t' + edges_jacc[e] + '\tundirected\n')               
        iout.close()

        #iout  = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_aa_gephi' + str(yearLIMIT) + '.dat', 'w')
        #iout.write('Source\tTarget\tWeight\tType\n')
        #for e in edges_jacc.keys():
        #    if edges_aa[e] > 0:
        #        iout.write(e + '\t' + edges_aa[e] + '\tundirected\n')               
        ##iout.close()



        iout  = open(outfolder + '/ALL' + ctype + '_' + ctype + tipus + '_node_list_gephi' + str(yearLIMIT) + '.dat', 'w')
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
    tipusok   = ['-ALL']#, '-QE', '']
    ids_names = {}

 
    for line in open('../../../IMDb/IMDB_NAMES_IDs.dat'):
        fields = line.strip().split('\t')
        ids_names[fields[0]] = fields[1]


    if sample: sam = '_sample'

    if sample and not os.path.exists('networks_sample'): 
        os.makedirs('networks_sample')
    if not os.path.exists('networks'): 
        os.makedirs('networks')



    #for tipus in tipusok: 

    if len(sam) > get_sample():
        get_sample()


    root   = 'collab-careers/' + field + '-' + ctype + '-ALL-collab-careers/'
    root2  = 'collab-cumulative-careers/' + field + '_' + ctype + '-collab-cumulative-careers-ALL'  


    files2 = os.listdir(root2)  
    files  = os.listdir(root)
    nnn    = len(files)
    nnnn   = len(files2)


    


    individuals_movie_seq = {}
    for ind, fn in enumerate(files2[0:10000]):
        
        if ind % 100 == 0: print ind, '/', nnnn

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



    yearLIMITs = range(1999, 2000)#[1990, 2000, 2010, 2020]
    random.shuffle(yearLIMITs)

    num_threads = 1
    files_chunks = chunkIt(yearLIMITs, num_threads)
    Pros = []
                
    tipus = 'ALL'        

    for i in range(0,num_threads):  
        p = Process(target = process_yearly_nw, args=([files_chunks[i], i+1, num_threads, sam, ctype, tipus, root, files, individuals_movie_seq, ids_names], ))
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


  #  t1 = time.time()
  #  betweenness  = G.betweenness( weights = iweight )
  #  print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'betweennesses,  ',  weighttype,  round(time.time() - t1,2), ' seconds'


  #  t1 = time.time()
  #  closenesses    = G.closeness( weights = iweight )
  #  print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'closeness,      ',    weighttype,  round(time.time() - t1,2), ' seconds'


    t1 = time.time()
    clustering     = G.transitivity_local_undirected( weights = iweight )
    print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'clustering,     ',   weighttype,  round(time.time() - t1,2), ' seconds'


    t1 = time.time()
    pagerank      = G.pagerank( weights = iweight )
    print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'pagerank,       ',     weighttype,  round(time.time() - t1,2), ' seconds'


   # t1 = time.time()
   # eigenvector   = G.eigenvector_centrality( weights = iweight )
   # print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'eigenvector,    ',  weighttype,  round(time.time() - t1,2), ' seconds'
    

   # t1 = time.time()
   # constraint   = G.constraint( weights = iweight )
   # print thread_id, '/', num_threads, '   ', yearLIMIT,  '\t', 'constraint,     ',   weighttype,  round(time.time() - t1,2), ' seconds'
    

    node_centralities = {}
    

    for i in range(len(G.vs)):

        node = G.vs[i]['name']

        node_centralities[node] = { 'degree'        : degree[i],
                                    'strength'      : strength[i], 
                                   # 'betweenness'   : betweenness[i], 
                                   # 'closeness'     : closenesses[i],
                                    'clustering'    : clustering[i],
                                    'pagerank'      : pagerank[i] 
                                   # 'eigenvector'   : eigenvector[i],
                                   # 'constraint'    : constraint[i]
                                   }
 




    if iweight is None:
        isweighted = 'unweighted'
    else:
        isweighted = 'weighted'

    df_centr = pd.DataFrame.from_dict(node_centralities, orient = 'index')
    df_centr.to_csv(outfolder + '/ALL' + ctype + '_' + ctype + '_' + tipus + '_NODE_CENTRALITIES_' + weighttype + '_' + str(yearLIMIT) + '_' + isweighted + '.dat', sep = '\t', index = True)




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
        #outfolder_aa   = infolder + '/' + ctype + tipus + '_' + str(yearLIMIT) + '_aa' 
        #outfolder_cnt  = infolder + '/' + ctype + tipus + '_' + str(yearLIMIT) + '_cnt' 

        if not os.path.exists(outfolder_jacc): os.makedirs(outfolder_jacc)
        #if not os.path.exists(outfolder_aa)  : os.makedirs(outfolder_aa)
        #if not os.path.exists(outfolder_cnt) : os.makedirs(outfolder_cnt)

        filename_jacc = infolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_jaccard_' + str(yearLIMIT) + '.dat'       
        #filename_aa   = infolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_aa_'      + str(yearLIMIT) + '.dat'       
        #filename_cnt  = infolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_cnt_'     + str(yearLIMIT) + '.dat'       
 

        G_jacc = Graph.Read_Ncol(filename_jacc, weights = True, directed = False)
        #G_aa   = Graph.Read_Ncol(filename_aa,   weights = True, directed = False)
        #G_cnt  = Graph.Read_Ncol(filename_cnt,  weights = True, directed = False)
 

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

    #for tipus in tipusok: 

    yearLIMITs = range(1900, 2018)#[1990, 2000, 2010, 2020]
    random.shuffle(yearLIMITs)


    num_threads = 40
    files_chunks = chunkIt(yearLIMITs, num_threads)
    Pros = []
                
        
    for i in range(0,num_threads):  
        p = Process(target = yearly_graph_data, args=([files_chunks[i], i+1, num_threads, sam, ctype, 'ALL', sample], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()



     
         




  



    
if __name__ == '__main__':         


    sample = False
      

    if sys.argv[1] == 'remap_collab_careers':
        if sys.argv[2] == 'sample':
            sample = True  
        remapping_collab_careers(sample)


    elif sys.argv[1] == 'get_first_dates':
        get_everyones_first()


    elif sys.argv[1] == 'get_nws':
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










