import os
import time
import sys
from shutil import copyfile
import gzip
import os
import time
import pandas as pd
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


    if len( os.listdir(file_Qdir_EVER)) == 0:
        get_sample()

    for ind, fn in enumerate(os.listdir(file_Qdir_EVER)):
        

        fQEout = open(file_Qdir_QEVER + fn.replace('.dat', '') + '_QE.dat', 'w')
        fQQout = open(file_Qdir_Qdir  + fn.replace('.dat', '') + '_QQ.dat',  'w')
        
        director = fn.split('_')[0]


        for line in open(file_Qdir_EVER + fn):
            fields = line.strip().split('\t')

            if len(fields) == 4:

                cast    = fields[3].split(',')
                cast_QE = ','.join([ccc for ccc in cast if ccc in QEVER and ccc != director])
                cast_QQ = ','.join([ccc for ccc in cast if ccc in Qdir  and ccc != director])

                if len(cast_QE) > 0:
                    fQEout.write(fields[0] + '\t' + fields[1] + '\t' + fields[2] + '\t' + cast_QE + '\n')
                    
                if len(cast_QQ) > 0:
                    fQQout.write(fields[0] + '\t' + fields[1] + '\t' + fields[2] + '\t' + cast_QQ + '\n')
      
     
        



''' ================================================= '''
'''    REMAP COLLAB NETWORKS BASED ON WHO HAS > 4     '''
''' ================================================= '''


'''
- film-director-collab-careers_QDIR   -->  
- film-director-collab-careers_QEVER  -->
- film-director-collab-careers_EVER   -->
'''



def create_full_nws(sample):


    ### THIS CODE HERE CREATES LGL EDGELIST FILE OUT OF THE COLLAB CARREER FILES !!!!
    ### LGL-S CREATED HERE

    ## 'QQ' : Qdirector - Qdirextor
    ## 'QE' : Qdirector - Qeveryone
    ## ''   :  


    ctype     = 'director'
    sam       = ''
    neighbrs  = {}

    tipusok   = ['-QQ']#, '-QE', '']


    if sample: sam = '_sample'
    dataout   = open('networks' + sam + '/networks_statistics.dat', 'w') 
    dataout.write('network\tyear\tnodes\tedges\n')


    for tipus in tipusok: 



        for yearLIMIT in [1990, 2000, 2010, 2020]:



            edges     = {}
            edge_dist = {}
            nodes     = set()


            outfolder = 'networks' + sam + '/' + ctype + tipus + '_' + str(yearLIMIT)


            if not os.path.exists(outfolder): os.makedirs(outfolder)


            root  = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers' + tipus + sam + '/'
            files = os.listdir(root)

            #fout = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_'      + str(yearLIMIT) + '.lgl', 'w')
            gout = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_rating_'  + str(yearLIMIT) + '.dat', 'w')
            hout = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_'    + str(yearLIMIT) + '.dat', 'w')

            n = len(files)

            x = 0
        
            for ind, fn in enumerate(files):
                      
                director = fn.split('_')[0]
                      

                #if ind == 2: break
            #    print ind, '/', n

                for line in open(root + fn):

                    fields = line.strip().split('\t') 
           
                    if len(fields) == 4:

                        movie, year, rating, cast = fields
                        if len(year) > 0:
                            year = min([float(y) for y in year.split('-')])

                        
                        year = float(year)

                        try:

                            rating = float(rating)                        

                            if year <= yearLIMIT and rating > 0.0:                        



                                # casts need to be handled as full graphs 
                                cast =  [ccc for ccc in list(set(cast.split(',') + [director])) if 'cast' not in ccc]


                                for c1 in cast:

                                    if c1 not in nodes:
                                        nodes.add(c1)

                                    for c2 in cast:
                                        if c1 != c2:

                                            edge = '\t'.join(sorted([c1, c2]))

                                             

                                            if edge not in edges:
                                                edges[edge]     = 1
                                                edge_dist[edge] = rating
                                            else:
                                                edges[edge]     += 1
                                                edge_dist[edge] += rating

                                            #if c1 not in neighbrs:
                                           #     neighbrs[c1] = [(c2,     movie, str(year), rating)]
                                           # else:
                                            #    neighbrs[c1].append((c2, movie, str(year), rating))
                            
                        except:
                            pass

                    
            '''aa = 0
            bb = 0      

            for user, nghb in neighbrs.items():
                
                fout.write('# ' + director + '\n')

                for n in nghb: 
                    fout.write(n[0] + ' '  + str(edges['\t'.join([user, n[0]])])    + '\n')
                    gout.write(user + '\t' + n[0] + '\t' + n[1] + '\t' + n[2] + '\t' + n[3] + '\n')
                    hout.write(user + '\t' + n[0] + '\t' + str(edges['\t'.join(sorted([user, n[0]]))]) + '\n')
                  
                    aa += 1
                    bb += 1

            '''
          
    
        
            dataout.write(ctype + tipus + '\t' + str(yearLIMIT) + '\t' + str(len(nodes)) + '\t' + str(len(edges)) + '\n')

            print 'Parsing  ', len(edges)


            f = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_gephi_edges' + str(yearLIMIT) + '.dat', 'w')
            f.write('Source'+'\t'+'Target'+'\t'+'Weight'+'\t'+'Type'+'\n')      


            for e, v in edges.items():

                if 'nm0797928\tnm0522327' == e:
                    print 'hello'

                gout.write(e + '\t' + str(edge_dist[e]) + '\n')
                hout.write(e + '\t' + str(v)            + '\n')              
                #f.write(e + '\t' + str(v) + '\t' + 'undirected' + '\n')  

            f.close()
            

            ''' PROB ADDING THE DICT OF BEST PRODUCTS HERE AND WRITE THEM AS NODE ATTRIBUTES '''

            #g = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_gephi_nodes.dat', 'w')
            #g.write('ID' + '\t' + 'Label' +'\t'+ 'House'+ '\n')    
            #for n, h in list(nodes):
            #    g.write(n + '\t' + n + '\t' + h + '\n')
            #g.close()


            #fout.close()
            gout.close()
            hout.close()
            




        dataout.close()



#''' ================================================= '''
#'''    PARSE AS AN IGRAPH FILE AND CALC A FEW BASIC MEASURES...MEASURE TIME ON CNS2     '''
#''' ================================================= '''


def create_igraphnw(sample):


    ctype     = 'director'
    sam       = ''
    neighbrs  = {}

    tipusok   = ['-QQ']#, '-QE', '']

    print tipusok

    if sample: sam = '_sample'

    for tipus in tipusok: 


        for yearLIMIT in [1990, 2000, 2010, 2020][0:1]:


            print 'Parsing the network of year ', yearLIMIT

            edges = {}
            nodes = set()


            outfolder = 'networks' + sam + '/' + ctype + tipus + '_' + str(yearLIMIT)


            print outfolder
            if not os.path.exists(outfolder): os.makedirs(outfolder)


            root  = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers' + tipus + sam + '/'
            files = os.listdir(root)

            #filename = outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_'      + str(yearLIMIT) + '.lgl'
            gilename = outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_rating_' + str(yearLIMIT) + '.dat'
            hilename = outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_list_'   + str(yearLIMIT) + '.dat'


            #G         = Graph.Read_Lgl(filename, names = True, weights = True, directed = False)
            G            = Graph.Read_Ncol(hilename, weights = True, directed = False) 
            edge_ratings = {}

            for line in open(gilename):
                source, target, rating = line.strip().split('\t')

                rating = float(rating)
                edge   =  '\t'.join([source, target])
  
                edge_ratings[edge] = rating



            print 'igraph  ', len(G.es()),len(edge_ratings)

            

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
            print 'degree   ', time.time() - t1


            t1 = time.time()
            strength     = G.strength(                      weights='weight')
            print 'strengthes   ', time.time() - t1


            t1 = time.time()
            ratings     = G.strength(                      weights='ratings')
            print 'ratings   ', time.time() - t1


            '''t1 = time.time()
            betweenness  = G.betweenness(                   weights='weight')
            print 'betweennesses   ', time.time() - t1


            t1 = time.time()
            clustering    = G.transitivity_local_undirected( weights='weight')
            print 'clusterings   ', time.time() - t1


            t1 = time.time()
            pagerank      = G.pagerank(                      weights='weight')
            print 'pageranks   ', time.time() - t1


            t1 = time.time()
            eigenvector   = G.eigenvector_centrality(        weights='weight')
            print 'eigenvectors   ', time.time() - t1
            '''




            node_centralities = {}
            
           

            for i in range(len(G.vs)):

                node = G.vs[i]['name']


                node_centralities[node] = { 
                                            'degree'   : degree[i],
                                            'strength' : strength[i], 
                                            'ratings'  : ratings[i]}


            df_centr = pd.DataFrame.from_dict(node_centralities, orient = 'index')

            print df_centr.head()

            df_centr.to_csv(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_NODE_CENTRALITIES_' + str(yearLIMIT) + '.dat', sep = '\t', index = True)





            #t1 = time.time()
            #closenesses    = G.closeness(                     weights='weight')
            #print 'closenesses   ', time.time() - t1


            #t1 = time.time()
            #constraint     = G.constraint(                    weights='weight') 
            #print 'constraint   ', time.time() - t1       



  



    
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















