

import os
import time
import sys
from shutil import copyfile
import gzip
import os
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


    for tipus in tipusok: 


        edges = {}
        nodes = set()

        if sample: sam = '_sample'
        outfolder = 'networks' + sam + '/' + ctype + tipus

        print outfolder
        if not os.path.exists(outfolder): os.makedirs(outfolder)


        root  = 'collab-careers' + sam + '/film-' + ctype + '-collab-careers' + tipus + sam + '/'
        files = os.listdir(root)

        fout = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges.lgl',     'w')
        gout = open(outfolder + '/Q' + ctype + '_' + ctype + tipus + '_edges_att.dat', 'w')

        n = len(files)

        x = 0
    
        for ind, fn in enumerate(files):
                  
            director = fn.split('_')[0]
                  
            print ind, '/', n

            for line in open(root + fn):

                fields = line.strip().split('\t') 
       
                if len(fields) == 4:

                    movie, year, rating, cast = fields
         
                    # casts need to be handled as full graphs 
                    cast =  [ccc for ccc in list(set(cast.split(',') + [director])) if 'cast' not in ccc]

                    x += len(cast)*len(cast)/2

                    '''for c1 in cast:

                        if c1 not in nodes:
                            nodes.add(c1)

                        for c2 in cast:
                            if c1 != c2:

                                edge = '_'.join(sorted([c1, c2]))
                                if edge not in edges:
                                    edges[edge] = 1
                                else:
                                    edges[edge] += 1

                                if c1 not in neighbrs:
                                    neighbrs[c1] = [(c2, movie, year, rating)]
                                else:
                                    neighbrs[c1].append((c2, year, rating, movie))
                    '''
            #if ind == 0: break
                
        print x#len(neighbrs), len(nodes), len(edges)


        '''for user, nghb in neighbrs.items():
            
            fout.write('# ' + director + '\n')

            for n in nghb: 

                fout.write(n[0] + '\n')
                gout.write(user + '\t' + n[0] + '\t' + n[1] + '\t' + n[2] + '\t' + n[3] + '\n')


                
        fout.close()
        gout.close()
        '''


    '''f = open('edges_s7.dat', 'w')
    f.write('Source'+'\t'+'Target'+'\t'+'Weight'+'\t'+'Type'+'\n')      
    for e, v in edges_two.items():
        f.write(e[0]+'\t'+e[1]+'\t'+str(math.log(v)) + '\t' + 'undirected'+'\n')  
    f.close()
    
    g = open('nodes_s7.dat', 'w')
    g.write('ID' + '\t' + 'Label' +'\t'+ 'House'+ '\n')    
    for n, h in names.items():
        g.write(n+'\t'+n+'\t'+h+'\n')
    g.close()
    '''    



#''' ================================================= '''
#'''    PARSE AS AN IGRAPH FILE AND CALC A FEW BASIC MEASURES...MEASURE TIME ON CNS2     '''
#''' ================================================= '''


#def get_temporal_networks():




    
if __name__ == '__main__':         


    if sys.argv[1] == 'basic_stat':
        get_users_numbers_stats()
    
    elif sys.argv[1] == 'merge_users':
        get_merged_user_lists()     
     
    elif sys.argv[1] == 'remap_collab_careers':
        remapping_collab_careers(sample = True)


    elif sys.argv[1] == 'get_full_network':
        create_full_nws(sample = True)
    
        



### counts the user numbers
### get_users_numbers_stats()
### get_merged_user_lists()


#remapping_collab_careers()















