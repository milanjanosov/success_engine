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
import matplotlib.pyplot as plt
from shutil import copyfile
import gzip
import math
from igraph import Graph
import networkx as nx
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr 
import numpy as np
sys.path.append("./backboning")
import backboning



def get_network_edge_list():

    nodes = set()
    edges = {}
    gen   = 0


    nodes_folder = 'networks/FULL_GRAPH_NODES'
    edges_folder = 'networks/FULL_GRAPH_EDGES'

    if not os.path.exists(nodes_folder): os.makedirs(nodes_folder)
    if not os.path.exists(edges_folder): os.makedirs(edges_folder)



    for jind, line in enumerate(open('ALL_movies_casts.dat')):

        if jind == 2000: break

        fields = line.strip().split('\t')
        movie  = fields[0]
        cast   = sorted(fields[1:])

        for ind, c1 in enumerate(cast):

            for c2 in cast[ind+1:]:

                edge = '\t'.join(sorted([c1, c2]))

                nodes.add(c1)
                nodes.add(c2)

                '''if edge not in edges:
                    edges[edge] = 1
                else:
                    edges[edge] += 1
                '''



        if jind % 100 == 0: 

            print 'Processing cast...    ', jind, '/703216'

            print 'Write nodes...'
            fout = open(nodes_folder + '/nodes_' + str(gen), 'w')

            print nodes_folder + '/nodes_' + str(gen)
            for n in list(nodes):
                fout.write(n + '\n')
            fout.close()

            nodes = None


            print 'Write edges...'
            fout = open(edges_folder + '/edges_' + str(gen), 'w')
            for e, w in edges.items():
                fout.write(e + '\t' + str(w) + '\n')
            fout.close()

            

            nodes = set()
            edges = {}
            gen  += 1






def get_network_edge_list_directors():

    nodes = set()
    edges = {}


    directors = set([f.split('_')[0] for f in os.listdir('simple-careers/film-director-simple-careers')])



    for jind, line in enumerate(open('ALL_movies_casts.dat')):

        #if jind == 2000: break


        if jind % 1000 == 0: 
            print 'directors nw   ', jind

        fields = line.strip().split('\t')
        movie  = fields[0]
        cast   = sorted(fields[1:])

        for ind, c1 in enumerate(cast):

            for c2 in cast[ind+1:]:

                if c1 in directors and c2 in directors:            

                    edge = '\t'.join(sorted([c1, c2]))
                    #print edge

                    nodes.add(c1)
                    nodes.add(c2)

                    if edge not in edges:
                        edges[edge] = 1
                    else:
                        edges[edge] += 1
              


    print 'Writing nodes...'
    fout = open('networks/FULL_GRAPH_nodes_dircetors.dat', 'w')
    for n in list(nodes):
        fout.write(n + '\n')
    fout.close()



    print 'Writing edges...'
    ne = 0
    fout = open('networks/FULL_GRAPH_edges_directors_1.dat', 'w')
    for e, w in edges.items():
        if w > 1.0:
            fout.write(e + '\t' + str(w) + '\n')
            ne += 1
    fout.close()



    print 'nodes:  ', len(nodes)
    print 'edges:  ', ne



def merge_nodes_and_edges():


    nodes_folder = 'networks/FULL_GRAPH_NODES'
    edges_folder = 'networks/FULL_GRAPH_EDGES'


    node_files = os.listdir(nodes_folder)
    edge_files = os.listdir(edges_folder)


    nodes = set()
    nnn   = len(node_files)

    for ind, n in enumerate(node_files):
        print 'nodes   ', ind, '/', nnn
        for line in open(nodes_folder + '/' + n):
            nodes.add(line.strip())


    fout = open('networks/FULL_GRAPH_nodes.dat', 'w')
    for n in list(nodes):
        fout.write(n + '\n')
    fout.close()

    nodes = None


    edges = {}
    nnn   = len(edge_files)

    for ind, e in enumerate(edge_files):

        print 'edges   ', ind, '/', nnn
        for line in open(edges_folder + '/' + e):
            edge, weight = line.strip().rsplit('\t', 1)
            weight = float(weight)

            if edge not in edges:
                edges[edge] = weight
            else:
                edges[edge] += weight   


    fout = open('networks/FULL_GRAPH_edges_1.dat', 'w')
    for e, w in edges.items():
        if w > 1.0:
            fout.write(e + '\t' + str(w) + '\n')
            del edges[e]
    fout.close()

    edges = None




def get_nodes(table):
    
    return len(set(list(table.src) + list(table.trg)))

def get_backboned_edgelists():


    edgefile   = 'networks/FULL_GRAPH_edges_directors_1.dat'
    folderout  = 'networks/backboning/'

    if not os.path.exists(folderout): os.makedirs(folderout)


    for ind, line in enumerate(open(edgefile)):

        if 'src' and 'trg' in line:
            print 'FAASZ'
        else:
            with file(edgefile, 'r') as original: data = original.read()
            with file(edgefile, 'w') as modified: modified.write('src\ttrg\tnij\n'  + data)


        if ind == 0: break




  

    real_table = pd.read_csv(edgefile, sep = "\t")
    nedges_o   = len(real_table)
    nnodes_o   = get_nodes(real_table)


    fstatout = open(folderout + 'backboned_size_stats_insitu.dat', 'w')
    fstatout.write('type\tparam\tnnodes\tnedges\n')
    fstatout.write('original\t-\t' + str( nnodes_o ) + '\t' + str( nedges_o ) + '\n' )




    nc = 1
    df = 0.0

    for nc in [-20, -10, -5, -4, -3, -2, -1, 0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1,  0, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:

        t1        = time.time()
        table_nc  = backboning.noise_corrected(real_table,   undirected = True)
        bb_neffke = backboning.thresholding(table_nc, nc)
        nodes_nc  = get_nodes(bb_neffke)
        edges_nc  = len(bb_neffke)
        bb_neffke.to_csv(folderout  + 'nc_backboned_' + str(nc),  sep = '\t', index = False)

        fstatout.write('nc      \t' + str(nc) + '\t' + str( nodes_nc ) + '\t' + str( edges_nc ) + '\n' )

        print nc, '\tnodes: ', nodes_nc, '\t\tedges: ', edges_nc, '\t\tdensity: ', round(2*float(edges_nc)/(nodes_nc*(nodes_nc-1)),5), '\t\ttime: ', round(time.time()-t1), '\n'


    for df in [round(k,4) for k in list(np.arange(0,1,0.05))]:

        t1            = time.time()
        table_df      = backboning.disparity_filter(real_table,  undirected = True)
        bb_vespignani = backboning.thresholding(table_df, df)
        nodes_df      = get_nodes(bb_vespignani)
        edges_df      = len(bb_vespignani)
        bb_vespignani.to_csv(folderout + 'df_backboned_' + str(df),  sep = '\t', index = False)

        fstatout.write('df      \t' + str(df) + '\t' + str( nodes_df ) + '\t' + str( edges_df ) + '\n' )

        print 'nodes: ', nodes_df, '\t\tedges: ', edges_df, '\ttime: ', round(time.time()-t1), '\n'


    # for nc_threshold in [5000, 2000, 1000, 500, 100]:
    fstatout.close()
    
  




def get_backbonestats():

    files    = os.listdir('networks/backboning')
    files_df = [f for f in files if 'df' in f]
    files_nc = [f for f in files if 'nc' in f]




    fstatout = open('networks/backboning/backboned_size_stats_nc.dat', 'w')

    for fn in files_nc:

        nodes = set()

        for ind, line in enumerate(open('networks/backboning/' + fn)):
            
            src, trg, nij, score = line.strip().split('\t')
            nodes.add(src)
            nodes.add(trg)
            param = fn.split('_')[2]

        fstatout.write( param + '\t'  + str(len(nodes)) + '\t' + str(ind) + '\n')


    fstatout.close()




    fstatout = open('networks/backboning/backboned_size_stats_df.dat', 'w')

    for fn in files_df:

        nodes = set()

        for ind, line in enumerate(open('networks/backboning/' + fn)):
            
            src, trg, nij, score = line.strip().split('\t')
            nodes.add(src)
            nodes.add(trg)
            param = fn.split('_')[2]

        fstatout.write( param + '\t'  + str(len(nodes)) + '\t' + str(ind) + '\n')


    fstatout.close()
    




def rescale(vector):
    v = StandardScaler().fit_transform(np.asarray(vector).reshape(-1, 1) ) 
    return preprocessing.quantile_transform(v, output_distribution = 'normal')


def get_reg_stuff(impact, network, title):

    impact, network = zip(*[(impact[i], network[i]) for i in range(len(network)) if not np.isnan(network[i])  ])

    impact  = rescale(impact)
    network = rescale(network)

    print title
    reg = LinearRegression().fit(impact, network)
    print reg.score(impact, network)
    print reg.coef_, '\n'


def get_impact_nw(directors_best, network):
    
    impact = []
    nwmeas = []

    for d, Imax in directors_best.items():
        if d in network:
            impact.append(Imax)
            nwmeas.append(network[d])

    return impact, nwmeas





def add_df_meas(meas, tipus):

    df = pd.DataFrame(meas.items(), columns = ['name', tipus])
    df.index = df.name
    df = df.drop(columns = ['name'])    
    
    return df


def compare(measname, meas_nx, meas_ig, GC):

    same_full    = 0.0
    nosame_full  = 0.0
    same_giant   = 0.0
    nosame_giant = 0.0

    for ind, (name, value_nx) in enumerate(meas_nx.items()):


        value_ig = round(meas_ig[name],15)
        value_nx = round(value_nx,     15)


        if value_ig == value_nx:
            same_full += 1.0
        else:
            nosame_full += 1.0

        if name in GC:

            # to debug closeness
            #print value_ig, value_nx,  value_ig / value_nx

            if value_ig == value_nx:
                same_giant += 1.0
            else:
                nosame_giant += 1.0


    print measname, '\tnodes: ', len(meas_nx), len(meas_ig), '\tfullnw: ', round(same_full / (same_full + nosame_full),3), '\tgiantC_ ', round(same_giant / (same_giant + nosame_giant),3)




def get_centralities(compare):


    params = [5000, 2000, 1000, 500, 100, 50, 40, 30, 20, 10, 5, 4, 3, 2, 1, 0]

    folderout = 'networks/backboning_centralities/'
    if not os.path.exists(folderout):
        os.makedirs(folderout)
       

    time_nx = []
    time_ig = []
    ftimes  = open(folderout + 'compare_comp_time.dat', 'w')

    ftimes.write('nc\tt_nx\tt_ig\n')

    for nc in params[0:4]:

      

        ''' NETWORKX '''

        edges_nx = []
        t1       = time.time()

        print 'Parse edges'
        for ind, line in enumerate(open('networks/backboning/nc_backboned_' + str(nc))):
            if 'nij' not in line:
                e1, e2, w, sign = line.strip().split('\t')
                edges_nx.append((e1, e2, {'weight' : float(w)}))

           
        G_nx  = nx.Graph()
        G_nx.add_edges_from(edges_nx)
        GC_nx = [c for c in sorted(nx.connected_components(G_nx), key=len, reverse=True)][0]


        print nc, '\tGet NC degrees'
        degrees_nx = add_df_meas(nx.degree_centrality(G_nx), 'degree_nx')

        print nc, '\tGet NC clustering'
        clusterings_nx = add_df_meas(nx.clustering(G_nx), 'clustering_nx')

        print nc, '\tGet NC pageranks'
        pageranks_nx   = add_df_meas(nx.pagerank(G_nx), 'pagerank_nx')

        print nc, '\tGet NC betweenness'
        betweennesses_nx   = add_df_meas(nx.betweenness_centrality(G_nx), 'betweenness_nx')
  
        print nc, '\tGet NC closeness'
        closenesses_nx   = add_df_meas(nx.closeness_centrality(G_nx), 'closeness_nx')

        #print 'Get eigenvector'
        #eigenvectors_nx   = add_df_meas(nx.eigenvector_centrality(G_nx), 'eigenvector_mx')

        print nc, '\tGet NC constraint'
        constraints_nx   = add_df_meas(nx.constraint(G_nx), 'constraint_nx')
        
        df_nx = degrees_nx.merge(clusterings_nx, left_index=True,  right_index=True)
        df_nx = df_nx.merge(pageranks_nx,        left_index=True,  right_index=True)
        df_nx = df_nx.merge(betweennesses_nx,    left_index=True,  right_index=True)
        df_nx = df_nx.merge(closenesses_nx,      left_index=True,  right_index=True)
        df_nx = df_nx.merge(constraints_nx,      left_index=True,  right_index=True)

        t2   = time.time()
        t_nx = t2-t1
        time_nx.append(t_nx)

        print 'Time for NX:  ', round(t_nx , 2   ), ' s'



        ''' IGRAPH '''

        # get the igraph network
        t1       = time.time()
        ftempname = 'tempfile_nc_backboned' + str(nc)
        ftemp = open(ftempname, 'w')
        for line in open('networks/backboning/nc_backboned_' + str(nc)):
            if 'src' not in line:
                ftemp.write('\t'.join(line.strip().split('\t')[0:3]) + '\n' ) 
        ftemp.close()
        G_ig = Graph.Read_Ncol(ftempname, weights = True, directed=False)
        os.remove(ftempname)
        

        # get degree thats matching
        # nw computes degree centrality, which is the k/(N-1), while ig computes k  
        # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.algorithms.centrality.degree_centrality.html
        print '\n', nc, '\tGet IG degrees'
        degrees_ig = {}
        G_ig.vs['degree_ig'] =  G_ig.degree()
        N = len(G_ig.vs['degree_ig'])
        for v in G_ig.vs():
            degrees_ig[v['name']] = v['degree_ig']/float(N-1)
        

        # get the matching clustering
        # when nw gives 0 for clustering, ig gives nan
        print nc, '\tGet IG clustering'
        clusterings_ig = {}
        G_ig.vs['clustering_ig'] = G_ig.transitivity_local_undirected( weights = None)
        for v in G_ig.vs():
            if np.isnan(v['clustering_ig']):
                v['clustering_ig'] = 0
            clusterings_ig[v['name']] = v['clustering_ig']
        

        # match betweenness
        # nx gives the normalzed betweenness, while igraph gives the raw value. normalization vactor is
        # Bnorm = =  (n*n-3*n+2) / 2.0                      http://igraph.org/r/doc/betweenness.html    
        print nc, '\tGet IG betweenness'
        G_ig.vs['betweenness_ig']  = G_ig.betweenness( weights = None)
        betweennesses_ig = {}
        n = len(G_ig.vs())
        for  v in G_ig.vs():
            Bnormalizer =  (n*n-3*n+2) / 2.0
            betweennesses_ig[v['name']] = v['betweenness_ig']/Bnormalizer
           
        
        # comparing closeness:
        # NX: If the graph is not completely connected, this algorithm computes the closeness centrality for each connected part separately.
        #    https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.closeness_centrality.html
        # IG: If the graph is not connected, and there is no path between two vertices, the number of vertices is used instead the length of the geodesic. This is always longer than the longest possible geodesic.
        # http://igraph.org/python/doc/igraph.GraphBase-class.html#closeness
        print nc, '\tGet IG closeness'
        closenesses_ig = {}
        G_ig.vs['closeness_ig']  = G_ig.closeness( weights = None, normalized = False )
        for v in G_ig.vs():
            closenesses_ig[v['name']] = v['closeness_ig']
        
  
        # get matching pagerank values
        # they match, besides some numerical things
        print nc, '\tGet IG pageranks'
        pageranks_ig = {}
        G_ig.vs['pagerank_ig'] = G_ig.pagerank( weights = None)
        for v in G_ig.vs():
            pageranks_ig[v['name']] = v['pagerank_ig']
        
                
        # constrains match well
        print nc, '\tGet IG constraint'
        constraints_ig = {}
        G_ig.vs['constraint_ig']  = G_ig.constraint( weights = None )
        for v in G_ig.vs():
            constraints_ig[v['name']] = v['constraint_ig']

   
        # G_ig.vs['eigenvector_ig']  = G_ig.eigenvector_centrality( weights = None )


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

        t2   = time.time()
        t_ig = t2 - t1
        time_nx.append(t_ig)

        print 'Time for IG:  ', round( t_ig , 2  ), ' s\n\n'



        df_nx.to_csv(folderout + 'nc_backboned_centralities_NX_' + str(nc), na_rep='nan')
        df_ig.to_csv(folderout + 'nc_backboned_centralities_IG_' + str(nc), na_rep='nan')
      

        if compare:
            compare('degree    ',  dict(degrees_nx.degree_nx),            degrees_ig,       GC_nx)
            compare('clustering',  dict(clusterings_nx.clustering_nx),    clusterings_ig,   GC_nx)
            compare('pagerank   ', dict(pageranks_nx.pagerank_nx),        pageranks_ig,     GC_nx)
            compare('betweenness', dict(betweennesses_nx.betweenness_nx), betweennesses_ig, GC_nx)
            compare('closeness',   dict(closenesses_nx.closeness_nx),     closenesses_ig,   GC_nx)
            compare('constraint',  dict(constraints_nx.constraint_nx),    constraints_ig,   GC_nx)


        
        ftimes.write(str(nc) + '\t' + str(t_nx) + '\t' + str(t_ig) + '\n')
    ftimes.close()



def do_analysis():


    print 'Creating NX Graph'


   




    ''' SAVE CENTRALITIES AS INDIVIDUAL FEATUREs, CSV '''




  

    
    directors, best = zip(*[line.strip().split('\t') for line in open('directors_Imax.dat')])
    directors_best  = {directors[i] : float(best[i]) for i in range(len(best))}


    impact, nwmeasD  = get_impact_nw(directors_best, degrees)
    get_reg_stuff(impact, nwmeasD, 'degree')


    impact, nwmeasC  = get_impact_nw(directors_best, clusterings)
    get_reg_stuff(impact, nwmeasC, 'clustering')


    impact, nwmeasP  = get_impact_nw(directors_best, pageranks)
    get_reg_stuff(impact, nwmeasP, 'pagerank')



#get_reg_stuff(Is,Cs, 'clustering')
#get_reg_stuff(Is,Ss, 'strength')
#get_reg_stuff(Is,Ps, 'pagerank')


if sys.argv[1] == 'edges':
    get_network_edge_list()
elif sys.argv[1] == 'merge':
    merge_nodes_and_edges()
elif sys.argv[1] == 'backbone':
    get_backboned_edgelists()
elif sys.argv[1] == 'backbonestats':
    get_backbonestats()
elif sys.argv[1] == 'results':
    do_analysis()
elif sys.argv[1] == 'directors':
    get_network_edge_list_directors()
elif sys.argv[1] == 'centralities':
    get_centralities(compare = False)










 

                         