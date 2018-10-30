import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

def get_centrality_careers():

    field    = 'film'
    ctype    = 'director'
    tipus    = 'QQ'
    infolder = 'collab-careers/' + field + '-' + ctype + '-collab-careers-' + tipus
    weight   = 'jaccard'
    Qdir     = set([line.strip() for line in open('users_types/Q_' + field + '_' + ctype + '_namelist.dat')])



    ''' get the centralities '''

    centralityfiles = os.listdir('networks/director')
    beweennesses    = {}
    clusterings     = {}
    degrees         = {}
    pageranks       = {}
    strengths       = {}


    for centr in centralityfiles:
        
        year      = centr.split('_')[1]

        if int(year) > 2000:

            print year

            datafile  = 'networks/director/' + centr + '/director_' + year + '_jacc/ALLdirector_director__NODE_CENTRALITIES_jaccard_' + year + '_weighted.dat'
            dataframe = pd.read_csv(datafile, sep = '\t', index_col=0)
     
            degree     = dict(dataframe.degree)
            strength   = dict(dataframe.strength)
            clustering = dict(dataframe.clustering)
            pagerank   = dict(dataframe.pagerank)


            for name in degree.keys():

                if name in Qdir:

                    if name not in degrees:     degrees[name]     = {}
                    if name not in strengths:   strengths[name]   = {}
                    if name not in clusterings: clusterings[name] = {}
                    if name not in pageranks:   pageranks[name]   = {}

                    degrees[name][year]      = degree[name]
                    strengths[name][year]    = strength[name]
                    clusterings[name][year]  = clustering[name]
                    pageranks[name][year]    = pagerank[name]


  



    ''' get the centrality careers '''

    files     = os.listdir(infolder)
    outfolder = 'centrality-careers/centrality_careers_directors_ALL'
    nnn       = len(files)
    Qnames    = degree.keys()

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)


    for ind, fn in enumerate(files):

        print ind, '/', nnn


        name = fn.split('_')[0]

#        if name in Qdir:

        if name == 'nm0996502':
    


            fout = open(outfolder + '/' + name + '_centrality_careers.dat', 'w')

            


            for line in open(infolder + '/' + fn):      
                movie, year, impact, cast = line.strip().split('\t')

                print name, year, impact, movie

      
                if year in strengths[name]:
                    strength = str(strengths[name][year])
                else:
                    strength = '0'


                if year in clusterings[name]:
                    clustering  = str(clusterings[name][year])
                else:
                    clustering = '0'


                if year in degrees[name]:
                    degree = str(degrees[name][year])
                else:
                    degree = '0'


                if year in pageranks[name]:
                    pagerank = str(pageranks[name][year])
                else:
                    pagerank = '0'





                fout.write(name + '\t' + movie + '\t' + year + '\t' + impact + '\t' + strength + '\t' + clustering + '\t' + degree + '\t' + pagerank + '\n')
                #except: 
                #    pass


            fout.close()
   
    

#
def plot_hist(ax, data, xlabel, title):

    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_title(title, fontsize = 14)
    data = [x for x in data if str(x) != 'nan']

    y, x, _ = ax.hist(data, bins = 30)

    #y = list(y)
    #x = list(x)
    #ax.axvline(x[y.index(max(y))], color='r', linestyle='dashed', linewidth=3)
    ax.set_xlim([-1.0,1.0])
    ax.axvline(np.mean(data), color='r', linestyle='dashed', linewidth=3)



def get_correlations():

    infolder = 'centrality-careers/centrality_careers_directors_ALL'
    files    = os.listdir(infolder)
    nnn      = len(files)

    BcP = []
    BcS = []
    BcK = []

    DcP = []
    DcS = []
    DcK = []

    PcP = []
    PcS = []
    PcK = []


    for ind, fn in enumerate(files):
        
        impacts = []
        betws   = []
        clusts  = []
        degs    = []
        prs     = []

        print ind, '/', nnn

        for line in open(infolder + '/' + fn):

            name, movie, year, impact, betweenness, clustering, degree, pagerank = line.strip().split('\t')

            if impact != 'None':

                impacts.append( float( impact ) )
                betws.append( float( betweenness ) )
                #clusts.append( float( clustering ) )
                degs.append( float( degree ) )
                prs.append( float( pagerank ) )

        if len(impacts) > 10:

            BcP.append( scipy.stats.pearsonr(impacts,   betws)[0] ) 
            BcS.append( scipy.stats.spearmanr(impacts,  betws)[0] )
            BcK.append( scipy.stats.kendalltau(impacts, betws)[0] )

            DcP.append( scipy.stats.pearsonr(impacts,   degs)[0] ) 
            DcS.append( scipy.stats.spearmanr(impacts,  degs)[0] )
            DcK.append( scipy.stats.kendalltau(impacts, degs)[0] )

            PcP.append( scipy.stats.pearsonr(impacts,   prs)[0] ) 
            PcS.append( scipy.stats.spearmanr(impacts,  prs)[0] )
            PcK.append( scipy.stats.kendalltau(impacts, prs)[0] )



    f, ax = plt.subplots(3,3, figsize = (15,12))

    plot_hist(ax[0,0], BcP, 'betwenness', 'pearsonr')
    plot_hist(ax[0,1], BcS, 'betwenness', 'spearmanr')
    plot_hist(ax[0,2], BcK, 'betwenness', 'kendalltau')

    plot_hist(ax[1,0], DcP, 'degree', '')
    plot_hist(ax[1,1], DcS, 'degree', '')
    plot_hist(ax[1,2], DcK, 'degree', '')

    plot_hist(ax[2,0], PcP, 'pagerank', '')
    plot_hist(ax[2,1], PcS, 'pagerank', '')
    plot_hist(ax[2,2], PcK, 'pagerank', '')


    plt.tight_layout()
    plt.savefig('correl.png')
    plt.show()




if sys.argv[1] == 'get_careers':
    get_centrality_careers()
elif sys.argv[1] == 'get_correlations':
    get_correlations()





