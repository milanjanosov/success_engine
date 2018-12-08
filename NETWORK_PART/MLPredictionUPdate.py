import numpy as np
import os
import gzip
import sys
from collections import Counter
import math
import pandas as pd
import random
import scipy.stats
from numpy.linalg import norm
from multiprocessing import Process
from scipy.stats import spearmanr
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from multiprocessing import Process




''' ============================================================ '''
'''  -----------------     HELPER FUNCTIONS    ----------------- ''' 
''' ============================================================ '''


def get_centralities(dirid, measures, column = 1):

    centralities = {}

    for line in open('NEWTemporal/4_directors_centralities/' +dirid+'.dat'):
        
        fields = line.strip().split('\t')
        fields = [float(f) for f in fields]
        centralities[fields[0]] = fields[column]
        
    centralities_list = sorted(centralities.iteritems(), key=itemgetter(0), reverse=False)    
        
    return centralities, centralities_list 

def get_year(year):

    if len(year) == 4: return int(year)
    else: return int(year.split('-')[0])


def get_career_data(dirid, centralities):
    
    x = []
    y = []


    for line in gzip.open('simple-careers/film-director-simple-careers/' + dirid + '_director_simple_career.gz'): 
    
        if 'year' not in line:
            fields = line.strip().split()
            
            if len( fields[1] ) > 0:
                    
                year = get_year(fields[1])
                if 'None' != fields[3]:

                    year   = float(get_year(fields[1]))
                    impact = float(fields[3])

                    if year in centralities:
                        x.append(year)
                        y.append(impact)

    if len(x) > 0:
        y, x = zip(*sorted([ (y[ijk], x[ijk]) for ijk in range(len(x)) ], key=lambda tup: tup[1]))        

        Istar  = max(y)
        Nstar  = y.index(Istar)    
        yR     = list(y)
        random.shuffle(yR)    
        IstarR = max(yR)
        NstarR = yR.index(IstarR)    

        return Istar, Nstar, NstarR, y
    
    else:
        return 0, 0, 0, 0





''' ============================================================= '''
'''  -----------------  GET SPECIFIC CFEATURES  ----------------- ''' 
''' ============================================================= '''


def get_centr_features(TauLimit, dirids, cumulative, measureid):

    centralityFeatures  = pd.DataFrame()

    for dirid in dirids:

        column  = {}

        centralities_d, centralities  = get_centralities(dirid, measures, column =measureid)
        Istar, Nstar, NstarR, impacts = get_career_data(dirid, centralities_d)

        
        if (Nstar - abs(TauLimit)) >= 0 and (-1+len(centralities) - abs(TauLimit)) >= Nstar:
          
            if cumulative:

                if TauLimit < 0:
                    interval = [TauLimit, 1]
                else:
                    interval = [0, TauLimit +1]


                for i in range(interval[0], interval[1]):
                    column[i] = centralities[Nstar + i][1]
            else:
                column[TauLimit] = centralities[Nstar + TauLimit][1]
            
                    
            column['Istar']  = Istar

            if Istar == 0: Istar = 1
            column['logIstar']  = math.log(Istar)

            df_column  = pd.DataFrame({dirid : column}).T
            centralityFeatures  = centralityFeatures.append(df_column, ignore_index=True)         
        
            
    centralityFeatures_Q = centralityFeatures         
            
    centralityFeatures_Q['IstarQ']     = pd.qcut(centralityFeatures_Q['Istar'],4, ['q1','q2','q3','q4'])
    centralityFeatures_Q['logIstarQ']  = pd.qcut(centralityFeatures_Q['logIstar'],4, ['q1','q2','q3','q4'])
    centralityFeatures_Q  = centralityFeatures_Q.dropna()    


    labels =  [str(10*(i+1))+ '%' for i in range(10)]
    centralityFeatures_T = centralityFeatures         

    
    centralityFeatures_T['IstarQ']    = pd.qcut(centralityFeatures_T['Istar'],10,  labels)    
    centralityFeatures_T['IstarQ']    = centralityFeatures_T['IstarQ'].replace('100%', 'top10')
    centralityFeatures_T['IstarQ']    = centralityFeatures_T['IstarQ'].replace(r'^.*%.*$', 'bottom90',  regex=True)
    centralityFeatures_T['logIstarQ'] = pd.qcut(centralityFeatures_T['logIstar'],10,  labels)
    centralityFeatures_T['logIstarQ'] = centralityFeatures_T['logIstarQ'].replace('100%', 'top10')
    centralityFeatures_T['logIstarQ'] = centralityFeatures_T['logIstarQ'].replace(r'^.*%.*$', 'bottom90',  regex=True)

    centralityFeatures_T  = centralityFeatures_T.dropna()    

    centralityFeatures_top    = centralityFeatures_T[centralityFeatures_T.IstarQ=='top10']
    centralityFeatures_bottom = centralityFeatures_T[centralityFeatures_T.IstarQ=='bottom90'].sample(len(centralityFeatures_top))
    
    centralityFeatures_T = centralityFeatures_bottom.append(centralityFeatures_top, ignore_index=True)   

    centralityFeatures_Qbin = centralityFeatures_Q[centralityFeatures_Q.IstarQ.isin(['q1', 'q4'])]
    
    return centralityFeatures_Q, centralityFeatures_Qbin, centralityFeatures_T





''' ============================================================= '''
'''  -----------------  GET SPECIFIC CFEATURES  ----------------- ''' 
''' ============================================================= '''


def get_combined_features(TauLimit, dirids, cumulative):

    centralityFeatures  = pd.DataFrame()
    measures = ['degree', 'clustering', 'pagerank', 'betweenness', 'constraint']  
    
    
    
    for dirid in dirids:

        column  = {}
        
        for jind, measure in enumerate(measures):

            centralities_d, centralities  = get_centralities(dirid, measures, column = jind + 1)
            Istar, Nstar, NstarR, impacts = get_career_data(dirid, centralities_d)


            if (Nstar - abs(TauLimit)) >= 0 and ( -1 + len(centralities) - abs(TauLimit)) >= Nstar:

                if cumulative:

                    if TauLimit < 0:
                        interval = [TauLimit, 1]
                    else:
                        interval = [0, TauLimit +1]

                    for i in range(interval[0], interval[1]):
                        column[str(i) + '_' + measure] = centralities[Nstar + i][1]

                else:
                   # print len(centralities), Nstar,  TauLimit
                    column[str(TauLimit) + '_' + measure] = centralities[Nstar + TauLimit]  [1]

                column['Istar']  = Istar

                if Istar == 0: Istar = 1
                column['logIstar']  = math.log(Istar)

                df_column  = pd.DataFrame({dirid : column}).T
                centralityFeatures  = centralityFeatures.append(df_column, ignore_index=True)         


   # print centralityFeatures_Q.head()

    centralityFeatures_Q = centralityFeatures         

    centralityFeatures_Q['IstarQ']     = pd.qcut(centralityFeatures_Q['Istar'],4, ['q1','q2','q3','q4'])
    centralityFeatures_Q['logIstarQ']  = pd.qcut(centralityFeatures_Q['logIstar'],4, ['q1','q2','q3','q4'])
    centralityFeatures_Q  = centralityFeatures_Q.dropna()    


    labels =  [str(10*(i+1))+ '%' for i in range(10)]
    centralityFeatures_T = centralityFeatures         


    centralityFeatures_T['IstarQ']    = pd.qcut(centralityFeatures_T['Istar'],10,  labels)    
    centralityFeatures_T['IstarQ']    = centralityFeatures_T['IstarQ'].replace('100%', 'top10')
    centralityFeatures_T['IstarQ']    = centralityFeatures_T['IstarQ'].replace(r'^.*%.*$', 'bottom90',  regex=True)
    centralityFeatures_T['logIstarQ'] = pd.qcut(centralityFeatures_T['logIstar'],10,  labels)
    centralityFeatures_T['logIstarQ'] = centralityFeatures_T['logIstarQ'].replace('100%', 'top10')
    centralityFeatures_T['logIstarQ'] = centralityFeatures_T['logIstarQ'].replace(r'^.*%.*$', 'bottom90',  regex=True)

    centralityFeatures_T  = centralityFeatures_T.dropna()    

    centralityFeatures_top    = centralityFeatures_T[centralityFeatures_T.IstarQ=='top10']
    centralityFeatures_bottom = centralityFeatures_T[centralityFeatures_T.IstarQ=='bottom90'].sample(len(centralityFeatures_top))

    centralityFeatures_T = centralityFeatures_bottom.append(centralityFeatures_top, ignore_index=True)   
    centralityFeatures_Qbin = centralityFeatures_Q[centralityFeatures_Q.IstarQ.isin(['q1', 'q4'])]
    

    return centralityFeatures_Q, centralityFeatures_Qbin, centralityFeatures_T





''' ============================================================= '''
'''  -----------------  PREDICTION FUNCTIONS    ----------------- ''' 
''' ============================================================= '''


def xgb_cl(data, Nest, CV, max_depth_ ,learning_rate_, subsample_):
      
    X = data.drop(columns = ['Istar', 'logIstar', 'IstarQ', 'logIstarQ'])
    y = list(data['logIstarQ'])
    accuracies = []
        
    for i in range(1):

        train_data, test_data, train_label, test_label =  train_test_split(X, y, test_size=.33, random_state=42)    
          
        model2       = xgb.XGBClassifier(n_estimators=Nest   , max_depth=max_depth_, learning_rate=learning_rate_, subsample=subsample_)
        train_model2 = model2.fit(train_data, train_label)
        pred2        = train_model2.predict(test_data)
        accuracies   += list(cross_val_score(train_model2, train_data, train_label, cv=CV))    

    return np.mean(accuracies), np.std(accuracies), len(X)



def NB_cl(data, Nest, CV):
    
    X = data.drop(columns = ['Istar', 'logIstar', 'IstarQ', 'logIstarQ'])
    y = list(data['logIstarQ'])


    accuracies = []

    for i in range(Nest):

        train_data, test_data, train_label, test_label =  train_test_split(X, y, test_size=.33, random_state=42)    

        name       = 'Naive Bayes' 
        classifier = GaussianNB()  
        clas       = classifier.fit(train_data, train_label)
        accuracies += list(cross_val_score(clas, train_data, train_label, cv=CV))       
            
    return np.mean(accuracies), np.std(accuracies), len(X)



def get_meas_prediction_results(TauLimit, Nest, CV, cumulative):

    # get the data feature file
    centralityFeatures_Q, centralityFeatures_Qbin, centralityFeatures_T = get_centr_features(TauLimit, dirids, cumulative, measureid = 1 + measures.index(measure))    


    # do optimized predictions
    for dataset, data in [('quartiles', centralityFeatures_Q), ('quartbinary', centralityFeatures_Qbin), ('topbottom', centralityFeatures_T)]:


        print measure, '\t', TauLimit, '\t', dataset

        bestacc = (0, 0)

        for sample in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            for rate in [0.01, 0.05, 0.1, 0.15, 0.2]:

                acc, err, N = xgb_cl(data, Nest, CV, 6 , rate, sample)
                if acc > bestacc[0]:
                    bestacc = (acc, err)


        nbacc   = NB_cl(data, Nest, CV)
        xgb_res = str(bestacc[0]) + '\t' + str(bestacc[1]) 
        nb_res  = str(nbacc[0])   + '\t' + str(nbacc[1])   + '\t' + str(nbacc[2])  

        fout = open(outfolder + dataset + '_' + measure + '.dat', 'a')
        fout.write(str(TauLimit) + '\t' + xgb_res+ '\t' + nb_res + '\n')
        fout.close()



def get_combined_prediction_results(TauLimit, Nest, CV, ijk, cumulative):

    # get the data feature file
    centralityFeatures_Q, centralityFeatures_Qbin, centralityFeatures_T = get_combined_features(TauLimit, directors, cumulative = cumulative)    

    # do optimized predictions
    for dataset, data in [('quartiles', centralityFeatures_Q), ('quartbinary', centralityFeatures_Qbin), ('topbottom', centralityFeatures_T)]:


        print 'COMBINED\t', TauLimit, '\t', dataset

        bestacc = (0, 0)

        for sample in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            for rate in [0.01, 0.05, 0.1, 0.15, 0.2]:

                acc, err, N = xgb_cl(data, Nest, CV, 6 , rate, sample)
                if acc > bestacc[0]:
                    bestacc = (acc, err)


        nbacc   = NB_cl(data, Nest, CV)
        xgb_res = str(bestacc[0]) + '\t' + str(bestacc[1]) 
        nb_res  = str(nbacc[0])   + '\t' + str(nbacc[1])   + '\t' + str(nbacc[2])  

        fout = open(outfolder + dataset + '_COMBINED_TauNeg_' + str(ijk) + '.dat', 'a')
        fout.write(str(TauLimit) + '\t' + xgb_res+ '\t' + nb_res + '\n')
        fout.close()









''' ============================================================= '''
'''  -----------------  MAIN PREDICTION STUFF   ----------------- ''' 
''' ============================================================= '''

if sys.argv[1] == 'allfeatures':

    
    # Nest      --> 100
    # directors --> entire file
    # TauLimit  --> for loop
    # cv        --> 10
    # measure   --> for loop

    Nest       = 100
    CV         = 10
    cumulative = False
    measure    = 'degree'
    measures   = ['degree', 'clustering', 'pagerank', 'betweenness', 'constraint']  


    directors  = [aaa.replace('.dat', '') for aaa in os.listdir('NEWTemporal/4_directors_centralities_QEVER')]#[0:100]
   # dirids    = ['nm0000184', 'nm0000233',  'nm0000229', 'nm0000040', 'nm0000122', 'nm0000033', 'nm0000122', 'nm0000631', 'nm0001053', 'nm0000142', 'nm0001392', 'nm0000591', 'nm0000154', 'nm0001232', 'nm0001628']
   # directors = dirids+directors

    # get output follder
    if cumulative: 
        outfolder = 'MLResultsUP_cumulative/' 
    else:
        outfolder = 'MLResultsUP_local/'

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)


    # create output folders
    for dataset in ['quartiles', 'quartbinary', 'topbottom']:
        fout = open(outfolder + dataset + '_' + measure + '.dat', 'w')
        fout.close()


    Pros = []
    for measure in measures:
        for TauLimit in range(-20, 21):
            p = Process(target = get_meas_prediction_results, args=(TauLimit, Nest, CV, cumulative, ))
            Pros.append(p)
            p.start()
                 
        for t in Pros:
            t.join()


 

elif sys.argv[1] == 'combined':

    
    # Nest      --> 100
    # directors --> entire file
    # TauLimit  --> for loop
    # cv        --> 10
    # measure   --> for loop

    # switch to get_combined_features


    Nest       = 100
    CV         = 2
    cumulative = False

   



    # get output follder
    if cumulative: 
        outfolder = 'MLResultsUP_cumulative/' 
    else:
        outfolder = 'MLResultsUP_local/'

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)





   # get_combined_prediction_results(-3, Nest, CV, cumulative)


    for ijk in [0, 1,2,3,4]:


        # create output folders
        for dataset in ['quartiles', 'quartbinary', 'topbottom']:
            fout = open(outfolder + dataset + '_COMBINED_TauNeg_' + str(ijk) + '.dat', 'w')
            fout.close()



        measures = ['degree',    'clustering', 'pagerank', 'betweenness', 'constraint']  
        ids_meas = {}
        for meas in measures:
            
          
            for line in open('NEWTemporal/3_corr_shift_'+meas+'_QEVER.dat'):
                imdb, tau, _, _ = line.strip().split('\t')
                tau = float(tau)

                if tau <= 0:
                    if imdb not in ids_meas:
                        ids_meas[imdb] = 1
                    else:
                        ids_meas[imdb] += +1

   
        directors = [imdbid for imdbid, cnt in ids_meas.items() if cnt >ijk]



        Pros = []
        for TauLimit in range(-20, 21):
            p = Process(target = get_combined_prediction_results, args=(TauLimit, Nest, CV, ijk, cumulative,))
            Pros.append(p)
            p.start()
                 
        for t in Pros:
            t.join()

      
                




## source /opt/virtualenv-python2.7/bin/activate








