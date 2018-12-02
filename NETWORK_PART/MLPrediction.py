import numpy as np
import os
import gzip
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
import xgboost as xgb








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





def get_centr_features(Nlimit, dirids, measureid):

    centralityFeatures  = pd.DataFrame()
    centralityFeaturesR = pd.DataFrame()

    for dirid in dirids:

        column  = {}
        columnR = {}

        centralities_d, centralities  = get_centralities(dirid, measures, column =measureid)
        Istar, Nstar, NstarR, impacts = get_career_data(dirid, centralities_d)
        
        if Istar > 0.0 and Nstar > 0.0 and NstarR > 0.0:

            if Nstar >= Nlimit and len(centralities) >= Nstar:
                for i in range(Nlimit+1):
                    try:
                        column[i] = centralities[i][1]
                    except:
                        pass        


            if NstarR >= Nlimit and len(centralities) >= NstarR:
                try:
                    for i in range(Nlimit+1):
                        columnR[i] = centralities[i][1]        
                except:
                    pass


            if len(column) == Nlimit+1:
                column['Istar']  = Istar
                if Istar == 0: Istar = 1
                column['logIstar']  = math.log(Istar)
                df_column  = pd.DataFrame({dirid : column}).T
                centralityFeatures  = centralityFeatures.append(df_column, ignore_index=True)         


            if len(columnR) == Nlimit+1:
                columnR['Istar'] = Istar
                columnR['logIstar'] = math.log(Istar)
                df_columnR = pd.DataFrame({dirid : columnR}).T
                centralityFeaturesR = centralityFeaturesR.append(df_columnR, ignore_index=True)         



    centralityFeatures['IstarQ']  = pd.qcut(centralityFeatures['Istar'], 4, ['q1','q2','q3','q4'])
    centralityFeaturesR['IstarQ'] = pd.qcut(centralityFeaturesR['Istar'],4, ['q1','q2','q3','q4'])
    
    centralityFeatures['logIstarQ']  = pd.qcut(centralityFeatures['logIstar'], 4, ['q1','q2','q3','q4'])
    centralityFeaturesR['logIstarQ'] = pd.qcut(centralityFeaturesR['logIstar'],4, ['q1','q2','q3','q4'])
    
    
    centralityFeatures  = centralityFeatures.dropna()    
    centralityFeaturesR = centralityFeaturesR.dropna()    
    
    
    return centralityFeatures, centralityFeaturesR





def get_centr_features_10perc(Nlimit, dirids, measureid):

    centralityFeatures  = pd.DataFrame()
    centralityFeaturesR = pd.DataFrame()

    for dirid in dirids:

        column  = {}
        columnR = {}

        centralities_d, centralities  = get_centralities(dirid, measures, column =measureid)
        Istar, Nstar, NstarR, impacts = get_career_data(dirid, centralities_d)
        
        if Istar > 0.0 and Nstar > 0.0 and NstarR > 0.0:

            if Nstar >= Nlimit and len(centralities) >= Nstar:
                for i in range(Nlimit+1):
                    try:
                        column[i] = centralities[i][1]
                    except:
                        pass        


            if NstarR >= Nlimit and len(centralities) >= NstarR:
                try:
                    for i in range(Nlimit+1):
                        columnR[i] = centralities[i][1]        
                except:
                    pass


            if len(column) == Nlimit+1:
                column['Istar']  = Istar
                if Istar == 0: Istar = 1
                column['logIstar']  = math.log(Istar)
                df_column  = pd.DataFrame({dirid : column}).T
                centralityFeatures  = centralityFeatures.append(df_column, ignore_index=True)         


            if len(columnR) == Nlimit+1:
                columnR['Istar'] = Istar
                columnR['logIstar'] = math.log(Istar)
                df_columnR = pd.DataFrame({dirid : columnR}).T
                centralityFeaturesR = centralityFeaturesR.append(df_columnR, ignore_index=True)      


    labels =  [str(10*(i+1))+ '%' for i in range(10)]

    centralityFeatures['IstarQ']  = pd.qcut(centralityFeatures['Istar'],10,  labels)    
    centralityFeatures['IstarQ'] = centralityFeatures['IstarQ'].replace('100%', 'top10')
    centralityFeatures['IstarQ'] = centralityFeatures['IstarQ'].replace(r'^.*%.*$', 'bottom90',  regex=True)


    centralityFeatures['logIstarQ']  = pd.qcut(centralityFeatures['logIstar'],10,  labels)
    centralityFeatures['logIstarQ'] = centralityFeatures['logIstarQ'].replace('100%', 'top10')
    centralityFeatures['logIstarQ'] = centralityFeatures['logIstarQ'].replace(r'^.*%.*$', 'bottom90',  regex=True)


    centralityFeatures  = centralityFeatures.dropna()    
    centralityFeaturesR = centralityFeaturesR.dropna()    

    centralityFeatures_top    = centralityFeatures[centralityFeatures.IstarQ=='top10']
    centralityFeatures_bottom = centralityFeatures[centralityFeatures.IstarQ=='bottom90'].sample(len(centralityFeatures_top))
    
    centralityFeatures = centralityFeatures_bottom.append(centralityFeatures_top, ignore_index=True)   
        
    
    

    return centralityFeatures, centralityFeaturesR







def xgb_pred(X, y, max_depth_ ,learning_rate_, subsample_):
              
    train_data, test_data, train_label, test_label =  train_test_split(X, y, test_size=.33, random_state=42)    
      
    model2       = xgb.XGBClassifier(n_estimators=1000   , max_depth=max_depth_, learning_rate=learning_rate_, subsample=subsample_)
    train_model2 = model2.fit(train_data, train_label)
    pred2        = train_model2.predict(test_data)
    accuracies   = list(cross_val_score(train_model2, train_data, train_label, cv=5))    

    return np.mean(accuracies), np.std(accuracies), len(X)










    

folderout = 'ML_FIRST_RES/' 
if not os.path.exists(folderout): os.makedirs(folderout)

measures  = ['degree',    'clustering', 'pagerank', 'betweenness', 'closeness', 'constraint']  
dirids    = ['nm0000184', 'nm0000233',  'nm0000229', 'nm0000040', 'nm0000122', 'nm0000033', 'nm0000122', 'nm0000631', 'nm0001053', 'nm0000142', 'nm0001392', 'nm0000591', 'nm0000154', 'nm0001232', 'nm0001628']
directors = [aaa.replace('.dat', '') for aaa in os.listdir('NEWTemporal/4_directors_centralities_QEVER') if 'swp' not in aaa]


for measure in measures:
    fout = open(folderout + measure + '_results.dat', 'w')
    fout.close()







def get_prediction(Nlimit):

 for measureid, measure in enumerate(measures):

        print Nlimit, '\t', measure

 
        #centralityFeatures, centralityFeaturesR = get_centr_features(Nlimit, directors, measureid = measureid+1)    
        centralityFeatures, centralityFeaturesR = get_centr_features_10perc(Nlimit, directors, measureid = measureid+1)  


        X = centralityFeatures.drop(columns = ['Istar', 'logIstar', 'IstarQ', 'logIstarQ'])
        y = list(centralityFeatures['logIstarQ'])

        if len(X) > 10:

            max_depth_     = 5
            learning_rate_ = 0.05
            subsample_     = 0.8


            acc, err, num = xgb_pred(X, y, max_depth_ ,learning_rate_, subsample_)

            fout = open(folderout + measure + '_results.dat', 'a')
            fout.write(str(Nlimit) + '\t' + str(acc) + '\t' + str(err) + '\t' + str(num) + '\n')
            fout.close()





def optimize_prediction(Nlimit):

    folderout2 = 'ML_FIRST_RES/Optimize/' 
    if not os.path.exists(folderout2): os.makedirs(folderout2)

    for measureid, measure in enumerate(measures):

        print Nlimit, '\t', measure

 
        #centralityFeatures, centralityFeaturesR = get_centr_features(Nlimit, directors, measureid = measureid+1)    
        centralityFeatures, centralityFeaturesR = get_centr_features_10perc(Nlimit, directors, measureid = measureid+1)  


        X = centralityFeatures.drop(columns = ['Istar', 'logIstar', 'IstarQ', 'logIstarQ'])
        y = list(centralityFeatures['logIstarQ'])

        if len(X) > 10:


            best = [0,0,0,0,0]

            for depth in [4,5,6]:

                for rate in [0.01, 0.05, 0.1, 0.15, 0.2]:
            
                    for sample in [0.7, 0.85, 0.9, 0.95]: 
      
                        acc, err, num = xgb_pred(X, y, depth, rate, sample)
                        if acc > best[0]:
                            best = [acc, err, depth, rate, sample, len(X)]
                    

            fout = open(folderout2 + measure + '_' + str(Nlimit) + '_best.dat', 'w')
            fout.write(  '\t'.join( [str(fff) for fff in best ] ) )
            fout.close()















  


def get_centr_features_combined(Nlimit, dirids):

    centralityFeatures  = pd.DataFrame()
    centralityFeaturesR = pd.DataFrame()

    for dirid in dirids:

        column  = {}
        columnR = {}
        collen  = 0
        
        measures = ['degree',    'clustering', 'pagerank', 'betweenness', 'constraint']  

        for jind, measure in enumerate(measures):

            centralities_d, centralities  = get_centralities(dirid, measures, column = jind + 1)
            Istar, Nstar, NstarR, impacts = get_career_data(dirid, centralities_d)

            if Istar > 0.0 and Nstar > 0.0 and NstarR > 0.0:

                if Nstar >= Nlimit and len(centralities) >= Nstar:
                    for i in range(Nlimit+1):
                        collen += 1
                        try:
                            column[str(i) + '_' + measure] = centralities[i][1]
                        except:
                            pass   


                if NstarR >= Nlimit and len(centralities) >= NstarR:
                    for i in range(Nlimit+1):
                        try:
                            columnR[str(i) + '_' + measure] = centralities[i][1]        
                        except:
                            pass   



        if len(column) == collen:
            column['Istar']  = Istar
            if Istar == 0: Istar = 1
            column['logIstar']  = math.log(Istar)
            df_column  = pd.DataFrame({dirid : column}).T
            centralityFeatures  = centralityFeatures.append(df_column, ignore_index=True)         


        if len(columnR) == collen:
            columnR['Istar'] = Istar
            columnR['logIstar'] = math.log(Istar)
            df_columnR = pd.DataFrame({dirid : columnR}).T
            centralityFeaturesR = centralityFeaturesR.append(df_columnR, ignore_index=True)    


             

    labels =  [str(10*(i+1))+ '%' for i in range(10)]

    centralityFeatures['IstarQ']  = pd.qcut(centralityFeatures['Istar'],10,  labels)    
    centralityFeatures['IstarQ'] = centralityFeatures['IstarQ'].replace('100%', 'top10')
    centralityFeatures['IstarQ'] = centralityFeatures['IstarQ'].replace(r'^.*%.*$', 'bottom90',  regex=True)


    centralityFeatures['logIstarQ']  = pd.qcut(centralityFeatures['logIstar'],10,  labels)
    centralityFeatures['logIstarQ'] = centralityFeatures['logIstarQ'].replace('100%', 'top10')
    centralityFeatures['logIstarQ'] = centralityFeatures['logIstarQ'].replace(r'^.*%.*$', 'bottom90',  regex=True)


    centralityFeatures  = centralityFeatures.dropna()    
    centralityFeaturesR = centralityFeaturesR.dropna()    

    centralityFeatures_top    = centralityFeatures[centralityFeatures.IstarQ=='top10']
    centralityFeatures_bottom = centralityFeatures[centralityFeatures.IstarQ=='bottom90'].sample(len(centralityFeatures_top))
    
    centralityFeatures = centralityFeatures_bottom.append(centralityFeatures_top, ignore_index=True)   
        
   
    return centralityFeatures, centralityFeaturesR












def optimize_prediction_combined(Nlimit):

    folderout2 = 'ML_FIRST_RES/Optimize/' 
    if not os.path.exists(folderout2): os.makedirs(folderout2)

 
    centralityFeatures, centralityFeaturesR = get_centr_features_combined(Nlimit, directors)  


    X = centralityFeatures.drop(columns = ['Istar', 'logIstar', 'IstarQ', 'logIstarQ'])
    y = list(centralityFeatures['logIstarQ'])

    if len(X) > 10:


        best = [0,0,0,0,0]

        for depth in [4,5,6]:

            for rate in [0.01, 0.05, 0.1, 0.15, 0.2]:
        
                for sample in [0.7, 0.85, 0.9, 0.95]: 
  
                    acc, err, num = xgb_pred(X, y, depth, rate, sample)
                    if acc > best[0]:
                        best = [acc, err, depth, rate, sample, len(X)]

                   
        fout = open(folderout2 + 'COMBINED_' + str(Nlimit) + '_best.dat', 'w')
        fout.write(  '\t'.join( [str(fff) for fff in best ] ) )
        fout.close()

















    
   
Pros = []





for Nlimit in range(20):
    p = Process(target = optimize_prediction, args=(Nlimit, ))
    Pros.append(p)
    p.start()
   
for t in Pros:
    t.join()






for Nlimit in range(20):
    p = Process(target = optimize_prediction_combined, args=(Nlimit, ))
    Pros.append(p)
    p.start()
   
for t in Pros:
    t.join()
        

folderout2 = 'ML_FIRST_RES/Optimize' 
outfiles  = os.listdir(folderout2)





for measure in measures:
    fout = open(folderout + measure + '_opt_results.dat', 'w')
    fout.close()


for fn in outfiles:

    meas, Nlimit, _ = fn.split('_')

    fout = open(folderout + meas + '_opt_results.dat', 'a')

    for line in open(folderout2 + '/'  + fn):
        fout.write( Nlimit + '\t' + line + '\n')
    fout.close()



## source /opt/virtualenv-python2.7/bin/activate






   







