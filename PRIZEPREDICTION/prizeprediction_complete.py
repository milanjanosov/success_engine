import pandas as pd
import numpy as np
import math
import random
import operator
from operator import itemgetter
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')



def encode_cats(df, column):
    le = preprocessing.LabelEncoder()
    le.fit(df[column])
    df[column] = le.transform(df[column])    
    return df



def get_sample_data(df_feats):

    df_feats_0 = df_feats[df_feats.prizewinner == 0]
    df_feats_1 = df_feats[df_feats.prizewinner == 1]
    df_feats_0 = df_feats_0.sample(len(df_feats_1))
 
    df_feats_balanced = df_feats_0.append(df_feats_1)
    
    X = df_feats_balanced.drop(columns = ['prizewinner'])
    y = np.asarray(df_feats_balanced.prizewinner)
    
    if 'gender' in X.keys():    encode_cats(X, 'gender')
    if 'location' in X.keys():  X = pd.get_dummies(X,dummy_na=True)  
    
    return X, y  



def xgb_model_params_importance(df_input, Nrand, text = '', feats = ''):
    
    accs = []
    
    if len(feats) > 0:
        df_input = df_input.drop(columns = [c for c in df_input.keys() if c not in [feats, 'prizewinner']])
    
    
    for i in range(Nrand):

        X, y = get_sample_data(df_input)
        print text, '\t', i

        train_data, test_data, train_label, test_label =  train_test_split(X, y, test_size=.33, random_state=42, stratify = y)    

        param_test1 = {
                 'max_depth':range(6,7,1),
                 'min_child_weight':range(1,6,1),
                 'subsample':np.arange(0.7,0.95, 0.05),
                 'learning_rate':np.arange(0.05, 0.2, 0.03)
                }


        gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=5,
                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                    objective= 'binary:logistic', nthread=2, scale_pos_weight=1, seed=27), 
                    param_grid = param_test1, scoring='accuracy',n_jobs=-1,iid=False, cv=5)

        gsearch1.fit(train_data,train_label) 

        accs.append(gsearch1.best_score_)



    return str(np.mean(accs)) + '\t' + str(np.std(accs))




def parse_data(label, field, lim):


    ### LOCATIONS
    fn_location = 'FinalLocationFiles/' + field + '_location_data.csv'
    df_loc_Q    = pd.DataFrame.from_csv(fn_location, sep = '\t')
    df_loc_Q    = df_loc_Q.replace('uk' , 'united kingdom')
    df_loc_Q    = df_loc_Q.replace('south korea' , 'korea')
    df_loc_Q    = df_loc_Q.replace('hong kong' , 'china')
    df_loc_Q    = df_loc_Q.replace('czech republic' , 'czech rep.')
    df_loc_Q    = df_loc_Q.replace('usa' , 'united states')
    df_loc_Q    = df_loc_Q.drop(columns = ['Q'])

    
    ### GENDER
    fn_gender = 'FinalGenderFiles/' + field + '_gender_data.csv'
    df_gender = pd.DataFrame.from_csv(fn_gender, sep = '\t', index_col = 'id')
    df_gender = df_gender[df_gender.gender.isin(['male', 'female'])]
    df_gender = df_gender.drop(columns = ['Q'])

   
    ### USERS CAREER TIME
    fn_career = 'UsersTimeData/'+field+'-' + lim + '_time_data.dat'
    df_career = pd.DataFrame.from_csv(fn_career, sep = '\t', header = None)
    df_career.index.name = 'id'
    df_career = df_career.rename(columns = { 1 : 'first year', 2 : 'total span', 3 : 'productivity', 4 : 'best'})


    ### Q PARAM
    fn_q = '../QMODELNEW/DataToPlot_linrescaled_final/3_pQ_distributions_processed/p_stat_data_' + field +'_0.dat'
    df_q = pd.DataFrame.from_csv(fn_q, sep = '\t')
    df_q = df_q.drop(columns = ['median_p'])

  
    ### FEATURES
    df_feats = df_loc_Q.merge(df_gender, left_index = True, right_index = True)
    df_feats = df_feats.merge(df_career, left_index = True, right_index = True)
    df_feats = df_feats.merge(df_q,      left_index = True, right_index = True)

    
    ### PRIZES
    books_nobel     = [int(line.strip().split('\t')[0]) for line in  open('Prize_data/Book/Nobel_winners_all.dat')]
    books_pulitzer  = [int(line.strip().split('\t')[0]) for line in  open('Prize_data/Book/Pulitzer_winners_all.dat')]
    film_oscar      = [line.strip().split('\t')[0] for line in  open('Prize_data/Film/Oscar_winners.dat')]
    genre_files_095 = ['Prize_data/Music/' + genre + '/' + genre + '_0.95.dat' for genre in os.listdir('Prize_data/Music')]
    music_grammies_095 = list(set([item for sublist in [ [int(line.strip().split('\t')[1]) for line in open(fn)]  for fn in genre_files_095]  for item in sublist]))



    if 'film'      == label:  prize_names = film_oscar
    elif 'music'   == label:  prize_names = music_grammies_095
    elif 'authors' == label:  prize_names = books_pulitzer
        

    prize_names_present = list(set(prize_names).intersection(set(df_feats.index)))     

    names_and_prizes = {}
    for name in list(df_feats.index):
        if name not in prize_names_present: 
            names_and_prizes[name] = 0
        else:
            names_and_prizes[name] = 1
            

    ### MERGING     
    df_prize = pd.DataFrame(names_and_prizes.items())      
    df_prize = df_prize.rename(columns = { 0 : 'id', 1 : 'prizewinner'})
    df_prize.index = df_prize.id
    df_prize = df_prize.drop(columns = ['id'])  
    df_feats = df_feats.merge(df_prize, left_index = True, right_index = True)

    return df_feats






inputdata = [('film', 'director', '10'), ('music', 'pop', '80'), ('authors', 'books', '50')]
Nrand     = 100   



for label, field, lim in inputdata:

    df_feats  = parse_data(label, field, lim)
    folderout = '../QMODELNEW/DataToPlot_linrescaled_final/8_prize_prediction'
    if not os.path.exists(folderout):
        os.makedirs(folderout)


    fileout  = open(folderout + '/' + field + '_prize_prediction.dat', 'w')
    features = df_feats.keys()

    print label, field, '\t', 'All features'
    fileout.write('All' + '\t' + xgb_model_params_importance(df_feats, Nrand, text = label + ' ' + field + '\tAll features') + '\n')

    for feat in features:
        if 'prizew' not in feat:
            print label, field, '\t', feat
            fileout.write( feat + '\t' + xgb_model_params_importance(df_feats, Nrand, text = label + ' ' + field + '\t' + feat, feats = feat) + '\n',)


    fileout.close()


##   source /opt/virtualenv-python2.7/bin/activate





