import pandas as pd
import numpy as np
import math
import random
import sys
import operator
import time
from operator import itemgetter
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process
from collections import Counter
import warnings
warnings.filterwarnings('ignore')



''' encode categorical variables (names to numbers) '''

def encode_cats(df, column):
    le = preprocessing.LabelEncoder()
    le.fit(df[column])
    df[column] = le.transform(df[column])    
    return df



''' parse and and order features of professions '''

def get_all_features():

    ### LOCATIONS
    folder   = 'FinalLocationFiles/'
    df_loc_Q = {}
    files    = os.listdir(folder)

    for fn in files:
        
        field = fn.rsplit('_', 2)[0]
        df    = pd.DataFrame.from_csv(folder + fn, sep = '\t')
        df = df.replace('uk' , 'united kingdom')
        df = df.replace('south korea' , 'korea')
        df = df.replace('hong kong' , 'china')
        df = df.replace('czech republic' , 'czech rep.')
        df = df.replace('usa' , 'united states')
        df_loc_Q[field] = df
        
        
    ### GENDER
    folder      = 'FinalGenderFiles/'
    df_gender_Q = {}
    files       = os.listdir(folder)
    field_avg_Q = []
    field_std_Q = []
    field_len_Q = []

    for fn in files:

        field = fn.split('_')[0]
        df    = pd.DataFrame.from_csv(folder + fn, sep = '\t', index_col = 'id')
        df    = df[df.gender.isin(['male', 'female'])]
        df_gender_Q[field] = df  
        
      
    ### USERS CAREER TIME
    folder     = 'UsersTimeData/'
    files      = os.listdir(folder)
    career_dfs = {}

    for fn in files:
        
        field = fn.split('_')[0]
        df    = pd.DataFrame.from_csv(folder + fn, sep = '\t', header = None)
        df.index.name = 'id'
        df    = df.rename(columns = { 1 : 'first year', 2 : 'total span', 3 : 'productivity', 4 : 'best'})
        career_dfs[field] = df
        
  
    # p PARAMS
    folder     = 'pData/p_stat_data_'
    files      = [(folder + f + '_0.dat', f ) for f in career_dfs.keys()  ]
    df_pparams = {}

    for (fn, field) in files:  
        field = field.split('-')[0]
        if 'art' in fn: fn = fn.replace('art', 'art_director-20')
        if 'art' in field: field = 'art_director'

        df = pd.DataFrame.from_csv(fn, sep = '\t')
        df = df.drop(columns = ['Q', 'median_p'])
        df_pparams[field] = df
        
        
    # MERGING
    merged_dfs ={}

    for field, df in career_dfs.items():
        
        field =  field.rsplit('-', 1)[0]
        df_g  = df_gender_Q[field]
        
        if 'art' in field: field = 'art_director'

        df_l     = df_loc_Q[field]
        df_feats = df_l.merge(df_g, left_index = True, right_index = True)
        df_feats = df_feats.merge(df, left_index = True, right_index = True)
        df_feats = df_feats.rename(columns = {'Q_x' : 'Q'})
        df_feats = df_feats.drop(columns = ['Q_y'])
        df_feats = df_feats.merge(df_pparams[field], left_index = True, right_index = True)

        merged_dfs[field] = df_feats


    return merged_dfs



''' get the data about the prizes '''

def get_all_prizes():

    books_nobel    = [int(line.strip().split('\t')[0]) for line in  open('Prize_data/Book/Nobel_winners_all.dat')]
    books_pulitzer = [int(line.strip().split('\t')[0]) for line in  open('Prize_data/Book/Pulitzer_winners_all.dat')]
    film_oscar     = [line.strip().split('\t')[0] for line in  open('Prize_data/Film/Oscar_winners.dat')]


    genres             = os.listdir('Prize_data/Music')
    genre_files_09     = ['Prize_data/Music/' + genre + '/' + genre + '_0.9.dat' for genre in genres]
    genre_files_095    = ['Prize_data/Music/' + genre + '/' + genre + '_0.95.dat' for genre in genres]
    music_grammies_09  = list(set([item for sublist in [ [int(line.strip().split('\t')[1]) for line in open(fn)]  for fn in genre_files_09]   for item in sublist]))
    music_grammies_095 = list(set([item for sublist in [ [int(line.strip().split('\t')[1]) for line in open(fn)]  for fn in genre_files_095]  for item in sublist]))
    professions_prizes = {}


    professions_prizes    = []
    professions_prizes.append(('authors', 'books_nobel',    books_nobel))
    professions_prizes.append(('authors', 'books_pulitzer', books_pulitzer))

    for prof in ['art_director', 'composer', 'director', 'producer', 'writer']:
        professions_prizes.append((prof, 'film_oscar', film_oscar))
        
    for prof in ['rock', 'electro', 'pop', 'funk', 'folk', 'jazz', 'hiphop', 'classical']:
        professions_prizes.append((prof, 'music_grammies_09',  music_grammies_09))
        professions_prizes.append((prof, 'music_grammies_095', music_grammies_095))


    return professions_prizes



''' merge features with prizes '''

def merge_features_and_prizes(merged_dfs, professions_prizes):

    features_prizes = {}

    for (prof, prize, prize_names) in professions_prizes:
        
        df = merged_dfs[prof]
        
        prize_names_present = list(set(prize_names).intersection(set(df.index)))      

        names_and_prizes = {}
        for name in list(df.index):
            if name not in prize_names_present:
                names_and_prizes[name] = 0
            else:
                names_and_prizes[name] = 1
                
        
        df_prize = pd.DataFrame(names_and_prizes.items())      
        df_prize = df_prize.rename(columns = { 0 : 'id', 1 : 'prizewinner'})
        df_prize.index = df_prize.id
        df_prize = df_prize.drop(columns = ['id'])  

        df = df.merge(df_prize, left_index = True, right_index = True)
        features_prizes[prof] = df  


    return features_prizes



''' get statistics about the groups of winners and losers '''

def get_data_stats(artists_prizes, title):
    
    directors_oscars_0 = artists_prizes[artists_prizes['prizewinner'] == 0]
    directors_oscars_1 = artists_prizes[artists_prizes['prizewinner'] == 1]
  

    countries_winner = '\t'.join([k + ':' + str(v) for k, v in dict(Counter(list(directors_oscars_1.location))).items()])
    countries_loser  = '\t'.join([k + ':' + str(v) for k, v in dict(Counter(list(directors_oscars_0.location))).items()])

    Q_winner = np.mean(list(directors_oscars_1.Q))
    Q_loser  = np.mean(list(directors_oscars_0.Q)) 
    
    female_winner = len(directors_oscars_1[directors_oscars_1.gender == 'female'])
    male_winner   = len(directors_oscars_1[directors_oscars_1.gender == 'male'])
    female_loser  = len(directors_oscars_0[directors_oscars_0.gender == 'female'])
    male_loser    = len(directors_oscars_0[directors_oscars_0.gender == 'male'])
    
    
    folderout = 'Prize_group_stats'
    if not os.path.exists(folderout):
        os.makedirs(folderout)


    fout = open(folderout + '/' + title + '_stat.dat', 'w')

    fout.write('female_winner\t' + str(female_winner) + '\n')
    fout.write('male_winner\t'   + str(male_winner)   + '\n')
    fout.write('female_loser\t'  + str(female_loser)  + '\n')
    fout.write('male_loser\t'    + str(male_loser)    + '\n')

    fout.write('Q_winner\t'      + str(Q_winner)      + '\n')
    fout.write('Q_loser\t'       + str(Q_loser)       + '\n\n')


    fout.write('Countries_winner\t' + countries_winner + '\n')
    fout.write('Countries_loser\t'  + countries_loser  + '\n')

    fout.close()

    

''' get balanced samples '''    

def get_sample_data(artists_prizes, prize, feature = ''):
    
    if len(feature) > 0:
        artists_prizes = artists_prizes.drop(columns= [c for c in artists_prizes.keys() if c not in [feature, 'prizewinner']])
    
    directors_oscars_0 = artists_prizes[artists_prizes[prize] == 0]
    directors_oscars_1 = artists_prizes[artists_prizes[prize] == 1]
    directors_oscars_0 = directors_oscars_0.sample(len(directors_oscars_1))
 
    directors_oscars_balanced = directors_oscars_0.append(directors_oscars_1)
    directors_oscars_balanced = directors_oscars_balanced.sample(frac = 1)

    directors_oscars_balanced.head()

    X = directors_oscars_balanced.drop(columns = [prize])
    y = np.asarray(directors_oscars_balanced[prize])
    
    
    if 'gender' in X.keys():    encode_cats(X, 'gender')
    if 'location' in X.keys():  encode_cats(X, 'location')
    
    return X, y



''' xgboost classifier '''

def xgb_model_params_importance(X, y, max_depth_, learning_rate_, subsample_, n_thread_):
    
    train_data, test_data, train_label, test_label =  train_test_split(X, y, test_size=.33, random_state=42, stratify = y)    
           
    model2       = xgb.XGBClassifier(n_estimators=100, n_thread = n_thread_, max_depth=max_depth_, learning_rate=learning_rate_, subsample=subsample_)
    train_model2 = model2.fit(train_data, train_label)
    pred2        = train_model2.predict(test_data)
    accuracies   = list(cross_val_score(model2, train_data, train_label, cv=10))
        
    return accuracies



''' do the predictions '''

def do_predictions(features_prizes, field, R, features):

    fout  = open(folderout + '/' + field + '_pred_results_R=' + str(R) + '.dat', 'w')

    fout.write('feature\tbest_acc\terror\tmax_depth\tlearning_rate\tsubsample_size\n')
    get_data_stats(features_prizes[field],  field)

    nnn = len(features)

    for ind, feature in enumerate(features):

        print field, '\t', ind+1, '/', nnn

        for max_depth_ in [4,5]:
            for learningrate_ in [0.01, 0.05]:
                for subsample_ in [0.5, 0.8]:

                    accuracies = []

                    for i in range(R):
                                    
                        X, y = get_sample_data(features_prizes['director'], 'prizewinner', feature)
                        accuracies += xgb_model_params_importance(X, y, max_depth_, learningrate_, subsample_, 1)
        
                    params =  str(max_depth_) + '_' + str(learningrate_) + '_' +  str(subsample_)
                    results[params] = (np.mean(accuracies), np.std(accuracies) / math.sqrt(len(accuracies)))


                    
        best_pred   = max(results.values(), key = itemgetter(1))
        best_acc    = best_pred[0]
        best_error  = best_pred[1]
        best_params = [k for k, v in results.items() if best_acc == v[0]][0].replace('_', '\t')
        if feature == '': feature = 'all'

        fout.write( feature +  '\t' + str(round(best_acc,5)) + '\t' + str(round(best_error, 5)) + '\t' + best_params + '\n')




if __name__ == "__main__":


    t1 = time.time()

    merged_dfs         = get_all_features()
    professions_prizes = get_all_prizes()
    features_prizes    = merge_features_and_prizes(merged_dfs, professions_prizes)
    folderout          = 'PredictionResults'

    if not os.path.exists(folderout):
        os.makedirs(folderout)



    print round(time.time()-t1), ' seconds to parse the data'
    t1 = time.time()

    fields   = features_prizes.keys()
    features = ['', 'location', 'Q', 'gender', 'first year', 'total span', 'productivity', 'best', 'mean_p']
    results  = {}
    R        = int(sys.argv[1])

    Pros    = []   

    fields = ['director', 'art_director']

    for field in fields: 
        p = Process(target = do_predictions, args=(features_prizes, field, R, features, ))
        Pros.append(p)
        p.start()


    for t in Pros:
        t.join()



    print round(time.time()-t1), ' to do the prediction with ' + str(R) + ' randomizations'












