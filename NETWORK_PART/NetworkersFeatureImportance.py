import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import gzip
import math
import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings("ignore")




'''  --------------------------------------------------------- '''
'''                    COMPARE DISTRIBUTIONS                   '''
'''  --------------------------------------------------------- '''

def KS_distance_comparison(data):



    df_0  = data[data.networker == 0]
    df_1  = data[data.networker == 1]
    feats = list(data.keys())
    res   = {}  


    print '\n\n----------------------------------------------\nKS distances\n'

    outfolder = 'NetworkersAnalysis/DistributionPlots'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for ind, feat in enumerate(feats):

        x0 = df_0[feat]
        x1 = df_1[feat]
        
        if 'meta' in feat:
            x0 = [xx for xx in x0  if xx > 0]
            x1 = [xx for xx in x1  if xx > 0]
            
        if 'T' == feat:  
            x0 = [xx for xx in x0  if xx < 100]
            x1 = [xx for xx in x1  if xx < 100]        

        
        KS, p = stats.ks_2samp(x0, x1)
        f, ax = plt.subplots(1,2,figsize = (10,5))

        res[KS] = p

        plt.suptitle(feat + ', KS = ' + str(round(KS,3)), fontsize = 15 )
        
        ax[0].hist(x0, alpha = 0.6, bins = 15, density = True)#, cumulative = True)
        ax[0].hist(x1, alpha = 0.6, bins = 15, density = True)#, cumulative = True)

        ax[1].hist(x0, alpha = 0.6, bins = 15, density = True, cumulative = True)
        ax[1].hist(x1, alpha = 0.6, bins = 15, density = True, cumulative = True)

        
        
        plt.savefig(outfolder + '/' + feat + '.png')  
        plt.close()


    dfres = pd.DataFrame(res.items()).T
    dfres = dfres.rename(columns = {ind : f for ind, f in enumerate(feats)})
    dfres.index = ['KS', 'p']

    print dfres.round(3), '\n'





'''  --------------------------------------------------------- '''
'''                        DO ML PREDICTIONS                   '''
'''  --------------------------------------------------------- '''


def balance_samples(data):
    
    data_0 = data[data.networker == 0]
    data_1 = data[data.networker == 1]
    
    num    = min([len(data_0), len(data_1)])
    data_0 = data_0.sample(num)
    data_1 = data_1.sample(num)
    
    return data_0.append(data_1)
    

#def xgb_cl(X, y, Nest, CV, max_depth_ ,learning_rate_, subsample_):
      
#    accuracies = []
        
#    for i in range(Nest):

 #       train_data, test_data, train_label, test_label =  train_test_split(X, y, test_size=.33, random_state=42)    
          
 #       model2       = xgb.XGBClassifier(n_estimators=100   , max_depth=max_depth_, learning_rate=learning_rate_, subsample=subsample_)
 #       train_model2 = model2.fit(train_data, train_label)


        # featurenames, importances = zip(*sorted(zip(train_data.keys(), model2.feature_importances_), key=lambda tup: tup[1], reverse = True)) 
        # for i in range(len(featurenames)):
        #    if importances[i] > 0:
        #        lenn = ' '.join((5+len('FirstImpacts_std') - len(featurenames[i]) )*[''])
        #        print featurenames[i], lenn, str(importances[i]) 

       


#        pred2        = train_model2.predict(test_data)
#        accuracies   += list(cross_val_score(train_model2, train_data, train_label, cv=CV))    

#    return np.mean(accuracies), np.std(accuracies), len(X)


def NB_cl(X, y, Nest, CV):
    
    accuracies = []

    for i in range(Nest):

        train_data, test_data, train_label, test_label =  train_test_split(X, y, test_size=.33, random_state=42)    

        name       = 'Naive Bayes' 
        classifier = GaussianNB()  
        clas       = classifier.fit(train_data, train_label)
        accuracies += list(cross_val_score(clas, train_data, train_label, cv=CV))       
            
    return np.mean(accuracies), np.std(accuracies), len(X)


def ML_classifications(data, Nest, CV, depth, rate, sample):

  #  print '\n\n----------------------------------------------\nML classifications -- Accuracy\n'

   

    #acc, err, N = NB_cl(X, y, 10, 5)
    #print 'NB:   ', round(acc, 3), ' +/- ', round(err,3), ',\t', N




    accs_xb = []
        
    for i in range(10):

        data = balance_samples(data)
        X  = data.drop(columns =  ['networker'])
        y  = data['networker']


        train_data, test_data, train_label, test_label =  train_test_split(X, y, test_size=.33, random_state=42)    
          


        param_test1 = {
             'max_depth':range(2,7,1),
             'min_child_weight':range(1,6,1),
             'subsample':np.arange(0.7,0.95, 0.05)
             'learning_rate':np.arange(0.05, 0.2, 0.03)
            }


        gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=5,
                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                    objective= 'binary:logistic', nthread=2, scale_pos_weight=1, seed=27), 
                    param_grid = param_test1, scoring='accuracy',n_jobs=-1,iid=False, cv=5)


        
        gsearch1.fit(train_data,train_label)


        accs_xb.append(gsearch1.best_score_ )

        #print gsearch1.best_score_
        #print gsearch1.best_params_





        #model2       = xgb.XGBClassifier(n_estimators=100   , max_depth=5, learning_rate=0.1, subsample=0.8)
        #train_model2 = model2.fit(train_data, train_label)
        #pred2        = train_model2.predict(test_data)
        #accuracies   += list(cross_val_score(train_model2, train_data, train_label, cv=CV))    







    print 'AVG  ', np.mean(accs_xb)
    #acc, err, N = xgb_cl(X, y, 10, 5, depth, rate, sample)
    #print 'XGB:  ', round(acc, 3), ' +/- ', round(err,3), ',\t', N, '\n'

    return np.mean(accs_xb)



'''  --------------------------------------------------------- '''
'''               REGERSSION WITHIN MATCHING SAMPLES           '''
'''  --------------------------------------------------------- '''


def get_propensity_score_bins(ddf, nbins, outcome):

    classifier = LogisticRegression(C=1.0, penalty='l1')

    #encode_cats(df_field, 'gender')
    #encode_cats(df_field, 'location')
    
    try:
        XL = ddf.drop(columns = ['networker', outcome, 'propensity', 'bins'])  
    except:
        XL = ddf.drop(columns = ['networker', outcome])  

        
    XL = StandardScaler().fit_transform(XL)
    XL = preprocessing.quantile_transform(XL, output_distribution = 'normal')
    yL = np.asarray(ddf['networker'])

    classifier.fit(XL, yL)
    yL_pred_cl   = classifier.predict(XL)
    yL_pred_prob = classifier.predict_proba(XL)


    df_prop = pd.DataFrame(ddf)
    df_prop['propensity'] = [p[1] for p in yL_pred_prob]
    

    df_prop['bins'] = pd.qcut(df_prop['propensity'], nbins, ['q' + str(i+1) for i in range(nbins)])
    df_prop.head()
    
    return df_prop, list(set(df_prop.bins))




def do_linear_regr(df_prop, labels, outcome): 

    regr      = LinearRegression()
    regr_data = {}
    df_res    = pd.DataFrame()

    for q in labels:

        df_q = df_prop[df_prop.bins == q]
        X_reg = df_q.drop(columns = ['bins', 'propensity', 'networker'])
        covs  = X_reg.keys()

        y_reg = df_q['networker']

        X_reg = StandardScaler().fit_transform(X_reg)
        X_reg = preprocessing.quantile_transform(X_reg, output_distribution = 'normal')

        y_reg = StandardScaler().fit_transform(np.asarray(y_reg).reshape(-1, 1) )
        #y_reg = preprocessing.quantile_transform(y_reg, output_distribution = 'normal')

        
        train_data, test_data, train_label, test_label =  train_test_split(X_reg, y_reg, test_size=.33, random_state=42)
        regr.fit(train_data, train_label)
     

        res = {}
        res['R2 train'] = r2_score(train_label, regr.predict(train_data))
        res['R2 test']  = r2_score(test_label, regr.predict(test_data))

        
        for i in range(len(covs)):
             res[covs[i]] = (regr.coef_[0][i],)
          
        
        res = pd.DataFrame(res)
        res.index = [q]
        res.index.name = 'quart'
        df_res = df_res.append(res)
        
        
    return df_res


def MS_matching_samples(data, nbins):



    outcome = 'N'
    feats   = list(data.keys())
    results = pd.DataFrame()

    print '\n\n----------------------------------------------\nMatching Samples -- R^2 test \n'

    for outcome in feats:
        df_prop, labels = get_propensity_score_bins(data, nbins = nbins, outcome = outcome)
        res = do_linear_regr(df_prop, labels, outcome)
        #print outcome
        #print res.round(3)
        results[outcome] = res['R2 test']#.round(3)

        print outcome, res['R2 test']


#    print results, '\n\n'
    










if __name__ == "__main__":


    inp = sys.argv[1]

    data = pd.DataFrame.from_csv('NetworkersAnalysis/directors_features_'+inp+'.csv')


    print len(data)
    # KS distance comparisons
   # KS_distance_comparison(data)


    # ML classification
    Nest   = 1
    CV     = 5
    depth  = 6
    rate   = 0.1
    sample = 0.9   

    aacc = 0    

    ML_classifications(data, Nest, CV, depth, rate, sample)

  

   #
    # matching samples
   # nbins = 4
   # MS_matching_samples(data, nbins)
    

    # MAX:  0.6757575757575758


#   >4  0.6885780885780884
#   >3  0.6509803921568627     0.6453431372549019
#   >2  





