import os 
import sys
import gzip
import time
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory



''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''          GET THE DISTRIBUTION OF ALL SUCCESS MEASURES          '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   


 
def add_max_impact(lista, maxvalue):

    if 'nan' != maxvalue:
        lista.append(maxvalue)
 


def add_time_series(impacts, time_series):

    for year, events in time_series.items():
        if year not in impacts:
            impacts[year] = events
        else:
            impacts[year] += events  


def get_dict_data(impacts):

    x = []
    for year, impact in impacts.items():
        for i in impact:
            x.append(str(year) + '\t' + str(i))

    return x
  



def getImpactDistribution(source):


    input_data = [(os.listdir(source + 'film-director-simple-careers'), 'k',  'director'),
                  (os.listdir(source + 'film-producer-simple-careers'), 'b',  'producer'),   
                  (os.listdir(source + 'film-writer-simple-careers'),   'r',  'writer'),   
                  (os.listdir(source + 'film-composer-simple-careers'), 'g',  'composer'),   
                  (os.listdir(source + 'film-art-director-simple-careers'), 'y',  'art-director'),   
                 ]

    
    average_ratings = []
    rating_counts   = []
    metascores      = []
    critic_review   = []
    user_review     = []

    max_average_ratings = []
    max_rating_counts   = []
    max_metascores      = []
    max_critic_review   = []
    max_user_review     = []
    
    
    average_ratings_year = {}
    rating_counts_year   = {}
    metascores_year      = {}
    critic_review_year   = {}
    user_review_year     = {}


    for (files, color, label) in input_data:

        i = 0
        n = len(files)

        for filename in files:
        
            
            i += 1
            print i, '/', n
            
            #avg ratings
            pista_avg_rating = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 0)
            average_ratings  += pista_avg_rating.getImpactValues()
            add_max_impact(max_average_ratings, pista_avg_rating.getMaxImpact())
            time_series = pista_avg_rating.getYearlyProducts()
            add_time_series(average_ratings_year, time_series)


            # rating counts
            pista_ratingcnt = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 1)
            rating_counts   += pista_ratingcnt.getImpactValues()  
            add_max_impact(max_rating_counts, pista_ratingcnt.getMaxImpact())   
            time_series = pista_ratingcnt.getYearlyProducts()
            add_time_series(rating_counts_year, time_series)    
            
                                     
            # metascore
            pista_meta  = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 2)
            metascores  += pista_meta.getImpactValues() 
            add_max_impact(max_metascores, pista_meta.getMaxImpact())           
            time_series = pista_meta.getYearlyProducts()
            add_time_series(metascores_year, time_series)      
             
                        
            # critic reviews
            pista_critic  = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 3)
            critic_review += pista_critic.getImpactValues()          
            add_max_impact(max_critic_review, pista_critic.getMaxImpact())         
            pista_critic = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 3)         
            time_series  = pista_critic.getYearlyProducts()
            add_time_series(critic_review_year, time_series)               
               
                       
            # user reviews
            pista_user   = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 4)
            user_review  += pista_user.getImpactValues()
            add_max_impact(max_user_review, pista_user.getMaxImpact())
            time_series = pista_user.getYearlyProducts()
            add_time_series(user_review_year, time_series)       
           




        dir1 = 'ProcessedData/1_impact_distributions'
        if not os.path.exists(dir1):
            os.makedirs(dir1)

        f = open(dir1 + '/' + 'film_average_ratings_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in average_ratings]
        f.close()
        
        f = open(dir1 + '/' + 'film_rating_counts_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in rating_counts]
        f.close()
        
        f = open(dir1 + '/' + 'film_metascores_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in metascores]
        f.close()
        
        f = open(dir1 + '/' + 'film_critic_review_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in critic_review]
        f.close()
        
        f = open(dir1 + '/' + 'film_user_review_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in user_review]
        f.close()
        
        
        
        
        
        dir2 = 'ProcessedData/2_max_impact_distributions'
        if not os.path.exists(dir2):
            os.makedirs(dir2)

        f = open(dir2 + '/' + 'film_max_average_ratings_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in max_average_ratings]
        f.close()
        
        f = open(dir2 + '/' + 'film_max_rating_counts_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in max_rating_counts]
        f.close()
        
        f = open(dir2 + '/' + 'film_max_metascores_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in max_metascores]
        f.close()
        
        f = open(dir2 + '/' + 'film_max_critic_review_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in max_critic_review]
        f.close()
        
        f = open(dir2 + '/' + 'film_max_user_review_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in max_user_review]
        f.close()





        dir3 = 'ProcessedData/3_inflation_curves'
        if not os.path.exists(dir3):
            os.makedirs(dir3)

        x_average_ratings_year = get_dict_data(average_ratings_year) 
        x_rating_counts_year   = get_dict_data(rating_counts_year)   
        x_metascores_year      = get_dict_data(metascores_year)      
        x_critic_review_year   = get_dict_data(critic_review_year)   
        x_user_review_year     = get_dict_data(user_review_year)   


        f = open(dir3 + '/' + 'film_yearly_average_ratings_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in x_average_ratings_year]
        f.close()
        
        f = open(dir3 + '/' + 'film_yearly_rating_counts_dist_'   + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in x_rating_counts_year]
        f.close()
        
        f = open(dir3 + '/' + 'film_yearly_metascores_dist_'      + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in x_metascores_year]
        f.close()
        
        f = open(dir3 + '/' + 'film_yearly_critic_review_dist_' + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in x_critic_review_year]
        f.close()
        
        f = open(dir3 + '/' + 'film_yearly_user_review_dist_'     + label + '.dat', 'w')
        [f.write(str(a) + '\n') for a in x_user_review_year]
        f.close()
 
 
 
 
 
 

       
        
        
        
if __name__ == '__main__':         

    source = 'Data/Film/'

    t1 = time.time()
    getImpactDistribution(source)
    t2 = time.time()
    print 'This took ', round(t2-t1, 2), ' seconds.'


    
    
    
    
    
    

