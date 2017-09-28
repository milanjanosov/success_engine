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
  



def process_simple_career_trajectories():


    



    input_data = [(os.listdir('Data/Music/music-pop-simple-careers'),        'music', 'pop'),
                  (os.listdir('Data/Music/music-electro-simple-careers'),    'music', 'electro'),
                  (os.listdir('Data/Film/film-director-simple-careers'),     'film',  'director'),
                  (os.listdir('Data/Film/film-producer-simple-careers'),     'film',  'producer'),   
                  (os.listdir('Data/Film/film-writer-simple-careers'),       'film',  'writer'),   
                  (os.listdir('Data/Film/film-composer-simple-careers'),     'film',  'composer'),   
                  (os.listdir('Data/Film/film-art-director-simple-careers'), 'film',  'art-director'),   
                 ]

    



    for (files, field, label) in input_data:

        ijk = 0
        nnn = len(files)
        
        
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
        
        
        NN_all_avg_rating  = []
        NN_rand_avg_rating = []    
        
        NN_all_rating_count  = []
        NN_rand_rating_count = []    

        NN_all_metascores  = []
        NN_rand_metascores = []    

        NN_all_critic_review  = []
        NN_rand_critic_review = []    

        NN_all_user_review  = []
        NN_rand_user_review = []    

        
        

        for filename in files[0:10]:
        
            
            
            ijk += 1
            print ijk, '/', nnn
            
            #avg ratings
            if 'literature' in field or 'film' in field:

                impact_id = 0
            
                pista_avg_rating = SimpleCareerTrajectory(filename, 'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, impact_id)
                average_ratings  += pista_avg_rating.getImpactValues()
                add_max_impact(max_average_ratings, pista_avg_rating.getMaxImpact())
                
                time_series = pista_avg_rating.getYearlyProducts()
                add_time_series(average_ratings_year, time_series)
                         
                
                (NN_all, NN_rand, N) = pista_avg_rating.getRankOfMaxImpact()  
                if 'nan' not in str(NN_rand):
                    NN_all_avg_rating  += [(n, N) for n in NN_all ]
                    NN_rand_avg_rating.append((NN_rand, N))


            
            # rating counts
            if 'music' in field or 'film' in field:
            
                
            
                impact_id = 1
                if 'music' in field:
                    impact_id = 0
               
                try:

                    pista_ratingcnt = SimpleCareerTrajectory(filename, 'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, impact_id)
                    rating_counts   += pista_ratingcnt.getImpactValues()  
                    
                    add_max_impact(max_rating_counts, pista_ratingcnt.getMaxImpact())   
                    time_series = pista_ratingcnt.getYearlyProducts()
                    add_time_series(rating_counts_year, time_series)    


                    (NN_all, NN_rand, N) = pista_ratingcnt.getRankOfMaxImpact()  
                    if 'nan' not in str(NN_rand):
                        NN_all_rating_count  += [(n, N) for n in NN_all ]
                        NN_rand_rating_count.append((NN_rand, N))
                    
                except:
                    error.write(filename + '\t' + field  + '\t' + label + '\n')

            
                              
            # metascore
            if  'film' in field:
            
                pista_meta  = SimpleCareerTrajectory(filename, 'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, 2)
                metascores  += pista_meta.getImpactValues() 
                add_max_impact(max_metascores, pista_meta.getMaxImpact())           
                
                time_series = pista_meta.getYearlyProducts()
                add_time_series(metascores_year, time_series)      


                (NN_all, NN_rand, N) = pista_meta.getRankOfMaxImpact()  
                if 'nan' not in str(NN_rand):
                    NN_all_metascores  += [(n, N) for n in NN_all ]
                    NN_rand_metascores.append((NN_rand, N))
                         
                 
                        
            # critic reviews
            if 'film' in field:
            
                pista_critic  = SimpleCareerTrajectory(filename,  'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, 3)
                critic_review += pista_critic.getImpactValues()                   
                add_max_impact(max_critic_review, pista_critic.getMaxImpact())         

                time_series  = pista_critic.getYearlyProducts()
                add_time_series(critic_review_year, time_series)               
                   
                 
                (NN_all, NN_rand, N) = pista_critic.getRankOfMaxImpact()  
                if 'nan' not in str(NN_rand):
                    NN_all_critic_review  += [(n, N) for n in NN_all ]
                    NN_rand_critic_review.append((NN_rand, N))   
                  
                       
            # user reviews
            if 'film' in field:
            
                pista_user   = SimpleCareerTrajectory(filename,  'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, 4)
                user_review  += pista_user.getImpactValues()
                add_max_impact(max_user_review, pista_user.getMaxImpact())
                
                time_series = pista_user.getYearlyProducts()
                add_time_series(user_review_year, time_series)       


                (NN_all, NN_rand, N) = pista_user.getRankOfMaxImpact()  
                if 'nan' not in str(NN_rand):
                    NN_all_user_review  += [(n, N) for n in NN_all ]
                    NN_rand_user_review.append((NN_rand, N))   
                             
            





        dir1 = 'ProcessedData/1_impact_distributions'
        if not os.path.exists(dir1):
            os.makedirs(dir1)


        if len(average_ratings) > 0:
            f = open(dir1 + '/' + field + '_average_ratings_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in average_ratings]
            f.close()

        if len(rating_counts) > 0:            
            f = open(dir1 + '/' + field + '_rating_counts_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in rating_counts]
            f.close()

        if len(metascores) > 0:            
            f = open(dir1 + '/' + field + '_metascores_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in metascores]
            f.close()
 
        if len(critic_review) > 0:           
            f = open(dir1 + '/' + field + '_critic_review_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in critic_review]
            f.close()

        if len(user_review) > 0:            
            f = open(dir1 + '/' + field + '_user_review_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in user_review]
            f.close()
            
            
            
            
            
        dir2 = 'ProcessedData/2_max_impact_distributions'
        if not os.path.exists(dir2):
            os.makedirs(dir2)

        if len(max_average_ratings) > 0:
            f = open(dir2 + '/' + field + '_max_average_ratings_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in max_average_ratings]
            f.close()

        if len(max_rating_counts) > 0:            
            f = open(dir2 + '/' + field + '_max_rating_counts_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in max_rating_counts]
            f.close()

        if len(max_metascores) > 0:            
            f = open(dir2 + '/' + field + '_max_metascores_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in max_metascores]
            f.close()

        if len(max_critic_review) > 0:            
            f = open(dir2 + '/' + field + '_max_critic_review_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in max_critic_review]
            f.close()

        if len(max_user_review) > 0:            
            f = open(dir2 + '/' + field + '_max_user_review_dist_' + label + '.dat', 'w')
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


        if len(x_average_ratings_year) > 0:
            f = open(dir3 + '/' + field + '_yearly_average_ratings_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in x_average_ratings_year]
            f.close()
 
        if len(x_rating_counts_year) > 0:            
            f = open(dir3 + '/' + field + '_yearly_rating_counts_dist_'   + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in x_rating_counts_year]
            f.close()
 
        if len(x_metascores_year) > 0:           
            f = open(dir3 + '/' + field + '_yearly_metascores_dist_'      + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in x_metascores_year]
            f.close()

        if len(x_critic_review_year) > 0:            
            f = open(dir3 + '/' + field + '_yearly_critic_review_dist_' + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in x_critic_review_year]
            f.close()

        if len(x_user_review_year) > 0:            
            f = open(dir3 + '/' + field + '_yearly_user_review_dist_'     + label + '.dat', 'w')
            [f.write(str(a) + '\n') for a in x_user_review_year]
            f.close()
     
     
     
     
     
        dir4 = 'ProcessedData/4_NN_rank_N'
        if not os.path.exists(dir4):
            os.makedirs(dir4) 
            
        if len(NN_rand_avg_rating) > 0:     
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_rand_avg_rating_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_rand_avg_rating]
            f.close() 
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_all_avg_rating_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_all_avg_rating]
            f.close()  

        if len(NN_rand_rating_count) > 0:
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_rand_rating_count_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_rand_rating_count]
            f.close() 
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_all_rating_count_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_all_rating_count]
            f.close()       

        if len(NN_rand_metascores) > 0:           
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_all_metascores_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_all_metascores]
            f.close() 
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_rand_metascores_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_rand_metascores]
            f.close()           

        if len(NN_rand_critic_review) > 0:           
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_rand_critic_review_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_rand_critic_review]
            f.close() 
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_all_critic_review_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_all_critic_review]
            f.close()        

        if len(NN_rand_user_review) > 0:           
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_rand_user_review_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_rand_user_review]
            f.close() 
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_all_user_review_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_all_user_review]
            f.close()        
           
       
       
       
        
        
        
        
if __name__ == '__main__':         


    error = open('error_unparsed.dat', 'w')

    t1 = time.time()
    process_simple_career_trajectories()
    t2 = time.time()
    print 'This took ', round(t2-t1, 2), ' seconds.'


    error.close()
    
    
    
    
    

