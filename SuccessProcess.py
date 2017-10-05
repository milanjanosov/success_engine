import os 
import sys
import gzip
import time
import numpy as np
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory
from CareerTrajectory.careerTrajectory import MultipleImpactCareerTrajectory


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
  


def write_yearly_avgs(dict_data, filename):

    if len(dict_data) >  0:
        f = open(filename, 'w')
        
        for year, values in  dict_data.items(): 
            f.write(str(year) + '\t' + str(np.mean(values)) + '\t' + str(np.std(values)) + '\n' )  
        
        f.close()



def parse_norm_factors(filename):
                
    norm_factors = {}
    for line in open(filename):   
        fields = line.strip().split('\t')
        norm_factors[float(fields[0])] = float(fields[1])
                        
    return norm_factors
    
 
def write_exponents(exponents, filename):

    f = open(filename, 'w')
    for e in exponents:
        f.write(str(e[0]) + '\t' + str(e[1]) + '\t'  + str(e[2]) + '\t'  + str(e[3]) + '\n'  )
    f.close()
    
    
    

def process_simple_career_trajectories(normalized, randomized):


    



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
        
        print label
        
        
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
        
        
        career_length = []
        multi_impacts = []
        
        
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


        average_rat_norm   = {} 
        rating_counts_norm = {}
        metascore_norm     = {}     
        critic_review_norm = {}   
        user_review_norm   = {}   
        
        
        exponents_avg_rating   = []
        exponents_rating_count = []
        exponents_metascore    = []
        exponents_ciric_rev    = []
        exponents_user_rev     = []
        
        
        
        combined_factors   = []      



        if normalized:
         
            dir6 = 'ProcessedData/6_yearly_averages'
          
            if 'film' in field:                       
                average_rat_norm   = parse_norm_factors( dir6 + '/' + field + '_yearly_average_avg_rating_'    + label + '.dat' ) 
                rating_counts_norm = parse_norm_factors( dir6 + '/' + field + '_yearly_average_rating_count_'  + label + '.dat' )   
                metascore_norm     = parse_norm_factors( dir6 + '/' + field + '_yearly_average_metascore_'     + label + '.dat' )      
                critic_review_norm = parse_norm_factors( dir6 + '/' + field + '_yearly_average_critic_review_' + label + '.dat' )   
                user_review_norm   = parse_norm_factors( dir6 + '/' + field + '_yearly_average_user_review_'   + label + '.dat' )   
                
                combined_factors   = [average_rat_norm, rating_counts_norm, metascore_norm, critic_review_norm, user_review_norm]        



            if 'music' in field:
                rating_counts_norm = parse_norm_factors( dir6 + '/' + field + '_yearly_average_rating_count_'  + label + '.dat' )                   

        


          

        for filename in files:
        
            
            
            ijk += 1
            print ijk, '/', nnn
            
            
            
            
            #avg ratings
            if 'literature' in field or 'film' in field:

                impact_id = 0

            
                pista_avg_rating = SimpleCareerTrajectory(filename, 'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, impact_id, average_rat_norm, randomized)
                average_ratings  += pista_avg_rating.getImpactValues()
                add_max_impact(max_average_ratings, pista_avg_rating.getMaxImpact())
                

                time_series = pista_avg_rating.getYearlyProducts()
                add_time_series(average_ratings_year, time_series)
                 
                
                #career_length.append(pista_avg_rating.getCareerLength())         
                
                (NN_all, NN_rand, N) = pista_avg_rating.getRankOfMaxImpact()  
                print NN_all, N
                if 'nan' not in str(NN_rand):
                    NN_all_avg_rating  += [(n, N) for n in NN_all ]
                    NN_rand_avg_rating.append((NN_rand, N))

                
                exponents = pista_avg_rating.get_exponents()
                if 0 != exponents:           
                    exponents_avg_rating.append(exponents)
                
 
                                                                                            
                gyurika = MultipleImpactCareerTrajectory(filename, 'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, combined_factors, randomized)
                multi_impacts += gyurika.getImpactValues()




                      
            # rating counts
            if 'music' in field or 'film' in field:
            
                impact_id = 1
                if 'music' in field:
                    impact_id = 0
               
                try:

                    pista_ratingcnt = SimpleCareerTrajectory(filename, 'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, impact_id, rating_counts_norm, randomized)
                    rating_counts   += pista_ratingcnt.getImpactValues()  
                    
                    add_max_impact(max_rating_counts, pista_ratingcnt.getMaxImpact())   
                    time_series = pista_ratingcnt.getYearlyProducts()
                    add_time_series(rating_counts_year, time_series)    


                    (NN_all, NN_rand, N) = pista_ratingcnt.getRankOfMaxImpact()  
                    if 'nan' not in str(NN_rand):
                        NN_all_rating_count  += [(n, N) for n in NN_all ]
                        NN_rand_rating_count.append((NN_rand, N))
                 
                 
                    exponents = pista_ratingcnt.get_exponents()
                    if 0 != exponents:           
                        exponents_rating_count.append(exponents)
                     
                 
                    career_length.append(pista_ratingcnt.getCareerLength())         
                    
                except:
                    error.write(filename + '\t' + field  + '\t' + label + '\n')


                


                                    
            # metascore
            if  'film' in field:
            
                pista_meta  = SimpleCareerTrajectory(filename, 'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, 2, metascore_norm, randomized)
                metascores  += pista_meta.getImpactValues() 
                add_max_impact(max_metascores, pista_meta.getMaxImpact())           
                
                time_series = pista_meta.getYearlyProducts()
                add_time_series(metascores_year, time_series)      


                (NN_all, NN_rand, N) = pista_meta.getRankOfMaxImpact()  
                if 'nan' not in str(NN_rand):
                    NN_all_metascores  += [(n, N) for n in NN_all ]
                    NN_rand_metascores.append((NN_rand, N))
                         
                 
                exponents = pista_meta.get_exponents()
                if 0 != exponents:           
                    exponents_metascore.append(exponents) 
                 
                 
                       
                       
                        
            # critic reviews
            if 'film' in field:
            
                pista_critic  = SimpleCareerTrajectory(filename,  'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, 3, critic_review_norm, randomized)
                critic_review += pista_critic.getImpactValues()                   
                add_max_impact(max_critic_review, pista_critic.getMaxImpact())         

                time_series  = pista_critic.getYearlyProducts()
                add_time_series(critic_review_year, time_series)               
                   
                 
                (NN_all, NN_rand, N) = pista_critic.getRankOfMaxImpact()  
                if 'nan' not in str(NN_rand):
                    NN_all_critic_review  += [(n, N) for n in NN_all ]
                    NN_rand_critic_review.append((NN_rand, N))   
                  

                exponents = pista_critic.get_exponents()
                if 0 != exponents:           
                    exponents_ciric_rev.append(exponents) 

                
                
                       
                       
            # user reviews
            if 'film' in field:
            
                pista_user   = SimpleCareerTrajectory(filename,  'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, 4, user_review_norm, randomized)
                user_review  += pista_user.getImpactValues()
                add_max_impact(max_user_review, pista_user.getMaxImpact())
                
                time_series = pista_user.getYearlyProducts()
                add_time_series(user_review_year, time_series)       


                (NN_all, NN_rand, N) = pista_user.getRankOfMaxImpact()  
                if 'nan' not in str(NN_rand):
                    NN_all_user_review  += [(n, N) for n in NN_all ]
                    NN_rand_user_review.append((NN_rand, N))   
                             
             
                exponents = pista_user.get_exponents()
                if 0 != exponents:           
                    exponents_user_rev.append(exponents) 

                
                            




        ''' ------------------ impact distros ------------------ '''
        
        root = 'ProcessedData'
        if normalized:
            root = root + 'Normalized'
        if randomized:
            root = root + 'Randomized'
     
        
        if not randomized:      
          
            dir1 = root + '/1_impact_distributions'
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
                
            
            
    
            
        ''' ------------------  max distros  ------------------ '''            
            
        if not randomized:    
        
            dir2 = root + '/2_max_impact_distributions'
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





        ''' ------------------ inflation curves ------------------ '''

        if not normalized:  

            dir3 = root + '/3_inflation_curves'
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
                                            
         
     
     
     
     
        ''' ------------------ N*/N distros ------------------ '''
     
        dir4 = root + '/4_NN_rank_N'
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
                
            
       
   
        ''' ------------------ career length ------------------ '''    
       
        if not normalized and not randomized:
           
            dir5 = root + '/5_career_length'
            if not os.path.exists(dir5):
                os.makedirs(dir5)

            if len(career_length) > 0:
                f = open(dir5 + '/' + field + '_career_length_' + label + '.dat', 'w')
                [f.write(str(a) + '\n') for a in career_length]
                f.close()




        ''' ------------------ yearly avg ------------------ '''
        
        
        dir6 = root + '/6_yearly_averages'
        if not os.path.exists(dir6):
            os.makedirs(dir6)
      
        write_yearly_avgs(average_ratings_year, dir6 + '/' + field + '_yearly_average_avg_rating_'    + label + '.dat') 
        write_yearly_avgs(rating_counts_year  , dir6 + '/' + field + '_yearly_average_rating_count_'  + label + '.dat')   
        write_yearly_avgs(metascores_year     , dir6 + '/' + field + '_yearly_average_metascore_'     + label + '.dat')      
        write_yearly_avgs(critic_review_year  , dir6 + '/' + field + '_yearly_average_critic_review_' + label + '.dat')   
        write_yearly_avgs(user_review_year    , dir6 + '/' + field + '_yearly_average_user_review_'   + label + '.dat')   

        
        
          
        
        ''' ------------------ multiple impact measures ------------------ '''
        
        if len(multi_impacts) > 0:
            
            dir7 = root + '/7_multiple_impacts'
            if not os.path.exists(dir7):
                os.makedirs(dir7)
            
            f = open(dir7 + '/' + field + '_multiple_impacts_'  + label + '.dat', 'w')
            [f.write(mm + '\n') for mm in multi_impacts]
            f.close()
            
            
            
            
        ''' ------------------  exponents  ------------------ '''
  

            
        dir8 = root + '/8_exponents'
        if not os.path.exists(dir8):
            os.makedirs(dir8)
            
        
        print exponents_avg_rating    
            
        write_exponents( exponents_avg_rating,   dir8 + '/' + field + '_exponents_avg_rating_' + label + '.dat') 
        write_exponents( exponents_rating_count, dir8 + '/' + field + '_exponents_rating_cnt_' + label + '.dat') 
        write_exponents( exponents_metascore,    dir8 + '/' + field + '_exponents_metascore_'  + label + '.dat') 
        write_exponents( exponents_ciric_rev,    dir8 + '/' + field + '_exponents_critic_rev_' + label + '.dat') 
        write_exponents( exponents_user_rev,     dir8 + '/' + field + '_exponents_user_rev_'   + label + '.dat') 
            
     
        
        
        
        
        
        
        
if __name__ == '__main__':         


    error = open('error_unparsed.dat', 'w')

    t1 = time.time()
    process_simple_career_trajectories(normalized = False, randomized = False)
    process_simple_career_trajectories(normalized = True,  randomized = False)
    process_simple_career_trajectories(normalized = True,  randomized = True)
    t2 = time.time()
    print 'This took ', round(t2-t1, 2), ' seconds.'


    error.close()
    
    
    
    
    

