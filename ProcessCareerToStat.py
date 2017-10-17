import os 
import sys
import gzip
import time
import numpy as np
import shutil
from multiprocessing import Process
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory
from CareerTrajectory.careerTrajectory import MultipleImpactCareerTrajectory



 
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
    
        
def write_distr_data(data, filename):
        
    if len(data) > 0:
        f = open(filename, 'w')
        [f.write(str(a) + '\n') for a in data]
        f.close()
    
  
def write_pairs(data, filename):

    if len(data) > 0:

        f = open(filename, 'w')
        for d in data:
            if 'nan' != d[0]:
                f.write(str(d[0]) + '\t' + str(d[1]) + '\n')
        f.close()  
        


def chunkIt(seq, num):

    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out



def process_simple_career_trajectories(args):


    input_data = args[0]
    normalized = args[1]
    randomized = args[2]
    

    

    for (files, field, label) in [input_data]:

        ijk = 0
        nnn = len(files)

        
        average_ratings = []
        rating_counts   = []
        metascores      = []
        critic_review   = []
        user_review     = []
        gross           = []


        max_average_ratings = []
        max_rating_counts   = []
        max_metascores      = []
        max_critic_review   = []
        max_user_review     = []
        max_gross           = []
        
        
        average_ratings_year = {}
        rating_counts_year   = {}
        metascores_year      = {}
        critic_review_year   = {}
        user_review_year     = {}
        gross_year           = {}
        
          
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

        NN_all_gross        = []
        NN_rand_gross       = []


        average_rat_norm   = {} 
        rating_counts_norm = {}
        metascore_norm     = {}     
        critic_review_norm = {}   
        user_review_norm   = {}   
        gross_norm         = {}    
        
        
        p_without_mean_avg_rating = []
        p_without_mean_rating_cnt = []
        p_without_mean_metascore  = []
        p_without_mean_critic_rev = []
        p_without_mean_user_rev   = []
        p_without_mean_gross      = []        
        
        max_avg_rat_N   = []
        max_rat_cnt_N   = []        
        max_metascore_N = []        
        max_crit_rev_N  = []
        max_user_rev_N  = []
        max_gross_N     = []
       
        
        combined_factors = []   
        career_length = []
        multi_impacts = []
           


        if normalized:
         
            dir6 = 'ProcessedData/6_yearly_averages'
          
            if 'film' in field:                       
                average_rat_norm   = parse_norm_factors( dir6 + '/' + field + '_yearly_average_avg_rating_'    + label + '.dat' ) 
                rating_counts_norm = parse_norm_factors( dir6 + '/' + field + '_yearly_average_rating_count_'  + label + '.dat' )   
                metascore_norm     = parse_norm_factors( dir6 + '/' + field + '_yearly_average_metascore_'     + label + '.dat' )      
                critic_review_norm = parse_norm_factors( dir6 + '/' + field + '_yearly_average_critic_review_' + label + '.dat' )   
                user_review_norm   = parse_norm_factors( dir6 + '/' + field + '_yearly_average_user_review_'   + label + '.dat' )   
                gross_norm         = parse_norm_factors( dir6 + '/' + field + '_yearly_average_gross_'         + label + '.dat' )   
                
                combined_factors  = [average_rat_norm, rating_counts_norm, metascore_norm, critic_review_norm, user_review_norm, gross_norm]        


            if 'music' in field:
                rating_counts_norm = parse_norm_factors( dir6 + '/' + field + '_yearly_average_rating_count_'  + label + '.dat' )                   

            
            if 'book' in field:
                average_rat_norm   = parse_norm_factors( dir6 + '/' + field + '_yearly_average_avg_rating_'    + label + '.dat' ) 
                rating_counts_norm = parse_norm_factors( dir6 + '/' + field + '_yearly_average_rating_count_'  + label + '.dat' )   
                metascore_norm     = parse_norm_factors( dir6 + '/' + field + '_yearly_average_metascore_'     + label + '.dat' )      
                
                combined_factors  = [average_rat_norm, rating_counts_norm, metascore_norm]        


          

        for filename in files:
        
                      
            ijk += 1
            print ijk, '/', nnn
            
                                 
            #avg ratings
            if 'book' in field or 'film' in field:
            
            
                impact_id = 0
  
                pista_avg_rating = SimpleCareerTrajectory(filename,'Data/'+field.title()+'/'+field+'-'+label+'-simple-careers/'+filename, impact_id, average_rat_norm, randomized)         
                average_ratings  += pista_avg_rating.getImpactValues()   
                add_max_impact(max_average_ratings, pista_avg_rating.getMaxImpact())
                
                
                time_series = pista_avg_rating.getYearlyProducts()
                add_time_series(average_ratings_year, time_series)
                 
                                       
                (NN_all, NN_rand, N) = pista_avg_rating.getRankOfMaxImpact() 
                if 'nan' not in str(NN_rand):
                    NN_all_avg_rating  += [(n, N) for n in NN_all ]
                    NN_rand_avg_rating.append((NN_rand, N))
       
                max_avg_rat_N.append((pista_avg_rating.getMaxImpact(), pista_avg_rating.getCareerLength()))
   
                p_without_mean_avg_rating += pista_avg_rating.getLogPwithZeroAvg()
                
                gyurika = MultipleImpactCareerTrajectory(filename, 'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, combined_factors, randomized)
                multi_impacts += gyurika.getImpactValues()
                

                    
            # rating counts
            if 'book' in field or 'music' in field or 'film' in field:
            
                impact_id = 1
                if 'music' in field:
                    impact_id = 0
               
                try:
  
                    pista_ratingcnt = SimpleCareerTrajectory(filename,'Data/'+field.title()+'/'+field + '-' + label + '-simple-careers/' + filename, impact_id, rating_counts_norm, randomized)
                    rating_counts   += pista_ratingcnt.getImpactValues()                   
                    add_max_impact(max_rating_counts, pista_ratingcnt.getMaxImpact())   
                    
                    
                    time_series = pista_ratingcnt.getYearlyProducts()
                    add_time_series(rating_counts_year, time_series)    


                    (NN_all, NN_rand, N) = pista_ratingcnt.getRankOfMaxImpact()  
                    if 'nan' not in str(NN_rand):
                        NN_all_rating_count  += [(n, N) for n in NN_all ]
                        NN_rand_rating_count.append((NN_rand, N))
                 
          
                    career_length.append(pista_ratingcnt.getCareerLength())  
                    
                    
                    max_rat_cnt_N.append((pista_ratingcnt.getMaxImpact(), pista_ratingcnt.getCareerLength()))
                    
                    
                    p_without_mean_rating_cnt += pista_ratingcnt.getLogPwithZeroAvg()   
                          
                except:
                    error.write(filename + '\t' + field  + '\t' + label + '\n')

               
                                 
            # metascore
            if 'book' in field or 'film' in field:
                    
                pista_meta  = SimpleCareerTrajectory(filename, 'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, 2, metascore_norm, randomized)
                metascores  += pista_meta.getImpactValues() 
                add_max_impact(max_metascores, pista_meta.getMaxImpact())           
                
                
                time_series = pista_meta.getYearlyProducts()
                add_time_series(metascores_year, time_series)      


                (NN_all, NN_rand, N) = pista_meta.getRankOfMaxImpact()  
                if 'nan' not in str(NN_rand):
                    NN_all_metascores  += [(n, N) for n in NN_all ]
                    NN_rand_metascores.append((NN_rand, N))
                         
      
                max_metascore_N.append((pista_meta.getMaxImpact(), pista_meta.getCareerLength()))
                 
                 
                p_without_mean_metascore += pista_meta.getLogPwithZeroAvg()    
                       
                                          
                        
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
                  

                max_crit_rev_N.append((pista_critic.getMaxImpact(), pista_critic.getCareerLength()))

     
                p_without_mean_critic_rev += pista_critic.getLogPwithZeroAvg()    
                
                                                         
                       
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
                             
             
                max_user_rev_N.append((pista_user.getMaxImpact(), pista_user.getCareerLength()))


                p_without_mean_user_rev += pista_user.getLogPwithZeroAvg()                    
                            


            # gross revenue
            if 'film' in field:
            
                pista_gross   = SimpleCareerTrajectory(filename,  'Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + filename, 5, gross_norm, randomized)
                gross  += pista_gross.getImpactValues()
                add_max_impact(max_gross, pista_gross.getMaxImpact())
                
                time_series = pista_gross.getYearlyProducts()
                add_time_series(gross_year, time_series)       


                (NN_all, NN_rand, N) = pista_gross.getRankOfMaxImpact()  
                if 'nan' not in str(NN_rand):
                    NN_all_gross  += [(n, N) for n in NN_all ]
                    NN_rand_gross.append((NN_rand, N))   
                             
      
                max_gross_N.append((pista_gross.getMaxImpact(), pista_gross.getCareerLength()))


                p_without_mean_gross += pista_gross.getLogPwithZeroAvg()                    
                            
           



        ''' ------------------ impact distros ------------------ '''
        
        root = 'ProcessedData'
        if normalized:
            root = root + 'Normalized'
        if randomized:
            root = root + 'Randomized'
     
 

        #shutil.rmtree('/folder_name')


        
        
        if not randomized:      
          
            dir1 = root + '/1_impact_distributions'
            if not os.path.exists(dir1):
                os.makedirs(dir1)

            write_distr_data(average_ratings, dir1 + '/' + field + '_average_ratings_dist_' + label + '.dat')
            write_distr_data(rating_counts,   dir1 + '/' + field + '_rating_counts_dist_'   + label + '.dat')
            write_distr_data(metascores,      dir1 + '/' + field + '_metascores_dist_'      + label + '.dat')
            write_distr_data(critic_review,   dir1 + '/' + field + '_critic_review_dist_'   + label + '.dat')
            write_distr_data(user_review,     dir1 + '/' + field + '_user_review_dist_'     + label + '.dat')
            write_distr_data(gross,           dir1 + '/' + field + '_gross_dist_'           + label + '.dat')                                                            
            
                 
            
        ''' ------------------  max distros  ------------------ '''            
            
        if not randomized:   
        
            dir2 = root + '/2_max_impact_distributions'
            if not os.path.exists(dir2):
                os.makedirs(dir2) 
        
            write_distr_data(max_average_ratings, dir2 + '/' + field + '_max_average_ratings_dist_' + label + '.dat')
            write_distr_data(max_rating_counts,   dir2 + '/' + field + '_max_rating_counts_dist_'   + label + '.dat' )  
            write_distr_data(max_metascores,      dir2 + '/' + field + '_max_metascores_dist_'      + label + '.dat')   
            write_distr_data(max_critic_review,   dir2 + '/' + field + '_max_metascores_dist_'      + label + '.dat')
            write_distr_data(max_user_review,     dir2 + '/' + field + '_max_user_review_dist_'     + label + '.dat')             
            write_distr_data(max_gross,           dir2 + '/' + field + '_max_gross_dist_'           + label + '.dat'  )             
              
              
                                  
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
            x_gross_year           = get_dict_data(gross_year)   

            write_distr_data(x_average_ratings_year, dir3 + '/' + field + '_yearly_average_ratings_dist_' + label + '.dat')
            write_distr_data(x_rating_counts_year,   dir3 + '/' + field + '_yearly_rating_counts_dist_'   + label + '.dat')            
            write_distr_data(x_metascores_year,      dir3 + '/' + field + '_yearly_metascores_dist_'      + label + '.dat')
            write_distr_data(x_critic_review_year,   dir3 + '/' + field + '_yearly_critic_review_dist_'   + label + '.dat')
            write_distr_data(x_user_review_year,     dir3 + '/' + field + '_yearly_user_review_dist_'     + label + '.dat')            
            write_distr_data(x_gross_year,           dir3 + '/' + field + '_yearly_gross_dist_'           + label + '.dat')

         
                          
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
         
        if len(NN_rand_gross) > 0:           
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_rand_gross_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_rand_gross]
            f.close() 
            f = open(dir4 + '/' + field + '_best_product_NN_ranks_all_gross_' + label + '.dat', 'w')
            [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NN_all_gross]
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
        write_yearly_avgs(gross_year          , dir6 + '/' + field + '_yearly_average_gross_'         + label + '.dat')                  
          
        
        ''' ------------------ multiple impact measures ------------------ '''
        
        if len(multi_impacts) > 0:
            
            dir7 = root + '/7_multiple_impacts'
            if not os.path.exists(dir7):
                os.makedirs(dir7)
            
            f = open(dir7 + '/' + field + '_multiple_impacts_'  + label + '.dat', 'w')
            [f.write(mm + '\n') for mm in multi_impacts]
            f.close()
                        
            
        
        ''' ------------------  log p  ------------------ '''
            
        dir9 = root + '/9_p_without_avg'
        if not os.path.exists(dir9):
            os.makedirs(dir9)
            
   
        write_distr_data(p_without_mean_avg_rating, dir9 + '/' + field + '_p_without_mean_avg_rating_' + label + '.dat')
        write_distr_data(p_without_mean_rating_cnt, dir9 + '/' + field + '_p_without_mean_rating_cnt_' + label + '.dat')        
        write_distr_data(p_without_mean_metascore,  dir9 + '/' + field + '_p_without_mean_metascore_'  + label + '.dat')      
        write_distr_data(p_without_mean_critic_rev, dir9 + '/' + field + '_p_without_mean_critic_rev_' + label + '.dat')        
        write_distr_data(p_without_mean_user_rev,   dir9 + '/' + field + '_p_without_mean_user_rev_'   + label + '.dat')        
        write_distr_data(p_without_mean_gross,      dir9 + '/' + field + '_p_without_mean_gross_'      + label + '.dat')        


        ''' ------------------  career length vs max impact  ------------------ '''
            
        dir10 = root + '/10_career_length_max_impact'
        if not os.path.exists(dir10):
            os.makedirs(dir10)

        write_pairs(max_avg_rat_N,   dir10 + '/career_length_max_avg_rat'   + field + '_' + label + '.dat')
        write_pairs(max_rat_cnt_N,   dir10 + '/career_length_max_rat_cnt'   + field + '_' + label + '.dat')
        write_pairs(max_metascore_N, dir10 + '/career_length_max_metascore' + field + '_' + label + '.dat')
        write_pairs(max_crit_rev_N,  dir10 + '/career_length_max_crit_rev'  + field + '_' + label + '.dat')      
        write_pairs(max_user_rev_N,  dir10 + '/career_length_max_user_rev'  + field + '_' + label + '.dat')        
        write_pairs(max_gross_N,     dir10 + '/career_length_max_gross'     + field + '_' + label + '.dat')        



def run_paralel(normalized, randomized):


    input_data = [(os.listdir('Data/Music/music-pop-simple-careers'),         'music', 'pop'),
                  (os.listdir('Data/Music/music-electro-simple-careers'),     'music', 'electro'),
                  (os.listdir('Data/Film/film-director-simple-careers'),      'film',  'director'),
                  (os.listdir('Data/Film/film-producer-simple-careers'),      'film',  'producer'),   
                  (os.listdir('Data/Film/film-writer-simple-careers'),        'film',  'writer'),   
                  (os.listdir('Data/Film/film-composer-simple-careers'),      'film',  'composer'),   
                  (os.listdir('Data/Film/film-art-director-simple-careers'),  'film',  'art-director'),   
                  (os.listdir('Data/Book/book-authors-simple-careers'),       'book',  'authors')   
                  ]



    Pros = []
    

    for inp in input_data:  
        p = Process(target = process_simple_career_trajectories, args=([inp, normalized, randomized], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()




  
    #process_simple_career_trajectories(normalized = False, randomized = False)        
        
        
if __name__ == '__main__':         


    error = open('error_unparsed.dat', 'w')

    #process_simple_career_trajectories(normalized = False, randomized = False)
    ##process_simple_career_trajectories(normalized = True,  randomized = False)
    #process_simple_career_trajectories(normalized = True,  randomized = True)

    #run_paralel(normalized = False, randomized = False)
    run_paralel(normalized = True,  randomized = False)
    #run_paralel(normalized = True,  randomized = True)

    error.close()
    
    
    
    
    

