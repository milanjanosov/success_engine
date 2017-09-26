import os 
import sys
import gzip
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn 
from multiprocessing import Process
from scipy import stats
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory
from CareerTrajectory.careerTrajectory import getDistribution
from CareerTrajectory.careerTrajectory import getBinnedDistribution


''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''          GET THE DISTRIBUTION OF ALL SUCCESS MEASURES          '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   


def align_plot(ax):

    font_tick = 14   

    for i in range(len(ax)):
        for j in range(len(ax[0])):
            ax[i,j].legend(loc = 'left', fontsize = font_tick) 
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].get_xaxis().tick_bottom()
            ax[i,j].get_yaxis().tick_left()
            ticklines  = ax[i,j].get_xticklines()  + ax[i,j].get_yticklines()
            gridlines  = ax[i,j].get_xgridlines()  + ax[i,j].get_ygridlines()
            ticklabels = ax[i,j].get_xticklabels() + ax[i,j].get_yticklabels()
            for line in ticklines:
                line.set_linewidth(1)

            for line in gridlines:
                line.set_linestyle('-.')

            ax[i,j].tick_params(labelsize = font_tick) 




def getImpactDistribution(source):


    title_font = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(15, 33))
    st = f.suptitle("IMDb impact distributions", fontsize=title_font)


    input_data = [(os.listdir(source + 'film-director-simple-careers'), 'k',  'director'),
                  (os.listdir(source + 'film-producer-simple-careers'), 'b',  'producer'),   
                  (os.listdir(source + 'film-writer-simple-careers'),   'r',  'writer'),   
                  (os.listdir(source + 'film-composer-simple-careers'), 'g',  'composer'),   
                  (os.listdir(source + 'film-art-director-simple-careers'), 'y',  'art-director'),   
                 ]

    
    num_of_bins = 12
    average_ratings = []
    rating_counts   = []
    metascores      = []
    critic_review   = []
    user_review     = []



    for (files, color, label) in input_data:

        for filename in files:
            
            #avg ratings
            pista_avg_rating = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 0)
            average_ratings  += pista_avg_rating.getImpactValues()

            # rating counts
            pista_ratingcnt = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 1)
            rating_counts   += pista_ratingcnt.getImpactValues()
                       
            # metascore
            pista_meta  = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 2)
            metascores  += pista_meta.getImpactValues() 
             
            # critic reviews
            pista_critic  = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 3)
            critic_review += pista_critic.getImpactValues()
                       
            # user reviews
            pista_user   = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 4)
            user_review  += pista_user.getImpactValues()



        # plot avg ratings
        x_average_ratings,  p_average_ratings = getDistribution(average_ratings, True)
        bx_average_ratings, bp_average_ratings, bperr_average_ratings = getBinnedDistribution(x_average_ratings, p_average_ratings, num_of_bins)

        ax[0,0].set_title('IMDb - average rating', fontsize = 20)
        ax[0,0].plot(x_average_ratings,  p_average_ratings, color + 'o', alpha = 0.4, label = label + ', ' + str(len(average_ratings)))
        ax[0,0].errorbar((bx_average_ratings[1:] + bx_average_ratings[:-1])/2, bp_average_ratings, yerr=bperr_average_ratings, fmt=color + '-', linewidth = 2)
        
        
        # plot rating counts  
        x_rating_counts,  p_rating_counts = getDistribution(rating_counts, True)
        bx_rating_counts, bp_rating_counts, bperr_rating_counts = getBinnedDistribution(x_rating_counts, p_rating_counts, num_of_bins)

        ax[0,1].set_title('IMDb - rating count', fontsize = 20)
        ax[0,1].set_xscale('log')
        ax[0,1].set_yscale('log')
        ax[0,1].plot(x_rating_counts,  p_rating_counts, color + 'o', alpha = 0.4, label = label + ', ' + str(len(rating_counts)))
        #ax[0,1].errorbar((bx_rating_counts[1:] + bx_rating_counts[:-1])/2, bp_rating_counts, yerr=bperr_rating_counts, fmt='b-', linewidth = 2)

        
        # plot metascores
        x_metascores,  p_metascores = getDistribution(metascores, True)
        ax[0,2].set_title('IMDb - metascores', fontsize = 20)
        ax[0,2].plot(x_metascores,  p_metascores, color + 'o', alpha = 0.4, label = label + ', ' + str(len(metascores)))
        
        
        # plot critic review count
        x_critic_review,  p_critic_review = getDistribution(critic_review, True)
        ax[1,0].set_title('IMDb - critic_review', fontsize = 20)
        ax[1,0].set_xscale('log')
        ax[1,0].set_yscale('log')
        ax[1,0].plot(x_critic_review, p_critic_review, color + 'o', alpha = 0.4, label = label + ', ' + str(len(critic_review)))
        
        
        # plot user review count
        x_user_review,  p_user_review = getDistribution(user_review, True)
        ax[1,1].set_title('IMDb - user_review', fontsize = 20)
        ax[1,1].set_xscale('log')
        ax[1,1].set_yscale('log')
        ax[1,1].plot(x_user_review, p_user_review, color + 'o', alpha = 0.4, label = label + ', ' + str(len(user_review)))
    
    
    align_plot(ax)
    plt.show()
   
 
 
 
 
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''          GET THE DISTRIBUTION OF MAX SUCCESS MEASURES          '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   
 
 
 
def add_max_impact(lista, maxvalue):

    if 'nan' != maxvalue:
        lista.append(maxvalue)
 
 
def getMaxImpactDistribution(source):


    title_font = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(15, 33))
    st = f.suptitle("IMDb Max impact distributions", fontsize=title_font)


    input_data = [(os.listdir(source + 'film-director-simple-careers'), 'k',  'director'),
                  (os.listdir(source + 'film-producer-simple-careers'), 'b',  'producer'),   
                  (os.listdir(source + 'film-writer-simple-careers'),   'r',  'writer'),   
                  (os.listdir(source + 'film-composer-simple-careers'), 'g',  'composer'),   
                  (os.listdir(source + 'film-art-director-simple-careers'), 'y',  'art-director'),   
                 ]

    
    num_of_bins = 12
    average_ratings = []
    rating_counts   = []
    metascores      = []
    critic_review   = []
    user_review     = []



    for (files, color, label) in input_data:

        for filename in files[0:1000]:
            
            #avg ratings
            pista_avg_rating = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 0)  
            add_max_impact(average_ratings, pista_avg_rating.getMaxImpact())
                           
            # rating counts
            pista_ratingcnt = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 1)
            add_max_impact(rating_counts, pista_ratingcnt.getMaxImpact())
               
            # metascore
            pista_meta  = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 2)
            add_max_impact(metascores, pista_meta.getMaxImpact())          
            
            # critic reviews
            pista_critic  = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 3)
            add_max_impact(critic_review, pista_critic.getMaxImpact())
              
            # user reviews
            pista_user   = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 4)
            add_max_impact(user_review, pista_user.getMaxImpact())
 



        # plot avg ratings
        x_average_ratings,  p_average_ratings = getDistribution(average_ratings, True)
        bx_average_ratings, bp_average_ratings, bperr_average_ratings = getBinnedDistribution(x_average_ratings, p_average_ratings, num_of_bins)

        ax[0,0].set_title('IMDb - average rating', fontsize = 20)
        ax[0,0].plot(x_average_ratings,  p_average_ratings, color + 'o', alpha = 0.4, label = label + ', ' + str(len(average_ratings)))
        ax[0,0].errorbar((bx_average_ratings[1:] + bx_average_ratings[:-1])/2, bp_average_ratings, yerr=bperr_average_ratings, fmt=color + '-', linewidth = 2)
        
        
        # plot rating counts  
        x_rating_counts,  p_rating_counts = getDistribution(rating_counts, True)
        bx_rating_counts, bp_rating_counts, bperr_rating_counts = getBinnedDistribution(x_rating_counts, p_rating_counts, num_of_bins)

        ax[0,1].set_title('IMDb - rating count', fontsize = 20)
        ax[0,1].set_xscale('log')
        ax[0,1].set_yscale('log')
        ax[0,1].plot(x_rating_counts,  p_rating_counts, color + 'o', alpha = 0.4, label = label + ', ' + str(len(rating_counts)))

      
        # plot metascores
        x_metascores,  p_metascores = getDistribution(metascores, True)
        ax[0,2].set_title('IMDb - metascores', fontsize = 20)
        ax[0,2].plot(x_metascores,  p_metascores, color + 'o', alpha = 0.4, label = label + ', ' + str(len(metascores)))
        
        
        # plot critic review count
        x_critic_review,  p_critic_review = getDistribution(critic_review, True)
        ax[1,0].set_title('IMDb - critic_review', fontsize = 20)
        ax[1,0].set_xscale('log')
        ax[1,0].set_yscale('log')
        ax[1,0].plot(x_critic_review, p_critic_review, color + 'o', alpha = 0.4, label = label + ', ' + str(len(critic_review)))
        
        
        # plot user review count
        x_user_review,  p_user_review = getDistribution(user_review, True)
        ax[1,1].set_title('IMDb - user_review', fontsize = 20)
        ax[1,1].set_xscale('log')
        ax[1,1].set_yscale('log')
        ax[1,1].plot(x_user_review, p_user_review, color + 'o', alpha = 0.4, label = label + ', ' + str(len(user_review)))
 
    
    
    align_plot(ax)
    plt.show() 
 




''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''                     GET INFLATION CURVES                       '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   
   

def add_time_series(impacts, time_series):

    for year, events in time_series.items():
        if year not in impacts:
            impacts[year] = events
        else:
            impacts[year] += events  


def get_dict_data(impacts):

    x = []
    y = []
    for year, impact in impacts.items():
        for i in impact:
            x.append(year)
            y.append(i)

    return np.asarray(x), np.asarray(y)
  
  
def get_movie_numbers(impacts):

    x = []
    y = []
    for year, impact in impacts.items():
        x.append(year)
        y.append(len(impact))

    return np.asarray(x), np.asarray(y)      
       
  
def getInflationCurves(source):



    title_font = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(35, 18))
    st = f.suptitle("IMDb Inflation of impact measures", fontsize=title_font)


    input_data = [(os.listdir(source + 'film-director-simple-careers'), 'k',  'director'),
                  (os.listdir(source + 'film-producer-simple-careers'), 'b',  'producer'),   
                  (os.listdir(source + 'film-writer-simple-careers'),   'r',  'writer'),   
                  (os.listdir(source + 'film-composer-simple-careers'), 'g',  'composer'),   
                  (os.listdir(source + 'film-art-director-simple-careers'), 'y',  'art-director'),   
                 ]



    
    num_of_bins = 20
    average_ratings_year = {}
    rating_counts_year   = {}
    metascores_year      = {}
    critic_review_year   = {}
    user_review_year     = {}


    for (files, color, label) in input_data:

        for filename in files[0:10000]:
            

            
            #avg ratings
            pista_avg_rating = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 0)            
            time_series = pista_avg_rating.getYearlyProducts()
            add_time_series(average_ratings_year, time_series)
 
            
           
            # rating counts
            pista_ratingcnt = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 1)          
            time_series = pista_ratingcnt.getYearlyProducts()
            add_time_series(rating_counts_year, time_series)


               
            # metascore
            pista_meta  = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 2)
            time_series = pista_meta.getYearlyProducts()
            add_time_series(metascores_year, time_series)


            
            # critic reviews
            pista_critic = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 3)         
            time_series  = pista_critic.getYearlyProducts()
            add_time_series(critic_review_year, time_series)
            
                          
            # user reviews
            pista_user  = SimpleCareerTrajectory(filename, source + '/film-' + label + '-simple-careers/' + filename, 4)
            time_series = pista_user.getYearlyProducts()
            add_time_series(user_review_year, time_series)
     

       


        # plot avg ratings
        x_average_ratings_year, y_average_ratings_year = get_dict_data(average_ratings_year)       
        bx_average_ratings_year, bp_average_ratings_year, bperr_average_ratings_year = getBinnedDistribution(x_average_ratings_year, y_average_ratings_year, num_of_bins)
        
        ax[0,0].set_title('IMDb - average rating', fontsize = 20)
        ax[0,0].plot(x_average_ratings_year,  y_average_ratings_year, color + 'o', alpha = 0.001, label = label + ', ' + str(len(average_ratings_year)))
        ax[0,0].errorbar((bx_average_ratings_year[1:] + bx_average_ratings_year[:-1])/2, bp_average_ratings_year, yerr=bperr_average_ratings_year, fmt=color + '-', linewidth = 3)       
    
    
    
        #plot rating counts
        x_rating_counts_year, y_rating_counts_year = get_dict_data(rating_counts_year)          
        bx_rating_counts_year, bp_rating_counts_year, bperr_rating_counts_year = getBinnedDistribution(x_rating_counts_year, y_rating_counts_year, num_of_bins)
        
        ax[0,1].set_title('IMDb - rating count', fontsize = 20)
        ax[0,1].plot(x_rating_counts_year,  y_rating_counts_year, color + 'o', alpha = 0.001, label = label + ', ' + str(len(rating_counts_year)))
        ax[0,1].errorbar((bx_rating_counts_year[1:] + bx_rating_counts_year[:-1])/2, bp_rating_counts_year, yerr=bperr_rating_counts_year, fmt=color + '-', linewidth = 3)       
        
        
        
        # plot number of movies
        x_num_of_movies_year,  y_num_of_movies_year = get_movie_numbers(average_ratings_year)       
        bx_num_of_movies_year, bp_num_of_movies_year, bperr_num_of_movies_year = getBinnedDistribution(x_num_of_movies_year, y_num_of_movies_year, num_of_bins)
        
        ax[1,0].set_title('IMDb - number of movies', fontsize = 20)
        ax[1,0].plot(x_num_of_movies_year,  y_num_of_movies_year, color + 'o', alpha = 0.001, label = label + ', ' + str(len(average_ratings_year)))
        ax[1,0].errorbar((bx_num_of_movies_year[1:] + bx_num_of_movies_year[:-1])/2, bp_num_of_movies_year, yerr=bperr_num_of_movies_year, fmt=color + '-', linewidth = 3)  
    
    
    
        # plot number of movies
        x_metascores_year,  y_metascores_year = get_movie_numbers(metascores_year)       
        bx_metascores_year, bp_metascores_year, bperr_metascores_year = getBinnedDistribution(x_metascores_year, y_metascores_year, num_of_bins)
        
        ax[1,1].set_title('IMDb - metascores_year', fontsize = 20)
        ax[1,1].plot(x_metascores_year,  y_metascores_year, color + 'o', alpha = 0.2, label = label + ', ' + str(len(metascores_year)))
        ax[1,1].errorbar((bx_metascores_year[1:] + bx_metascores_year[:-1])/2, bp_metascores_year, yerr=bperr_metascores_year, fmt=color + '-', linewidth = 3)  
        
        
        
        # plot critic reviews
        x_critic_review_year,  y_critic_review_year = get_movie_numbers(critic_review_year)       
        bx_critic_review_year, bp_critic_review_year, bperr_critic_review_year = getBinnedDistribution(x_critic_review_year, y_critic_review_year, num_of_bins)
        
        ax[0,2].set_title('IMDb - critic_review_year', fontsize = 20)
        ax[0,2].plot(x_critic_review_year,  y_critic_review_year, color + 'o', alpha = 0.2, label = label + ', ' + str(len(critic_review_year)))
        ax[0,2].errorbar((bx_critic_review_year[1:] + bx_critic_review_year[:-1])/2, bp_critic_review_year, yerr=bperr_critic_review_year, fmt=color + '-', linewidth = 3)  
        
        
        
        # plot user reviews
        x_user_review_year,  y_user_review_year = get_movie_numbers(user_review_year)       
        bx_user_review_year, bp_user_review_year, bperr_user_review_year = getBinnedDistribution(x_user_review_year, y_user_review_year, num_of_bins)
        
        ax[1,2].set_title('IMDb - user_review_year', fontsize = 20)
        ax[1,2].plot(x_user_review_year,  y_user_review_year, color + 'o', alpha = 0.2, label = label + ', ' + str(len(user_review_year)))
        ax[1,2].errorbar((bx_user_review_year[1:] + bx_user_review_year[:-1])/2, bp_user_review_year, yerr=bperr_user_review_year, fmt=color + '-', linewidth = 3)  
        
    
    
    align_plot(ax)    
    plt.show()
       
       
       
        
        
        
if __name__ == '__main__':         

    source = 'Data/Film/simple-careers/'

    if sys.argv[1] == 'impact':
        getImpactDistribution(source)
    elif sys.argv[1] == 'max':
        getMaxImpactDistribution(source)
    elif sys.argv[1] == 'inflation':
        getInflationCurves(source)

    
    
    
    
    
    

