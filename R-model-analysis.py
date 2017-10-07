import sys
import os
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import random
from scipy import stats
from CareerTrajectory.careerTrajectory import getDistribution
from CareerTrajectory.careerTrajectory import getBinnedDistribution




''' TODO '''
'''


WEDNESDAY: 



- get career exponent fit -> normalized vs randomized

--- add exp. calculator
--- add autocorrel calculator
--- 6. check goodreads scripts
--- read 2 papers





'''



''' plot helper functions '''

def align_plot(ax):

    font_tick = 15   

    for i in range(len(ax)):
        for j in range(len(ax[0])):
            ax[i,j].grid()
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
            
            
 


''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''          GET THE DISTRIBUTION OF ALL SUCCESS MEASURES          '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   
   
   
def get_imapct_distr():             
            


    ''' ---------------------------------------------- '''
    '''      MOVIE YO                                  '''
    
    
    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]

    num_of_bins = 20
    title_font  = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(25, 15))
    st = f.suptitle("IMDb normalized impact distributions", fontsize=title_font)



    FOLDER = 'ProcessedDataNormalized'


    for (label, color) in professions:
       
        num_car  = str(int(round(len(os.listdir('Data/Film/film-'+ label +'-simple-careers'))/1000.0))) + 'k'

        
        file_avg  = FOLDER + '/1_impact_distributions/film_average_ratings_dist_' + label + '.dat'
        file_cnt  = FOLDER + '/1_impact_distributions/film_rating_counts_dist_'   + label + '.dat'
        file_mets = FOLDER + '/1_impact_distributions/film_metascores_dist_'      + label + '.dat'
        file_crit = FOLDER + '/1_impact_distributions/film_critic_review_dist_'   + label + '.dat'
        file_user = FOLDER + '/1_impact_distributions/film_user_review_dist_'     + label + '.dat'

        average_ratings = np.asarray([round(float(line.strip()),2) for line in open(file_avg)])
        rating_counts   = [round(float(line.strip()),2) for line in open(file_cnt)]
        metascores      = [round(float(line.strip()),1) for line in open(file_mets)]
        critic_review   = [round(float(line.strip()),2) for line in open(file_crit)]
        user_review     = [round(float(line.strip()),2) for line in open(file_user)]
        
      
      
        
        # plot avg ratings
        x_average_ratings,  p_average_ratings = getDistribution(average_ratings, True)
        bx_average_ratings, bp_average_ratings, bperr_average_ratings = getBinnedDistribution(x_average_ratings, p_average_ratings, num_of_bins)
    
        
        ax[0,0].set_title('IMDb - average rating', fontsize = 20)
        ax[0,0].plot(x_average_ratings,  p_average_ratings, color, marker = 'o', alpha = 0.1, linewidth = 0, label = label+ ', ' + str(num_car))
        ax[0,0].errorbar((bx_average_ratings[1:] + bx_average_ratings[:-1])/2, bp_average_ratings, yerr=bperr_average_ratings, fmt=color + '-', linewidth = 2)
        
        
        # plot rating counts  
        x_rating_counts,  p_rating_counts = getDistribution(rating_counts, True)
        bx_rating_counts, bp_rating_counts, bperr_rating_counts = getBinnedDistribution(x_rating_counts, p_rating_counts, num_of_bins)

        ax[0,1].set_title('IMDb - rating count', fontsize = 20)
        ax[0,1].set_xscale('log')
        ax[0,1].set_yscale('log')
        ax[0,1].plot(x_rating_counts,  p_rating_counts, color + 'o', alpha = 0.8, label = label)# + ', ' + str(num_wr))
        #ax[0,1].errorbar((bx_rating_counts[1:] + bx_rating_counts[:-1])/2, bp_rating_counts, yerr=bperr_rating_counts, fmt='b-', linewidth = 2)

        
        # plot metascores
        x_metascores,  p_metascores = getDistribution(metascores, True)
        bx_metascores, bp_metascores, bperr_metascores = getBinnedDistribution(x_metascores,  p_metascores, num_of_bins)
        ax[0,2].set_title('IMDb - metascores', fontsize = 20)
        ax[0,2].plot(x_metascores,  p_metascores, color + 'o', alpha = 0.2, label = label)# + ', ' + str(len(metascores)))
        ax[0,2].errorbar((bx_metascores[1:] + bx_metascores[:-1])/2, bp_metascores, yerr=bperr_metascores, fmt=color + '-', linewidth = 2)
        
        # plot critic review count
        x_critic_review,  p_critic_review = getDistribution(critic_review, True)
        ax[1,0].set_title('IMDb - critic_review', fontsize = 20)
        ax[1,0].set_xscale('log')
        ax[1,0].set_yscale('log')
        ax[1,0].plot(x_critic_review, p_critic_review, color + 'o', alpha = 0.8, label = label )#+ ', ' + str(len(critic_review)))


        
        # plot user review count
        x_user_review,  p_user_review = getDistribution(user_review, True)
        ax[1,1].set_title('IMDb - user_review', fontsize = 20)
        ax[1,1].set_xscale('log')
        ax[1,1].set_yscale('log')
        ax[1,1].plot(x_user_review, p_user_review, color + 'o', alpha = 0.8, label = label)# + ', ' + str(len(user_review)))
    
    


    ''' ---------------------------------------------- '''
    '''      MOVIE YO                                  '''
    
    genres = [('electro', 'k'), ('pop', 'b')]
    
    
    for (genre, color) in genres:

        num_mus  = str(int(round(len(os.listdir('Data/Music/music-'+ genre +'-simple-careers'))/1000.0))) + 'k'
      
        file_music = FOLDER + '/1_impact_distributions/music_rating_counts_dist_' + genre + '.dat'
        average_ratings = np.asarray([round(float(line.strip())) for line in open(file_music)])    
        x_rating_counts,  p_rating_counts = getDistribution(average_ratings, True)    

        print len(average_ratings)

        ax[1,2].set_title('Music - playcount', fontsize = 20)
        ax[1,2].set_xscale('log')
        ax[1,2].set_yscale('log')
        ax[1,2].plot(x_rating_counts,  p_rating_counts, color + 'o', alpha = 0.2, label = genre + ', ' + num_mus)


    align_plot(ax)
    plt.savefig('impact_distributions_normalized.png')
    plt.close()
    #plt.show()          



    



''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''                        GET INFLATION CURVES                    '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   



def get_dict_data(impacts):

    x = []
    y = []
    for i in impacts:
        field = i.split('\t')
        x.append(float(field[0]))
        y.append(float(field[1]))

    return np.asarray(x), np.asarray(y)


def get_num_per_year(impacts):

    
    years = {} 
    for i in impacts:
        field = i.split('\t')
        year  = int(float((field[0])))
        impa  = field[1]
        if year not in years:
            years[year] = [impa]
        else:
            years[year].append(impa)

    x = []
    y = []
    for year, impas in years.items():
        x.append(year)
        y.append(len(impas))

    return np.asarray(x), np.asarray(y)



def get_inflation_curves():


    num_of_bins = 20
    title_font = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(25, 15))
    st = f.suptitle("IMDb Inflation of impact measures", fontsize=title_font)


    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]    
    
    
    for (label, color) in professions:

        
        file_avg_year  = 'ProcessedDataNormalizedRandomized/3_inflation_curves/film_yearly_average_ratings_dist_' + label + '.dat'
        file_cnt_year  = 'ProcessedDataNormalizedRandomized/3_inflation_curves/film_yearly_rating_counts_dist_'   + label + '.dat'
        file_mets_year = 'ProcessedDataNormalizedRandomized/3_inflation_curves/film_yearly_metascores_dist_'      + label + '.dat'
        file_crit_year = 'ProcessedDataNormalizedRandomized/3_inflation_curves/film_yearly_critic_review_dist_'   + label + '.dat'
        file_user_year = 'ProcessedDataNormalizedRandomized/3_inflation_curves/film_yearly_user_review_dist_'     + label + '.dat'

        average_ratings_year = np.asarray([line.strip() for line in open(file_avg_year)])
        rating_counts_year   = np.asarray([line.strip() for line in open(file_cnt_year)])
        metascores_year      = np.asarray([line.strip() for line in open(file_mets_year)])
        critic_review_year   = np.asarray([line.strip() for line in open(file_crit_year)])
        user_review_year     = np.asarray([line.strip() for line in open(file_user_year)])
        

        
        # plot average ratings
        x_average_ratings_year, y_average_ratings_year = get_dict_data(average_ratings_year)       
        bx_average_ratings_year, bp_average_ratings_year, bperr_average_ratings_year = getBinnedDistribution(x_average_ratings_year, y_average_ratings_year, num_of_bins)

        
    
        ax[0,0].set_title('IMDb - average rating', fontsize = 20)
        ax[0,0].errorbar((bx_average_ratings_year[1:] + bx_average_ratings_year[:-1])/2, bp_average_ratings_year, yerr=bperr_average_ratings_year, fmt=color + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)
        
        
        #plot rating counts
        x_rating_counts_year, y_rating_counts_year = get_dict_data(rating_counts_year)          
        bx_rating_counts_year, bp_rating_counts_year, bperr_rating_counts_year = getBinnedDistribution(x_rating_counts_year, y_rating_counts_year, num_of_bins)

        ax[0,1].set_title('IMDb - rating count', fontsize = 20)
        ax[0,1].set_ylim([-3000,20000])
        ax[0,1].errorbar((bx_rating_counts_year[1:] + bx_rating_counts_year[:-1])/2, bp_rating_counts_year, yerr=bperr_rating_counts_year, fmt=color + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)


        
        # plot metascorenumber of movies
        x_metascores_year,  y_metascores_year = get_dict_data(metascores_year)       
        bx_metascores_year, bp_metascores_year, bperr_metascores_year = getBinnedDistribution(x_metascores_year, y_metascores_year, num_of_bins)

        ax[1,1].set_title('IMDb - metascores_year', fontsize = 20)
        ax[1,1].errorbar((bx_metascores_year[1:] + bx_metascores_year[:-1])/2, bp_metascores_year, yerr=bperr_metascores_year, fmt=color + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)

        
        # plot critic reviews
        ax[0,2].set_ylim([-25,150])         
        x_critic_review_year,  y_critic_review_year = get_dict_data(critic_review_year)       
        bx_critic_review_year, bp_critic_review_year, bperr_critic_review_year = getBinnedDistribution(x_critic_review_year, y_critic_review_year, num_of_bins)

        
        ax[0,2].set_title('IMDb - critic_review_year', fontsize = 20)
        ax[0,2].errorbar((bx_critic_review_year[1:] + bx_critic_review_year[:-1])/2, bp_critic_review_year, yerr=bperr_critic_review_year, fmt=color + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)  



        # plot user reviews
        x_user_review_year,  y_user_review_year = get_dict_data(user_review_year)       
        bx_user_review_year, bp_user_review_year, bperr_user_review_year = getBinnedDistribution(x_user_review_year, y_user_review_year, num_of_bins)

        ax[1,2].set_ylim([-15,75])         
        ax[1,2].set_title('IMDb - user_review_year', fontsize = 20)
        ax[1,2].errorbar((bx_user_review_year[1:] + bx_user_review_year[:-1])/2, bp_user_review_year, yerr=bperr_user_review_year, fmt=color + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)  
        

     




        # music
        file_cnt_year    = 'ProcessedDataNormalizedRandomized/3_inflation_curves/music_yearly_rating_counts_dist_pop.dat'
        rating_cnt_music = np.asarray([line.strip() for line in open(file_cnt_year)])
        x_num_of_movies_year,  y_num_of_movies_year = get_dict_data(rating_cnt_music)       
        bx_num_of_movies_year, bp_num_of_movies_year, bperr_num_of_movies_year = getBinnedDistribution(x_num_of_movies_year, y_num_of_movies_year, num_of_bins)

        ax[1,0].set_ylim([-400,10000])         
        ax[1,0].set_title('Lastfm playcounts', fontsize = 20)
        ax[1,0].errorbar((bx_num_of_movies_year[1:] + bx_num_of_movies_year[:-1])/2, bp_num_of_movies_year, yerr=bperr_num_of_movies_year, fmt='k' + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)


        file_cnt_year    = 'ProcessedDataNormalizedRandomized/3_inflation_curves/music_yearly_rating_counts_dist_electro.dat'
        rating_cnt_music = np.asarray([line.strip() for line in open(file_cnt_year)])
        x_num_of_movies_year,  y_num_of_movies_year = get_dict_data(rating_cnt_music)       
        bx_num_of_movies_year, bp_num_of_movies_year, bperr_num_of_movies_year = getBinnedDistribution(x_num_of_movies_year, y_num_of_movies_year, num_of_bins)

        ax[1,0].errorbar((bx_num_of_movies_year[1:] + bx_num_of_movies_year[:-1])/2, bp_num_of_movies_year, yerr=bperr_num_of_movies_year, fmt='b' + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)
    
        
    align_plot(ax)   
    #plt.savefig('inflation_data.png') 
    #plt.close()    
    plt.show()
    







    
    
 

''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''                       CAREER LENGTH AND SHIT                   '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   



   

def get_length_plots():




    title_font  = 25 
    num_of_bins = 20
    seaborn.set_style('white')  

    f, ax = plt.subplots(2, 3, figsize=(22, 15))
    #st = f.suptitle("IMDb Inflation of impact measures", fontsize=title_font)
    file_avg_year  = 'ProcessedData/3_inflation_curves/film_yearly_average_ratings_dist_director.dat'       
    average_ratings_year = np.asarray([line.strip() for line in open(file_avg_year)])
    x_average_ratings_year, y_average_ratings_year = get_num_per_year(average_ratings_year)  
    ax[0,0].plot(x_average_ratings_year, y_average_ratings_year, 'ko', label = 'movies', alpha  = 0.8)
    ax[0,0].set_title('numbre of movies', fontsize = title_font)

    ax[0,1].set_yscale('log')
    ax[0,1].plot(x_average_ratings_year, y_average_ratings_year, 'ko', label = 'movies', alpha  = 0.8)
    ax[0,1].set_title('numbre of movies', fontsize = title_font)

    

    file_avg_year_electro = 'ProcessedData/3_inflation_curves/music_yearly_rating_counts_dist_electro.dat' 
    file_avg_year_pop     = 'ProcessedData/3_inflation_curves/music_yearly_rating_counts_dist_pop.dat'   
    average_ratings_year_electro = np.asarray([line.strip() for line in open(file_avg_year_electro)])
    x_average_ratings_year_electro, y_average_ratings_year_electro = get_num_per_year(average_ratings_year_electro)  

    average_ratings_year_pop = np.asarray([line.strip() for line in open(file_avg_year_pop)])
    x_average_ratings_year_pop, y_average_ratings_year_pop = get_num_per_year(average_ratings_year_pop)  

    
    ax[1,0].plot(x_average_ratings_year_electro, y_average_ratings_year_electro, 'ko', label = 'electro', alpha = 0.8)
    ax[1,0].plot(x_average_ratings_year_pop,     y_average_ratings_year_pop,     'bo', label = 'pop',     alpha = 0.8)
    ax[1,0].set_title('numbre of tracks', fontsize = title_font)

    ax[1,1].set_yscale('log')
    ax[1,1].plot(x_average_ratings_year_electro, y_average_ratings_year_electro, 'ko', label = 'electro', alpha = 0.8)
    ax[1,1].plot(x_average_ratings_year_pop,     y_average_ratings_year_pop,     'bo', label = 'pop',     alpha = 0.8)
    ax[1,1].set_title('numbre of tracks', fontsize = title_font)




    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y'),]    
       
    for (label, color) in professions:
    
        career_length  = [float(line.strip()) for line in open('ProcessedData/5_career_length/film_career_length_' + label + '.dat')]
        xcareer_length, pcareer_length = getDistribution(career_length)
        
        ax[0,2].set_xscale('log')
        ax[0,2].set_yscale('log')
        ax[0,2].plot(xcareer_length, pcareer_length, color, marker = 'o', alpha = 0.3, linewidth = 0, label = label+ ', ')





    ax[0,2].set_title('length of movie careers')
    ax[1,2].set_title('length of music careers')    
    professions = [('pop',     'k'), 
                   ('electro', 'b')]    
       
    for (label, color) in professions:
    
        career_length  = [float(line.strip()) for line in open('ProcessedData/5_career_length/music_career_length_' + label + '.dat')]
        xcareer_length, pcareer_length = getDistribution(career_length)
        
        ax[1,2].set_xscale('log')
        ax[1,2].set_yscale('log')
        ax[1,2].plot(xcareer_length, pcareer_length, color, marker = 'o', alpha = 0.3, linewidth = 0, label = label)
    







    
    
    align_plot(ax)
    plt.savefig('career_length_data.png')
    plt.show()







''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''                        GET CORRELATION STUFF                   '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   


def get_impact_correlations():



    num_of_bins = 20
    title_font = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(25, 15))
    st = f.suptitle("IMDb impact correlations", fontsize=title_font)



    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]    
    
    
    for (label, color) in professions:
        
        impacts = zip(*[ line.strip().split('\t')   for line in open('ProcessedDataNormalized/7_multiple_impacts/film_multiple_impacts_' + label + '.dat')])
        

        #ax[0,0].set_title('IMDb - average rating', fontsize = 20)
        ax[0,0].set_ylabel('avg rating', fontsize = 20)
        ax[0,0].set_xlabel('rating cnt', fontsize = 20)
        ax[0,0].xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax[0,0].plot(impacts[2],  impacts[1], color + 'o', alpha = 0.3, label = label)
 
 
        ax[0,1].set_ylabel('avg rating', fontsize = 20)
        ax[0,1].set_xlabel('metascore', fontsize = 20)
        #ax[0,1].set_xlim([0.01,2])
        ax[0,1].plot(impacts[3],  impacts[1], color + 'o', alpha = 0.3, label = label)
 
 
        #ax[0,2].set_xscale('log')
        #ax[0,2].set_yscale('log')
        ax[0,2].set_ylabel('#critic review', fontsize = 20)
        ax[0,2].set_xlabel('#user review', fontsize = 20)
        ax[0,2].plot(impacts[4],  impacts[5], color + 'o', alpha = 0.3, label = label)
 
 
        ax[1,0].xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax[1,0].set_xlabel('rating cnt', fontsize = 20)
        ax[1,0].set_ylabel('metascore', fontsize = 20)
        #ax[1,0].set_ylim([0.01,2])
        ax[1,0].plot(impacts[2],  impacts[3], color + 'o', alpha = 0.3, label = label)


        #ax[1,1].xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax[1,1].set_xlabel('rating cnt', fontsize = 20)
        #ax[1,1].set_xlim([0,150])
        ax[1,1].set_ylabel('#critic review', fontsize = 20)
        #ax[1,1].set_ylim([0.01,40])
        ax[1,1].loglog(impacts[2],  impacts[4], color + 'o', alpha = 0.3, label = label)
        
        #ax[1,2].xaxis.get_major_formatter().set_powerlimits((0, 1))     
        ax[1,2].set_xlabel('rating cnt', fontsize = 20)
        #ax[1,2].set_xlim([0,150])
        ax[1,2].set_ylabel('#user review', fontsize = 20)
        #ax[1,2].set_ylim([0.01,40])
        ax[1,2].loglog(impacts[2],  impacts[5], color + 'o', alpha = 0.3, label = label)


  
    
    align_plot(ax)
    plt.savefig('correlations_normalized.png')
    plt.close()
    #plt.show()







''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''           GET THE N*/N STUFF OF ALL SUCCESS MEASURES           '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   


def plot_red_lines(ax, x):

    plt.grid()  
    for i in range(len(ax)):
        for j in range(len(ax[0])):     
            ax[i,j].set_xlim(-0.05, 1.05)
            ax[i,j].set_ylim(-0.05, 1.05)
            ax[i,j].grid()
            yyy = [1 - y0 for y0 in x]
            ax[i,j].plot(x, yyy, 'r--', linewidth=2) 
            ax[i,j].set_xlabel('$N^{*}/N$', fontsize=17)
            ax[i,j].set_ylabel( r'$P( \geq  N^{*}/N)$' , fontsize=17)



def plot_ccdf(file_avg_all, num_of_bins, ax, color, label, Nmin, title):


    x_Nstar_avg_all, p_Nstar_avg_all, len_career, r_square = parse_N_star_N_data(file_avg_all, Nmin)
    
    
    
    bx_average_ratings, bp_average_ratings, bperr_average_ratings = getBinnedDistribution(np.asarray(x_Nstar_avg_all), np.asarray(p_Nstar_avg_all), num_of_bins)
    ax.set_title(title, fontsize = 19)        
    ax.plot(x_Nstar_avg_all, p_Nstar_avg_all, color = color,  marker = 'o', linewidth = 0, markersize = 5, alpha= 0.5, label = label + ', ' + str(len_career) + ' $R^2=$' + str(round(r_square, 4)),)  
    ax.errorbar((bx_average_ratings[1:] + bx_average_ratings[:-1])/2, bp_average_ratings, yerr=bperr_average_ratings, fmt=color + '-', linewidth = 2)
            
            



def parse_N_star_N_data(filename, cutoff_N1, cutoff_N2 = 10000000000):

    try:
        N_star_N = []
        
        for line in open(filename):
        
            fields   = line.strip().split('\t')    
            best_id  = float(fields[0])
            career_N = float(fields[1])
            if career_N >= cutoff_N1 and career_N <= cutoff_N2:
                N_star_N.append(best_id/career_N)


        x_stat = np.linspace(0,1, len(N_star_N))
        maxy = max(N_star_N)
        y_stat = np.asarray([1-yy/maxy for yy in sorted(N_star_N)])      
    
        slope, intercept, r_square, p_value, std_err = stats.linregress(x_stat,[1 - aaa for aaa in y_stat])
  
        return x_stat, y_stat, len(N_star_N), r_square

    except ValueError:
    
        return [], [], 0, 0
        



def get_r_test():


    ''' ---------------------------------------------- '''
    '''      MOVIE YO                                  '''
    
    
    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]

    num_of_bins = 20
    title_font  = 25 
    Nmin = 20
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(25, 15))
    st = f.suptitle("IMDb impact distributions, $N_{min}$" + str(Nmin), fontsize=title_font)

    folder = 'ProcessedDataNormalized'


    for (label, color) in professions:
       
        file_avg_all  = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_avg_rating_'    + label + '.dat'
        file_cnt_all  = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_rating_count_' + label + '.dat'
        file_mets_all = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_metascores_'    + label + '.dat'
        file_crit_all = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_critic_review_'+ label + '.dat'
        file_user_all = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_user_review_'  + label + '.dat'
        plot_ccdf(file_avg_all,  num_of_bins, ax[0,0], color, label, Nmin, 'average ratings')
        plot_ccdf(file_cnt_all,  num_of_bins, ax[0,1], color, label, Nmin, 'rating counts')
        plot_ccdf(file_mets_all, num_of_bins, ax[0,2], color, label, Nmin, 'metascores')
        plot_ccdf(file_crit_all, num_of_bins, ax[1,0], color, label, Nmin, '#critic reviews')
        plot_ccdf(file_user_all, num_of_bins, ax[1,1], color, label, Nmin, '#user reviews')

        





    professions = [('pop',     'k'), 
                   ('electro', 'b')]    
       
    for (label, color) in professions:
    
        file_music = folder + '/4_NN_rank_N/music_best_product_NN_ranks_all_rating_count_' + label + '.dat'
        plot_ccdf(file_music,  num_of_bins, ax[1,2], color, label, Nmin, 'ratings counts')
        
       



    xxx = np.arange(0,1, 1.0/20)
    plot_red_lines(ax, xxx)
    align_plot(ax)
    plt.savefig('N_Nstar_'+str(Nmin)+'_normalized.png')
    plt.close()
    #plt.show()          
           





def get_R_square_map():

    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]

    num_of_bins = 20
    title_font  = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(25, 15))
    st = f.suptitle("$R^2$ values as the function of career cutoffs", fontsize=title_font)
    
    Nmax = 20
    FOLDER = 'ProcessedDataNormalized'

    for (label, color) in professions:
     
        i = professions.index((label, color))
       
              
        file_avg_all  = FOLDER + '/4_NN_rank_N/film_best_product_NN_ranks_all_avg_rating_'    + label + '.dat'
        file_cnt_all  = FOLDER + '/4_NN_rank_N/film_best_product_NN_ranks_all_rating_count_'  + label + '.dat'
        file_mets_all = FOLDER + '/4_NN_rank_N/film_best_product_NN_ranks_all_metascores_'    + label + '.dat'
        file_crit_all = FOLDER + '/4_NN_rank_N/film_best_product_NN_ranks_all_critic_review_' + label + '.dat'
        file_user_all = FOLDER + '/4_NN_rank_N/film_best_product_NN_ranks_all_user_review_'   + label + '.dat'
       
        r_square_avg_all  = [parse_N_star_N_data(file_avg_all,  Nmin)[3] for Nmin in range(Nmax)]
        r_square_cnt_all  = [parse_N_star_N_data(file_cnt_all,  Nmin)[3] for Nmin in range(Nmax)]
        r_square_mets_all = [parse_N_star_N_data(file_mets_all, Nmin)[3] for Nmin in range(Nmax)]
        r_square_crit_all = [parse_N_star_N_data(file_crit_all, Nmin)[3] for Nmin in range(Nmax)]
        r_square_user_all = [parse_N_star_N_data(file_user_all, Nmin)[3] for Nmin in range(Nmax)]       

        ax[0,0].plot(r_square_avg_all,  color + 'o-', alpha = 0.8, label = label)
        ax[0,1].plot(r_square_cnt_all,  color + 'o-', alpha = 0.8, label = label)
        ax[0,2].plot(r_square_mets_all, color + 'o-', alpha = 0.8, label = label)                
        ax[1,0].plot(r_square_crit_all, color + 'o-', alpha = 0.8, label = label)
        ax[1,1].plot(r_square_user_all, color + 'o-', alpha = 0.8, label = label) 



    professions = [('pop',     'k'), 
                   ('electro', 'b')]    
       
    for (label, color) in professions:
        file_music = FOLDER + '/4_NN_rank_N/music_best_product_NN_ranks_all_rating_count_' + label + '.dat'
        r_square_cnt_all = [parse_N_star_N_data(file_music,  Nmin)[3] for Nmin in range(Nmax)]
        ax[1,2].plot(r_square_cnt_all,  color + 'o-', alpha = 0.8, label = label)
    
    ax[0,0].set_ylabel('$R^2$', fontsize = 19)                       
    ax[1,0].set_ylabel('$R^2$', fontsize = 19)
    ax[1,0].set_xlabel('career length', fontsize = 19)
    ax[1,1].set_xlabel('career length', fontsize = 19)
    ax[1,2].set_xlabel('career length', fontsize = 19)
    
    align_plot(ax)
    
    plt.savefig('N_Nstar_scan_normalized.png')
    plt.close()
    #plt.show()







''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''                    GET  THE EXPONENT VALUES                    '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   



def parse_exp_values(filename):

    alpha     = []
    alpha_err = []
    beta      = []
    beta_err  = []
    
    for line in open(filename):
    
        fields = [float(num) for num in line.strip().split('\t')]    
        alpha.append(round(fields[0],1))
        alpha_err.append(fields[1])
        beta.append(round(fields[2],1))
        beta_err.append(fields[3])

    return alpha, alpha_err, beta, beta_err
    



def plot_exponent(series, ax, num_of_bins, color, label):

    x_series,  p_series  = getDistribution(series, True)
    bx_series, bp_series, bp_series_err = getBinnedDistribution(x_series,  p_series, num_of_bins)
    ax.plot(x_series, p_series, color + 'o', alpha = 0.1, markersize = 7)
    ax.errorbar((bx_series[1:] + bx_series[:-1])/2, bp_series, yerr=bp_series_err, color = color, fmt='o-', alpha = 0.9, capsize = 3, elinewidth=1, linewidth = 3, label = label)
    
    



def add_measure(data, data_rand, color, num_of_bins, ax1, ax2, label):

    ax1.set_title(label, fontsize = 19)
    
    alpha, alpha_err, beta, beta_err = parse_exp_values(data)
    alpha_rand, alpha_err_rand, beta_rand, beta_err_rand = parse_exp_values(data_rand)

    plot_exponent(alpha_rand, ax1, num_of_bins, 'r', 'randomized')
    plot_exponent(alpha,      ax1, num_of_bins, color, 'data')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plot_exponent(beta_rand,  ax2, num_of_bins, 'r', 'randomized')        
    plot_exponent(beta,       ax2, num_of_bins, color, 'data')





def get_exponents():

    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]

    num_of_bins = 20
    title_font  = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 6, figsize=(25, 10))
    st = f.suptitle("Exponents - normalized - director", fontsize=title_font)
    
    FOLDER   = 'ProcessedDataNormalized'
    FOLDER_R = 'ProcessedDataNormalizedRandomized'
    folder   = '/8_exponents/'

    for (label, color) in professions[0:1]:
        
        print label
        file_avg_rating = FOLDER + folder + 'film_exponents_avg_rating_'  + label + '.dat'
        file_rating_cnt = FOLDER + folder + 'film_exponents_rating_cnt_'  + label + '.dat'
        file_metascore  = FOLDER + folder + 'film_exponents_metascore_'   + label + '.dat'
        file_critic_rev = FOLDER + folder + 'film_exponents_critic_rev_'  + label + '.dat'
        file_user_rev   = FOLDER + folder + 'film_exponents_user_rev_'    + label + '.dat'
        
        
        file_avg_rating_rand = FOLDER_R + folder + 'film_exponents_avg_rating_'  + label + '.dat'
        file_rating_cnt_rand = FOLDER_R + folder + 'film_exponents_rating_cnt_'  + label + '.dat'
        file_metascore_rand  = FOLDER_R + folder + 'film_exponents_metascore_'   + label + '.dat'
        file_critic_rev_rand = FOLDER_R + folder + 'film_exponents_critic_rev_'  + label + '.dat'
        file_user_rev_rand   = FOLDER_R + folder + 'film_exponents_user_rev_'    + label + '.dat'

        add_measure(file_avg_rating, file_avg_rating_rand, color, num_of_bins, ax[0,0], ax[1,0], 'imdb - avg rating')
        add_measure(file_rating_cnt, file_rating_cnt_rand, color, num_of_bins, ax[0,1], ax[1,1], 'imdb - rating cnt')
        add_measure(file_metascore,  file_metascore_rand,  color, num_of_bins, ax[0,2], ax[1,2], 'imdb - meta')
        add_measure(file_critic_rev, file_critic_rev_rand, color, num_of_bins, ax[0,3], ax[1,3], 'imdb - critic rev')
        add_measure(file_user_rev,   file_user_rev_rand,   color, num_of_bins, ax[0,4], ax[1,4], 'imdb - user rev')



    professions = [('pop',     'k'), 
                   ('electro', 'b')]    
       
    for (label, color) in professions[0:1]:
        file_music       = FOLDER   + folder + 'music_exponents_rating_cnt_'  + label + '.dat'
        file_music_raand = FOLDER_R + folder + 'music_exponents_rating_cnt_'  + label + '.dat'
        add_measure(file_music, file_music_raand, color, num_of_bins, ax[0,5], ax[1,5], 'music - electro')


    ax[0,0].set_ylabel('$\\beta$' , fontsize = 19)
    ax[1,0].set_ylabel('$\\alpha$', fontsize = 19)

    align_plot(ax)
    plt.savefig('exponents.png')
    plt.close()
    #plt.show()
    
    
if __name__ == '__main__':         


    if sys.argv[1] == '1':
        get_imapct_distr()
        
    elif sys.argv[1] == '2':
        get_inflation_curves()

    elif sys.argv[1] == '3':
        get_length_plots()
    
    elif sys.argv[1] == '4':
        get_r_test()
        
    elif sys.argv[1] == '5':
        get_R_square_map()
        
    elif sys.argv[1] == '6':
        get_impact_correlations()

    elif sys.argv[1] == '7':
        get_exponents()
            
