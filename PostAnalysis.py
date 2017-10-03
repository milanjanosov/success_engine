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
- check normalized inflation curve
- plot normalized impact distibution
- get correlation plots   -> normalized vs randomized
- get career exponent fit -> normalized vs randomized
- N*/N stuff....






SuccessProcess

+++ 1. save yearly avg values  -> field_measure_infl_avg
+++ 2. create ProcessedDataNormalized by doing the same anal but dividing the yearly avg (two main function calls)
+++ 3. save career length distributions
+++ 4. read multiple impact measures to see correlations
+++ 5. add randomized mode!

--- add exp. calculator
--- add autocorrel calculator
--- 6. check goodreads scripts
--- read 2 papers

0. 



2. impact distribution
  - add powerlaw fits
  - fit the distributions?!
  - try to fit the skewed gaussians (S4.1 lognormal)



1. r-rule ccdf
  - add _rand versions too
  - adding cutoff, like career length
  - plot career length distributions
  - career length vs R^2 of the ccdf
    N1 < N < N2    -> R^2(N1, N2)
  - career length distributions




3. inflation curves
 Success process csinalja meg a yearl avg fileokat es normalizaljon is mindent...
  - get the plots
  - get the renorm function
  - renorm all previous impacts
  - redraw impact distr plots 
  - redraw correlation plots



4. correlation plots
  - careertrajectory with all impact values
  

5. autocorrelation


'''



''' plot helper functions '''

def align_plot(ax):

    font_tick = 14   

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
    st = f.suptitle("IMDb impact distributions", fontsize=title_font)


    for (label, color) in professions:
       
        num_car  = str(int(round(len(os.listdir('Data/Film/film-'+ label +'-simple-careers'))/1000.0))) + 'k'

        
        file_avg  = 'ProcessedDataSample/1_impact_distributions/film_average_ratings_dist_' + label + '.dat'
        file_cnt  = 'ProcessedDataSample/1_impact_distributions/film_rating_counts_dist_'   + label + '.dat'
        file_mets = 'ProcessedDataSample/1_impact_distributions/film_metascores_dist_'      + label + '.dat'
        file_crit = 'ProcessedDataSample/1_impact_distributions/film_critic_review_dist_'   + label + '.dat'
        file_user = 'ProcessedDataSample/1_impact_distributions/film_user_review_dist_'     + label + '.dat'

        average_ratings = np.asarray([round(float(line.strip()),3) for line in open(file_avg)])
        rating_counts   = [round(float(line.strip()),3) for line in open(file_cnt)]
        metascores      = [round(float(line.strip()),3) for line in open(file_mets)]
        critic_review   = [round(float(line.strip()),3) for line in open(file_crit)]
        user_review     = [round(float(line.strip()),3) for line in open(file_user)]
        
      
      
        
        # plot avg ratings
        x_average_ratings,  p_average_ratings = getDistribution(average_ratings, True)
        bx_average_ratings, bp_average_ratings, bperr_average_ratings = getBinnedDistribution(x_average_ratings, p_average_ratings, num_of_bins)
    
        
        ax[0,0].set_title('IMDb - average rating', fontsize = 20)
        ax[0,0].plot(x_average_ratings,  p_average_ratings, color, marker = 'o', alpha = 0.3, linewidth = 0, label = label+ ', ' + str(num_car))
        ax[0,0].errorbar((bx_average_ratings[1:] + bx_average_ratings[:-1])/2, bp_average_ratings, yerr=bperr_average_ratings, fmt=color + '-', linewidth = 2)
        
        
        # plot rating counts  
        x_rating_counts,  p_rating_counts = getDistribution(rating_counts, True)
        bx_rating_counts, bp_rating_counts, bperr_rating_counts = getBinnedDistribution(x_rating_counts, p_rating_counts, num_of_bins)

        ax[0,1].set_title('IMDb - rating count', fontsize = 20)
        ax[0,1].set_xscale('log')
        ax[0,1].set_yscale('log')
        ax[0,1].plot(x_rating_counts,  p_rating_counts, color + 'o', alpha = 0.2, label = label)# + ', ' + str(num_wr))
        #ax[0,1].errorbar((bx_rating_counts[1:] + bx_rating_counts[:-1])/2, bp_rating_counts, yerr=bperr_rating_counts, fmt='b-', linewidth = 2)

        
        # plot metascores
        x_metascores,  p_metascores = getDistribution(metascores, True)
        ax[0,2].set_title('IMDb - metascores', fontsize = 20)
        ax[0,2].plot(x_metascores,  p_metascores, color + 'o', alpha = 0.2, label = label)# + ', ' + str(len(metascores)))

        
        # plot critic review count
        x_critic_review,  p_critic_review = getDistribution(critic_review, True)
        ax[1,0].set_title('IMDb - critic_review', fontsize = 20)
        ax[1,0].set_xscale('log')
        ax[1,0].set_yscale('log')
        ax[1,0].plot(x_critic_review, p_critic_review, color + 'o', alpha = 0.2, label = label )#+ ', ' + str(len(critic_review)))


        
        # plot user review count
        x_user_review,  p_user_review = getDistribution(user_review, True)
        ax[1,1].set_title('IMDb - user_review', fontsize = 20)
        ax[1,1].set_xscale('log')
        ax[1,1].set_yscale('log')
        ax[1,1].plot(x_user_review, p_user_review, color + 'o', alpha = 0.2, label = label)# + ', ' + str(len(user_review)))
    
    


    ''' ---------------------------------------------- '''
    '''      MOVIE YO                                  '''
    
    genres = [('electro', 'k'), ('pop', 'b')]
    
    
    for (genre, color) in genres:

        num_mus  = str(int(round(len(os.listdir('Data/Music/music-'+ genre +'-simple-careers'))/1000.0))) + 'k'
      
        file_music = 'ProcessedDataSample/1_impact_distributions/music_rating_counts_dist_' + genre + '.dat'
        average_ratings = np.asarray([float(line.strip()) for line in open(file_music)])    
        x_rating_counts,  p_rating_counts = getDistribution(average_ratings, True)    

        print len(average_ratings)

        ax[1,2].set_title('Music - playcount', fontsize = 20)
        ax[1,2].set_xscale('log')
        ax[1,2].set_yscale('log')
        ax[1,2].plot(x_rating_counts,  p_rating_counts, color + 'o', alpha = 0.2, label = genre + ', ' + num_mus)


    align_plot(ax)
    plt.savefig('impact_distributions_.png')
    #plt.close()
    plt.show()          



    



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

        
        file_avg_year  = 'ProcessedData/3_inflation_curves/film_yearly_average_ratings_dist_' + label + '.dat'
        file_cnt_year  = 'ProcessedData/3_inflation_curves/film_yearly_rating_counts_dist_'   + label + '.dat'
        file_mets_year = 'ProcessedData/3_inflation_curves/film_yearly_metascores_dist_'      + label + '.dat'
        file_crit_year = 'ProcessedData/3_inflation_curves/film_yearly_critic_review_dist_'   + label + '.dat'
        file_user_year = 'ProcessedData/3_inflation_curves/film_yearly_user_review_dist_'     + label + '.dat'

        average_ratings_year = np.asarray([line.strip() for line in open(file_avg_year)])
        rating_counts_year   = np.asarray([line.strip() for line in open(file_cnt_year)])
        metascores_year      = np.asarray([line.strip() for line in open(file_mets_year)])
        critic_review_year   = np.asarray([line.strip() for line in open(file_crit_year)])
        user_review_year     = np.asarray([line.strip() for line in open(file_user_year)])
        

        
        # plot average ratings
        x_average_ratings_year, y_average_ratings_year = get_dict_data(average_ratings_year)       
        bx_average_ratings_year, bp_average_ratings_year, bperr_average_ratings_year = getBinnedDistribution(x_average_ratings_year, y_average_ratings_year, num_of_bins)

        
    
        ax[0,0].set_title('IMDb - average rating', fontsize = 20)
        #ax[0,0].plot(x_average_ratings_year,  y_average_ratings_year, color + 'o', alpha = 0.001, label = label + ', ' + str(len(average_ratings_year)))
        ax[0,0].errorbar((bx_average_ratings_year[1:] + bx_average_ratings_year[:-1])/2, bp_average_ratings_year, yerr=bperr_average_ratings_year, fmt=color + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)
        
        
        #plot rating counts
        x_rating_counts_year, y_rating_counts_year = get_dict_data(rating_counts_year)          
        bx_rating_counts_year, bp_rating_counts_year, bperr_rating_counts_year = getBinnedDistribution(x_rating_counts_year, y_rating_counts_year, num_of_bins)

        ax[0,1].set_title('IMDb - rating count', fontsize = 20)
        ax[0,1].set_ylim([-3000,20000])
        #ax[0,1].plot(x_rating_counts_year,  y_rating_counts_year, color + 'o', alpha = 0.001, label = label + ', ' + str(len(rating_counts_year)))
        ax[0,1].errorbar((bx_rating_counts_year[1:] + bx_rating_counts_year[:-1])/2, bp_rating_counts_year, yerr=bperr_rating_counts_year, fmt=color + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)


        
        # plot metascorenumber of movies
        x_metascores_year,  y_metascores_year = get_dict_data(metascores_year)       
        bx_metascores_year, bp_metascores_year, bperr_metascores_year = getBinnedDistribution(x_metascores_year, y_metascores_year, num_of_bins)

        ax[1,1].set_title('IMDb - metascores_year', fontsize = 20)
        #ax[1,1].plot(x_metascores_year,  y_metascores_year, color + 'o', alpha = 0.2, label = label + ', ' + str(len(metascores_year)))
        ax[1,1].errorbar((bx_metascores_year[1:] + bx_metascores_year[:-1])/2, bp_metascores_year, yerr=bperr_metascores_year, fmt=color + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)

        
        # plot critic reviews
        ax[0,2].set_ylim([-25,150])         
        x_critic_review_year,  y_critic_review_year = get_dict_data(critic_review_year)       
        bx_critic_review_year, bp_critic_review_year, bperr_critic_review_year = getBinnedDistribution(x_critic_review_year, y_critic_review_year, num_of_bins)

        
        ax[0,2].set_title('IMDb - critic_review_year', fontsize = 20)
        #ax[0,2].plot(x_critic_review_year,  y_critic_review_year, color + 'o', alpha = 0.2, label = label + ', ' + str(len(critic_review_year)))
        ax[0,2].errorbar((bx_critic_review_year[1:] + bx_critic_review_year[:-1])/2, bp_critic_review_year, yerr=bperr_critic_review_year, fmt=color + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)  



        # plot user reviews
        x_user_review_year,  y_user_review_year = get_dict_data(user_review_year)       
        bx_user_review_year, bp_user_review_year, bperr_user_review_year = getBinnedDistribution(x_user_review_year, y_user_review_year, num_of_bins)

        ax[1,2].set_ylim([-15,75])         
        ax[1,2].set_title('IMDb - user_review_year', fontsize = 20)
        #ax[1,2].plot(x_user_review_year,  y_user_review_year, color + 'o', alpha = 0.2, label = label + ', ' + str(len(user_review_year)))
        ax[1,2].errorbar((bx_user_review_year[1:] + bx_user_review_year[:-1])/2, bp_user_review_year, yerr=bperr_user_review_year, fmt=color + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)  
        

     




        # music
        file_cnt_year    = 'ProcessedData/3_inflation_curves/music_yearly_rating_counts_dist_pop.dat'
        rating_cnt_music = np.asarray([line.strip() for line in open(file_cnt_year)])
        x_num_of_movies_year,  y_num_of_movies_year = get_dict_data(rating_cnt_music)       
        bx_num_of_movies_year, bp_num_of_movies_year, bperr_num_of_movies_year = getBinnedDistribution(x_num_of_movies_year, y_num_of_movies_year, num_of_bins)

        ax[1,0].set_ylim([-400,10000])         
        ax[1,0].set_title('Lastfm playcounts', fontsize = 20)
        #ax[1,0].plot(x_num_of_movies_year,  y_num_of_movies_year, color + 'o', alpha = 0.001, label = label + ', ' + str(len(average_ratings_year)))
        ax[1,0].errorbar((bx_num_of_movies_year[1:] + bx_num_of_movies_year[:-1])/2, bp_num_of_movies_year, yerr=bperr_num_of_movies_year, fmt='k' + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)


        file_cnt_year    = 'ProcessedData/3_inflation_curves/music_yearly_rating_counts_dist_electro.dat'
        rating_cnt_music = np.asarray([line.strip() for line in open(file_cnt_year)])
        x_num_of_movies_year,  y_num_of_movies_year = get_dict_data(rating_cnt_music)       
        bx_num_of_movies_year, bp_num_of_movies_year, bperr_num_of_movies_year = getBinnedDistribution(x_num_of_movies_year, y_num_of_movies_year, num_of_bins)

        #ax[1,0].plot(x_num_of_movies_year,  y_num_of_movies_year, color + 'o', alpha = 0.001, label = label + ', ' + str(len(average_ratings_year)))
        ax[1,0].errorbar((bx_num_of_movies_year[1:] + bx_num_of_movies_year[:-1])/2, bp_num_of_movies_year, yerr=bperr_num_of_movies_year, fmt='b' + 'o-', alpha = 0.8, capsize = 3, elinewidth=1, linewidth = 3)
   
        
    align_plot(ax)   
    plt.savefig('inflation_data.png') 
    plt.close()    
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
        ax[1,2].plot(xcareer_length, pcareer_length, color, marker = 'o', alpha = 0.3, linewidth = 0, label = label+ ', ')
    







    
    
    align_plot(ax)
    plt.savefig('career_length_data.png')
    plt.show()







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
            



def parse_N_star_N_data(filename, cutoff_N1, cutoff_N2):

    try:
        N_star_N = []
        
        for line in open(filename):
        
            fields   = line.strip().split('\t')    
            best_id  = float(fields[0])
            career_N = float(fields[1])
            
            if career_N >= cutoff_N1 and career_N <= cutoff_N2:
                N_star_N.append(best_id/career_N)


        NNN = sorted(N_star_N)

        CDF = np.cumsum(NNN)

        x = NNN
        y = CDF
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float) 
        maxy = max(y)
        y = [1-yy/maxy for yy in y]      


        #x_N_star_N,  p_N_star_N = getDistribution(N_star_N, True)         
        ###
        ###    E_i    : expected
        ###    O_i    : observed
        ###    \khi^2 : goodness = \sum_i (O_i - E_i)^2/E_i
        
        yNN = [1 - xy for xy in y]
        num = len(yNN)

        #r_square = sum([(yNN[i] - x[
        slope, intercept, r_square, p_value, std_err = stats.linregress(x,yNN)
        #r_square = random.random()
        
        return x, y, len(N_star_N), r_square

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
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(25, 15))
    st = f.suptitle("IMDb impact distributions", fontsize=title_font)


    for (label, color) in professions[0:1]:
       
        #num_car  = str(int(round(len(os.listdir('Data/Film/film-'+ label +'-simple-careers'))/1000.0))) + 'k'
      
        
        file_avg_all  = 'ProcessedDataNormalized/4_NN_rank_N/film_best_product_NN_ranks_all_avg_rating_'    + label + '.dat'
        #file_cnt_all  = 'ProcessedDataNormalizedRandomized/4_NN_rank_N/film_best_product_NN_ranks_rand_rating_count_' + label + '.dat'
        #file_mets_all = 'ProcessedDataNormalizedRandomized/4_NN_rank_N/film_best_product_NN_ranks_all_metascores_'    + label + '.dat'
        #file_crit_all = 'ProcessedDataNormalizedRandomized/4_NN_rank_N/film_best_product_NN_ranks_rand_critic_review_'+ label + '.dat'
        #file_user_all = 'ProcessedDataNormalizedRandomized/4_NN_rank_N/film_best_product_NN_ranks_rand_user_review_'  + label + '.dat'



        print file_avg_all
        plot_ccdf(file_avg_all, num_of_bins, ax[0,0], color, label)
        #plot_ccdf(file_cnt_all, num_of_bins, ax[0,1], color, label)   
        #rating_counts   = [float(line.strip().split('\t')) for line in open(file_cnt)]
        #metascores      = [float(line.strip()) for line in open(file_mets)]
        #critic_review   = [float(line.strip()) for line in open(file_crit)]
        #user_review     = [float(line.strip()) for line in open(file_user)]
       


 




    xxx = np.arange(0,1, 1.0/20)
    plot_red_lines(ax, xxx)
    align_plot(ax)
    #plt.savefig('impact_distributions.png')
    #plt.close()
    plt.show()          
           

def plot_ccdf(file_avg_all, num_of_bins, ax, color, label):

    x_Nstar_avg_all, p_Nstar_avg_all, len_career, r_square = parse_N_star_N_data(file_avg_all,  10,1000)
    
    bx_average_ratings, bp_average_ratings, bperr_average_ratings = getBinnedDistribution(np.asarray(x_Nstar_avg_all), np.asarray(p_Nstar_avg_all), num_of_bins)
    ax.set_title('IMDb - average ratings', fontsize = 19)        
    ax.plot(x_Nstar_avg_all, p_Nstar_avg_all, color = color,  marker = 'o', linewidth = 0, markersize = 10, alpha= 0.5, label = label + ', ' + str(len_career) + ' $R^2=$' + str(round(r_square, 4)),)  
    ax.errorbar((bx_average_ratings[1:] + bx_average_ratings[:-1])/2, bp_average_ratings, yerr=bperr_average_ratings, fmt=color + '-', linewidth = 2)
            




def get_R_square_map():

    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]

    num_of_bins = 20
    title_font  = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(3, 5, figsize=(25, 15))
    st = f.suptitle("$r^2$ values as the function of career cutoffs", fontsize=title_font)

    for (label, color) in professions:
     
        i = professions.index((label, color))
       
              
        file_avg_all  = 'ProcessedDataNormalizedRandomized/4_NN_rank_N/film_best_product_NN_ranks_all_avg_rating_'  + label + '.dat'
        file_cnt_all  = 'ProcessedDataNormalizedRandomized/4_NN_rank_N/film_best_product_NN_ranks_all_avg_rating_'  + label + '.dat'
        file_mets_all = 'ProcessedDataNormalizedRandomized/4_NN_rank_N/film_best_product_NN_ranks_all_avg_rating_'  + label + '.dat'
        file_crit_all = 'ProcessedDataNormalizedRandomized/4_NN_rank_N/film_best_product_NN_ranks_all_avg_rating_'  + label + '.dat'
        file_user_all = 'ProcessedDataNormalizedRandomized/4_NN_rank_N/film_best_product_NN_ranks_all_avg_rating_'  + label + '.dat'
       
        N_max = 10 # = max([line.strip() for line in open(...)])
        
        R_squar_mtx_avg_all = np.zeros((N_max,N_max))
        
       
        for N1 in range(N_max):
            for N2 in range(N_max):               
                x_Nstar_avg_all, p_Nstar_avg_all, len_career, r_square = parse_N_star_N_data(file_avg_all, N1, N2)            
                R_squar_mtx_avg_all[N1][N2] = r_square
        
       
    
        ax[0,0 + i].matshow(R_squar_mtx_avg_all)
        ax[0,0 + i].set_xlabel=('N1')
        ax[0,0 + i].set_ylabel('N2')
   
   
    plt.show()

    
    
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


            
