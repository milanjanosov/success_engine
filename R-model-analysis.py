import sys
import os
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import random
import powerlaw
from scipy import stats
from CareerTrajectory.careerTrajectory import getDistribution
from CareerTrajectory.careerTrajectory import getBinnedDistribution
from CareerTrajectory.careerTrajectory import getLogBinnedDistribution



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
            #ax[i,j].grid()
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
            
            
def align_plot1D(ax):

    font_tick = 15   

    for i in range(len(ax)):
        #ax[i,j].grid()
        ax[i].legend(loc = 'left', fontsize = font_tick) 
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].get_xaxis().tick_bottom()
        ax[i].get_yaxis().tick_left()
        ticklines  = ax[i].get_xticklines()  + ax[i].get_yticklines()
        gridlines  = ax[i].get_xgridlines()  + ax[i].get_ygridlines()
        ticklabels = ax[i].get_xticklabels() + ax[i].get_yticklabels()
        for line in ticklines:
            line.set_linewidth(1)

        for line in gridlines:
            line.set_linestyle('-.')

        ax[i].tick_params(labelsize = font_tick) 
            
            
 

 


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



def get_yearly_avg_data(impacts):

    
    years_impacts = {}
    for i in impacts:
        field  = i.split('\t')      
        year   = round(float(field[0]))
        impact = float(field[1])
        if year not in years_impacts:
            years_impacts[year] = [impact]
        else:
            years_impacts[year].append(impact)
            
    x = []
    y = []
    z = []
    maxx = 100
    maxi = 0
    N =0
    for year, impacts in years_impacts.items():
        x.append(year)
        y.append(np.mean(impacts))
        z.append(np.std(impacts))
    
        if np.mean(impacts) > maxx:
            maxx = np.mean(impacts)
            maxi = year
            N=len(impacts)
    

    return np.asarray(x), np.asarray(y), np.asarray(z)


def get_num_per_year(impacts):

    
    years = {} 
    for i in impacts:
        field = i.split('\t')
        year  = round(float((field[0])))
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



def plot_measure(average_ratings_year, title, num_of_bins, ax, color, label, music = False):
    
  
    x_average_ratings_year, y_average_ratings_year, yerr_average_ratings_year = get_yearly_avg_data(average_ratings_year)    
    bx_average_ratings_year, bp_average_ratings_year, bperr_average_ratings_year = getBinnedDistribution(x_average_ratings_year, y_average_ratings_year, num_of_bins)

    ax.set_title(title, fontsize = 20)
    
    if music:
        ax.errorbar(x_average_ratings_year, y_average_ratings_year, yerr=yerr_average_ratings_year, fmt=color + '-', alpha = 0.5, capsize = 3, elinewidth=1, linewidth = 2)
        ax.errorbar((bx_average_ratings_year[1:] + bx_average_ratings_year[:-1])/2, bp_average_ratings_year, yerr=bperr_average_ratings_year, fmt=color + 'o-', alpha = 0.6, capsize = 3, elinewidth=1, linewidth = 3, label = label)
    else:
        ax.errorbar((bx_average_ratings_year[1:] + bx_average_ratings_year[:-1])/2, bp_average_ratings_year, yerr=bperr_average_ratings_year, fmt=color + 'o-', alpha = 0.6, capsize = 3, elinewidth=1, linewidth = 3, label = label)
    
    ax.set_xlim([1880, 2020])



def get_inflation_curves():


    num_of_bins = 8
    title_font = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(25, 15))
    st = f.suptitle("Inflation of impact measures", fontsize=title_font)


    FOLDER = 'ProcessedData'

    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]    
    
    
    for (label, color) in professions:

        
        file_avg_year  = FOLDER + '/3_inflation_curves/film_yearly_average_ratings_dist_' + label + '.dat'
        file_cnt_year  = FOLDER + '/3_inflation_curves/film_yearly_rating_counts_dist_'   + label + '.dat'
        file_mets_year = FOLDER + '/3_inflation_curves/film_yearly_metascores_dist_'      + label + '.dat'
        file_crit_year = FOLDER + '/3_inflation_curves/film_yearly_critic_review_dist_'   + label + '.dat'
        file_user_year = FOLDER + '/3_inflation_curves/film_yearly_user_review_dist_'     + label + '.dat'

        average_ratings_year = np.asarray([line.strip() for line in open(file_avg_year)])
        rating_counts_year   = np.asarray([line.strip() for line in open(file_cnt_year)])
        metascores_year      = np.asarray([line.strip() for line in open(file_mets_year)])
        critic_review_year   = np.asarray([line.strip() for line in open(file_crit_year)])
        user_review_year     = np.asarray([line.strip() for line in open(file_user_year)])
        




        
        # plot average ratings
        plot_measure(average_ratings_year, 'IMDb - average rating', num_of_bins, ax[0,0], color, label)
        plot_measure(rating_counts_year,   'IMDb - rating counts',  num_of_bins, ax[0,1], color, label)
        plot_measure(metascores_year,      'IMDb - metascore',      num_of_bins, ax[0,2], color, label)  
        plot_measure(critic_review_year,   'IMDb - critic reviews', num_of_bins, ax[1,0], color, label)
        plot_measure(user_review_year,     'IMDb - user reviews',   num_of_bins, ax[1,1], color, label)        
       

   
    
    professions = [('pop',     'k'), 
                   ('electro', 'b')]    
    
    for (label, color) in professions:

        file_cnt_year    = FOLDER + '/3_inflation_curves/music_yearly_rating_counts_dist_' + label +'.dat'
        rating_cnt_music = np.asarray([line.strip() for line in open(file_cnt_year)])
        
       
        
        plot_measure(rating_cnt_music, 'Music - playcount', num_of_bins, ax[1,2], color, label, music = True)          
        
    
    ax[1,2].set_ylim([-1000,40000])
        
    align_plot(ax)   
    plt.savefig('inflation_data.png') 
    plt.close()    
    #plt.show()
    







    
    
 

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
    #ax[0,0].plot(x_average_ratings_year, y_average_ratings_year, 'ko', label = 'movies', alpha  = 0.8)
    #ax[0,0].set_title('#movies', fontsize = title_font)

    ax[0,0].set_yscale('log')
    ax[0,0].plot(x_average_ratings_year, y_average_ratings_year, 'ko', label = 'movies', alpha  = 0.6)
    ax[0,0].set_title('#movies', fontsize = title_font)

    

    file_avg_year_electro = 'ProcessedData/3_inflation_curves/music_yearly_rating_counts_dist_electro.dat' 
    file_avg_year_pop     = 'ProcessedData/3_inflation_curves/music_yearly_rating_counts_dist_pop.dat'   
    
    average_ratings_year_electro = np.asarray([line.strip() for line in open(file_avg_year_electro)])
    average_ratings_year_pop = np.asarray([line.strip() for line in open(file_avg_year_pop)])    
    
    x_average_ratings_year_electro, y_average_ratings_year_electro = get_num_per_year(average_ratings_year_electro)  
    x_average_ratings_year_pop, y_average_ratings_year_pop = get_num_per_year(average_ratings_year_pop)  

  
    xb_average_ratings_year_electro, pb_average_ratings_year_electro, pberr_average_ratings_year_electro= getBinnedDistribution(x_average_ratings_year_electro, y_average_ratings_year_electro, num_of_bins)
     
     
  
    ax[0,1].set_yscale('log') 
    ax[0,1].plot(x_average_ratings_year_electro, y_average_ratings_year_electro, 'ko', label = 'electro', alpha = 0.6)   
    ax[0,1].plot(x_average_ratings_year_pop,     y_average_ratings_year_pop,     'bo', label = 'pop',     alpha = 0.6)
    ax[0,1].set_title('#tracks', fontsize = title_font)


    ax[0,2].set_title('#books', fontsize = title_font)



    
    
    align_plot(ax)
    plt.savefig('num_of_products_length_data.png')
    #plt.show()








def fitPowerLaw(rand, ax, label):


   
    ax.set_title(label, fontsize = 18)

    
    
    # histogram 
    print 'Fitting lognormal...'
    x_rand, p_rand = getDistribution(rand)    
    counts, bins, bars = ax.hist(rand, normed = True, bins = 10 ** np.linspace(np.log10(min(x_rand)), np.log10(max(x_rand)), 15), log=True,alpha=0.0)#,   histtype='step', linewidth = 0)
    ax.plot((bins[1:] + bins[:-1])/2, counts, 's-', color = 'royalblue', alpha = 0.5, markersize = 12, linewidth = 2)


    # get the lognormal
    param = stats.lognorm.fit(rand)
    pdf_fitted = stats.lognorm.pdf(x_rand, param[0], loc=param[1], scale=param[2])#
    mu =  np.log(param[2])
    sigma = param[0]
    sk_results_norm = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.lognorm.cdf(x_rand, param[0], loc=param[1], scale=param[2]))   # stats.ks_2samp(np.cumsum(p_rand), np.cumsu
    ax.plot(x_rand,pdf_fitted,'k-', linewidth = 4, label = '$\\mu$=' + str(round(mu,2)) + ' $\\sigma$=' + str(round(sigma, 2)) + ', $D$='+str(round(sk_results_norm[0], 2)))


    
    # fit and plot the powerlaw   
    results = powerlaw.Fit(rand, xmin = min(x_rand), fit_method = 'KS')
    alpha  = results.power_law.alpha
    xmin   = results.power_law.xmin 
    D = results.power_law.KS()
 
    results.power_law.plot_pdf( color='r', ax = ax,  linestyle='-', linewidth=4, label ='$\\alpha$= ' + str(round(alpha,2)) + ', $x_{min}$=' + str(round(xmin,2)) + '\n$D$='+str(round(D, 2)
    ))     

                
    ax.set_ylim([ min(counts), 1.1])
    ax.set_xlim([ min(x_rand),  max(bins)])
  
  
    return alpha, xmin, D







def get_career_length():




    title_font  = 25 
    num_of_bins = 20
    seaborn.set_style('white')  

    f, ax = plt.subplots(1, 2, figsize=(22, 9))
    st = f.suptitle("Career length distributions", fontsize=title_font)    


    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y'),]    
       

    for (label, color) in professions[0:1]:
    
        career_length  = [float(line.strip()) for line in open('ProcessedData/5_career_length/film_career_length_' + label + '.dat')]
        xcareer_length, pcareer_length = getDistribution(career_length)
        

        #ax[0].plot(xcareer_length, pcareer_length, color, marker = 'o', alpha = 0.3, linewidth = 0, label = label+ ', ')
        
        fitPowerLaw([ c for c in career_length if c > 10], ax[0], label)
        
        
        
        #results = powerlaw.Fit(career_length, xmin = min(xcareer_length), fit_method = 'KS')
       # alpha  = results.power_law.alpha
        #D = results.power_law.KS()
     
        #results.plot_pdf(color=color, ax = ax[0],  linestyle='-', linewidth=0, marker = 'o', alpha = 0.5) 
       # results.power_law.plot_pdf(color=color, ax = ax[0],  linestyle='-', linewidth=3, alpha = 0.9,  label ='$\\alpha$= ' + str(round(alpha,2)) + ', $D$='+str(round(D, 2)  ))     








    ax[0].set_title('Length of director careers', fontsize = 18)
    ax[1].set_title('Length of DJ careers', fontsize = 18)    
    professions = [('pop',     'k'), 
                   ('electro', 'b')]    
       
    for (label, color) in professions[1:2]:
    
        career_length  = [float(line.strip()) for line in open('ProcessedData/5_career_length/music_career_length_' + label + '.dat')]
        xcareer_length, pcareer_length = getDistribution(career_length)
        

        fitPowerLaw([ c for c in career_length if c > 10], ax[1], label)
        
        '''
        results = powerlaw.Fit(career_length, xmin = min(xcareer_length), fit_method = 'KS')
        alpha  = results.power_law.alpha
        D = results.power_law.KS()
        results.plot_pdf(color=color, ax = ax[1],  linestyle = '-', linewidth = 0, marker = 'o', alpha = 0.5) 
        
        results.power_law.plot_pdf(color=color, ax = ax[1],  linestyle='-', linewidth=3, alpha = 0.9,  label ='$\\alpha$= ' + str(round(alpha,2)) + ', $D$='+str(round(D, 2)  ))     
        '''


    align_plot1D(ax)
    plt.savefig('career_length.png')
    plt.close()
    #plt.show()










''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''                        GET CORRELATION STUFF                   '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   


def get_rid_of_zeros(imp1, imp2):
        
    imp10 = []
    imp21 = []
    for i in range(len(imp1)):
        if imp1[i] != 0 and imp2[i] != 0:
            imp10.append(imp1[i])
            imp21.append(imp2[i])
            
    return imp10, imp21
    
    


def get_impact_correlations():



    num_of_bins = 12
    title_font = 25 
    seaborn.set_style('white')   
    



    professions = [('director',     'royalblue'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]    
    
    
    for mode in ['', 'Normalized']:
    
    
        f, ax = plt.subplots(2, 3, figsize=(25, 15))
        st = f.suptitle("IMDb impact correlations - " + mode , fontsize=title_font)
        
    
        for (label, color) in professions[0:1]:
            
            impacts = zip(*[ [float(aaa)  if 'tt' not in aaa else aaa for aaa in line.strip().split('\t')]   for line in open('ProcessedData'+mode+'/7_multiple_impacts/film_multiple_impacts_' + label + '.dat')])
            
            Alpha = 0.05
            
            ax[0,0].set_ylabel('avg rating', fontsize = 20)
            ax[0,0].set_xlabel('rating cnt', fontsize = 20)
            ax[0,0].set_xscale('log')            
            avg, cnt = get_rid_of_zeros(impacts[2], impacts[1])
            ax[0,0].plot(impacts[2],  impacts[1], 'o', color = color, alpha = Alpha, label = label)
            xb_avg, pb_avg, pberr_avg = getLogBinnedDistribution(np.asarray(avg), np.asarray(cnt), num_of_bins)    
            ax[0,0].errorbar(xb_avg, pb_avg, yerr = pberr_avg, fmt = '^-', color = 'r')             
     

            x_cnt_meta, p_cnt_meta, perr_cnt_meta = getBinnedDistribution(np.asarray(impacts[3]),  np.asarray(impacts[1]), num_of_bins)        
            meta, avg = get_rid_of_zeros(impacts[3], impacts[1])
            ax[0,1].set_xlabel('metascore', fontsize = 20)
            ax[0,1].plot(meta, avg,  'o', color = color,  alpha = Alpha, label = label)
            ax[0,1].errorbar((x_cnt_meta[1:] + x_cnt_meta[:-1])/2, p_cnt_meta, yerr = perr_cnt_meta, fmt = '^-', color = 'r', label = '$corr=$' + str(round(stats.pearsonr((x_cnt_meta[1:] + x_cnt_meta[:-1])/2, p_cnt_meta)[0] ,4)))#, alpha = Alpha, label = label)
          
            
            
            
            ax[0,2].set_ylabel('#critic review', fontsize = 20)       
            ax[0,2].set_xlabel('#user review', fontsize = 20)
            ax[0,2].set_xscale('log')
            ax[0,2].set_yscale('log')
            crit, user = get_rid_of_zeros(impacts[4], impacts[5]) 
            xb_crit, pb_crit, pberr_crit = getLogBinnedDistribution(np.asarray(crit), np.asarray(user), num_of_bins)    
            ax[0,2].plot(crit, user,  'o', color = color,  alpha = Alpha, label = label)
            ax[0,2].errorbar(xb_crit, pb_crit, yerr = pberr_crit, fmt = '^-', color = 'r')             
           
           
            
            #ax[1,0].xaxis.get_major_formatter().set_powerlimits((0, 1))
            ax[1,0].set_xscale('log')
            cnttt, metatt = get_rid_of_zeros(impacts[2], impacts[3])
            x_cnt_metat, p_cnt_metat, perr_cnt_metat = getLogBinnedDistribution(np.asarray(cnttt), np.asarray(metatt), num_of_bins)        
            ax[1,0].set_xlabel('rating cnt', fontsize = 20)
            ax[1,0].set_ylabel('metascore', fontsize = 20)
            ax[1,0].plot(cnttt, metatt,  'o', color = color,  alpha = Alpha, label = label)
            ax[1,0].errorbar(x_cnt_metat, p_cnt_metat, yerr = perr_cnt_metat, fmt = '^-', color = 'r')#, alpha = Alpha, label = label)

            
            ax[1,1].set_xlabel('rating cnt', fontsize = 20)
            ax[1,1].set_ylabel('#critic review', fontsize = 20)
            cnt, crit = get_rid_of_zeros(impacts[2], impacts[4])
            xb_cnt_crit, pb_cnt_crit, pberr_cnt_crit = getLogBinnedDistribution(np.asarray(cnt), np.asarray(crit), num_of_bins)    
            ax[1,1].loglog(impacts[2],  impacts[4], 'o', color = color,  alpha = Alpha, label = label)
            ax[1,1].errorbar(xb_cnt_crit, pb_cnt_crit, yerr = pberr_cnt_crit, fmt = '^-', color = 'r')             
            
            
            ax[1,2].set_xlabel('rating cnt', fontsize = 20)
            ax[1,2].set_ylabel('#user review', fontsize = 20)
            cnt, user = get_rid_of_zeros(impacts[2], impacts[5])
            xb_cnt_crit, pb_cnt_user, pberr_cnt_user = getLogBinnedDistribution(np.asarray(cnt), np.asarray(user), num_of_bins)    
            ax[1,2].loglog(impacts[2],  impacts[5], 'o', color = color,  alpha = Alpha, label = label)
            ax[1,2].errorbar(xb_cnt_crit, pb_cnt_user, yerr = pberr_cnt_user, fmt = '^-', color = 'r')             

      
        
        align_plot(ax)
        plt.savefig('correlations_'+ mode +'.png')
        plt.close()
        #plt.show()







''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''           GET THE N*/N STUFF OF ALL SUCCESS MEASURES           '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   


def plot_red_lines(ax, x):

    #plt.grid()  
    for i in range(len(ax)):
        for j in range(len(ax[0])):     
            ax[i,j].set_xlim(-0.05, 1.05)
            ax[i,j].set_ylim(-0.05, 1.05)
            #ax[i,j].grid()
            yyy = [1 - y0 for y0 in x]
            ax[i,j].plot(x, yyy, 'v-', linewidth=12, alpha = 0.1) 
            ax[i,j].set_xlabel('$N^{*}/N$', fontsize=17)
            ax[i,j].set_ylabel( r'$P( \geq  N^{*}/N)$' , fontsize=17)



def plot_ccdf(file_avg_all, num_of_bins, ax, color, label, Nmin, title, marker):


    x_Nstar_avg_all, p_Nstar_avg_all, len_career, r_square, numInd = parse_N_star_N_data(file_avg_all, Nmin)
    

    if 'orig' in label:
        ax.set_title(str(numInd) + ' ' + title, fontsize = 19)   
        
    bx_average_ratings, bp_average_ratings, bperr_average_ratings = getBinnedDistribution(np.asarray(x_Nstar_avg_all), np.asarray(p_Nstar_avg_all), num_of_bins)
         
    #ax.plot(x_Nstar_avg_all, p_Nstar_avg_all, color = color,  marker = 'o', linewidth = 0, markersize = 5, alpha= 0.5, label = label + ', ' + str(len_career) + ' $R^2=$' + str(round(r_square, 4)),)  
    ax.errorbar((bx_average_ratings[1:] + bx_average_ratings[:-1])/2, bp_average_ratings, yerr=bperr_average_ratings, fmt=color + '-', linewidth = 1,  markersize = 9, label = label + ' $R^2 = $' + str(round(r_square, 5)), marker = marker, alpha = 0.7)
            
    return r_square            



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
  
        numInd = len(N_star_N)
  
        return x_stat, y_stat, len(N_star_N), r_square, numInd

    except ValueError:
    
        return [], [], 0, 0
        



def get_r_test():


    ''' ---------------------------------------------- '''
    '''      MOVIE YO                                  '''
    
    
    
    num_of_bins = 20
    title_font  = 25 
    Nmin = 20
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(25, 15))
    st = f.suptitle("IMDb impact distributions, $N_{min}$ = " + str(Nmin), fontsize=title_font)

    
    outfile = open('ProcessedDataCombined/R^2_values.dat', 'w')
    outfile.write('domain\tmode\tprofession\tmeasure\tR^2\n')




    for (mode, colorm, markerm) in [('', 'k', 'o'), ('Normalized', 'b', '^'), ('NormalizedRandomized', 'r', 's')]:
    
        
        if '' == mode:
            mode_ = 'original'
        elif 'Rand' in mode:
            mode_ = 'randomized'
        else:
            mode_= 'normalized'
        
        
        folder = 'ProcessedData' + mode #+ 'Sample'

        professions = [('director',     'k',  'o'), 
                       ('producer',     'b',  's'),
                       ('writer'  ,     'r',  '^'),
                       ('composer',     'g',  '8'),
                       ('art-director', 'y',  'x')]


        for (label, color, marker) in professions: #[0:1]:
        
            file_avg_all  = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_avg_rating_'    + label + '.dat'
            file_cnt_all  = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_rating_count_'  + label + '.dat'
            file_mets_all = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_metascores_'    + label + '.dat'
            file_crit_all = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_critic_review_' + label + '.dat'
            file_user_all = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_user_review_'   + label + '.dat'
            
            r_square_avg  = plot_ccdf(file_avg_all,  num_of_bins, ax[0,0], colorm, mode_, Nmin, 'directors, average ratings' , markerm)
            r_square_cnt  = plot_ccdf(file_cnt_all,  num_of_bins, ax[0,1], colorm, mode_, Nmin, 'directors, rating counts'   , markerm)
            r_square_meta = plot_ccdf(file_mets_all, num_of_bins, ax[0,2], colorm, mode_, Nmin, 'directors, metascores'      , markerm)
            r_square_crit = plot_ccdf(file_crit_all, num_of_bins, ax[1,0], colorm, mode_, Nmin, 'directors, #critic reviews' , markerm)
            r_square_user = plot_ccdf(file_user_all, num_of_bins, ax[1,1], colorm, mode_, Nmin, 'directors, #user reviews'   , markerm)

            
            outfile.write( 'film\t' + mode_ + '\t' + label + '\t' + 'average ratings\t' + str(r_square_avg)  + '\n')
            outfile.write( 'film\t' + mode_ + '\t' + label + '\t' + 'rating counts\t'   + str(r_square_cnt)  + '\n')
            outfile.write( 'film\t' + mode_ + '\t' + label + '\t' + 'metascores\t'      + str(r_square_meta) + '\n')
            outfile.write( 'film\t' + mode_ + '\t' + label + '\t' + '#critic reviews\t' + str(r_square_crit) + '\n')
            outfile.write( 'film\t' + mode_ + '\t' + label + '\t' + '#user reviews\t'   + str(r_square_user) + '\n')           




        professions = [('pop',     'k', 'o'), 
                       ('electro', 'b', '^')]    
           
        for (label, color, marker) in professions:#[1:2]:
        
            file_music   = folder + '/4_NN_rank_N/music_best_product_NN_ranks_all_rating_count_' + label + '.dat'
            r_square_cnt = plot_ccdf(file_music,  num_of_bins, ax[1,2], colorm, label, Nmin, 'ratings counts', markerm)
    
            outfile.write(  'music\t' + mode_ + '\t' + label + '\t' + 'rating counts\t' + str(r_square_cnt) + '\n')           



    xxx = np.arange(0,1, 1.0/20)
    plot_red_lines(ax, xxx)
    align_plot(ax)
    #plt.savefig('N_Nstar_'+str(Nmin)+'_first.png')
    #plt.close()
    #plt.show()          
           



def get_ticks(original, colors, color):

    tick_l = []
    tick_x = []

    i = 0
    for (l, x) in original:
        i += 1
        if x < 0.98:
            tick_l.append(l)
            tick_x.append(i)
            colors.append(color)
    
    return tick_x, tick_l


def get_N_star_N_stat():


    original   = []
    normalized = []
    randomized =  []


    for line in open('ProcessedDataCombined/R^2_values.dat'):
    
        if 'mode' not in line:

            fields = line.strip().split('\t')
    
            if 'original' in fields[1]:
                original.append((fields[2] + ',' + fields[3], float(fields[4])))             
            if 'norm' in fields[1]:
                normalized.append((fields[2] + ',' + fields[3], float(fields[4])))             
            if 'rand' in fields[1]:
                randomized.append((fields[2] + ',' + fields[3], float(fields[4])))             
                  


    seaborn.set_style('white')      
    title_font  = 25 
    f, ax = plt.subplots(1, 1, figsize=(12, 12))


   




    font_tick = 15   
    ax.plot([i*1 for i in range(len(original))],   [original[i][1]   for i in range(len(original))],   'ko', label = 'original')
    ax.plot([i*1 for i in range(len(randomized))], [randomized[i][1] for i in range(len(randomized))], 'ro', label = 'randomized')
    ax.plot([i*1 for i in range(len(normalized))], [normalized[i][1] for i in range(len(normalized))], 'bo', label = 'normalized')
    
 
    colors = []
    tick_xo, tick_lo = get_ticks(original,   colors, 'k')
    tick_xn, tick_ln = get_ticks(normalized, colors, 'b')    
    tick_xr, tick_lr = get_ticks(randomized, colors, 'k')   
 
    ax.set_ylim([0.9, 1.01])
       
    ax.legend(loc = 'left', fontsize = font_tick)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()   
    ax.get_yaxis().tick_left()
    ticklines  = ax.get_xticklines()  + ax.get_yticklines()
    gridlines  = ax.get_xgridlines()  + ax.get_ygridlines()

    for line in ticklines:
        line.set_linewidth(1)

    for line in gridlines:
        line.set_linestyle('-.')

    ax.tick_params(labelsize = font_tick) 
    ax.set_xticks(tick_xo + tick_xn + tick_xr) 
    ax.set_xticklabels(tick_lo + tick_ln + tick_lr, rotation = '30')

    for xtick in ax.get_xticks():
        tick_labels = ax.get_xticklabels()
        tick_index  = ax.get_xticks().tolist().index(xtick)
        tick_labels[tick_index].set_color(colors[tick_index])
    

    
    plt.tight_layout()    
    plt.savefig('R^2goodnesses.png')
    plt.close()
    #plt.show()    



    for i in range(len(original)):
        print original[i][0] ,'\t', round(abs(100*(original[i][1] -  randomized[i][1])/ randomized[i][1]),3) ,'\t',round(abs(100*(normalized[i][1] -  randomized[i][1])/ randomized[i][1]),3)





def get_R_square_map():

   
    professions = [('director',     'k',  'o'), 
                   ('producer',     'b',  's'),
                   ('writer'  ,     'r',  '^'),
                   ('composer',     'g',  '8'),
                   ('art-director', 'y',  'x')]

    num_of_bins = 20
    title_font  = 25 
    seaborn.set_style('white')   
    f, ax = plt.subplots(2, 3, figsize=(25, 15))
    st = f.suptitle("$R^2$ values as the function of career cutoffs", fontsize=title_font)
    
    Nmax = 20
    FOLDER = 'ProcessedDataNormalized'

    for (label, color, marker) in professions:
     
        i = professions.index((label, color, marker))
       
              
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
        
    elif sys.argv[1] == '44':
        get_N_star_N_stat()
        
    elif sys.argv[1] == '5':
        get_R_square_map()
        
    elif sys.argv[1] == '6':
        get_impact_correlations()

    elif sys.argv[1] == '7':
        get_exponents()
        
    elif sys.argv[1] == '8':
        get_career_length()
        
  
        
        
        
