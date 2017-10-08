import os
import sys
import seaborn
import numpy as np
import powerlaw
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from CareerTrajectory.careerTrajectory import getDistribution
from CareerTrajectory.careerTrajectory import getBinnedDistribution





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





def fitPowerLaw(rand, ax, label):


   
    ax.set_title(label, fontsize = 18)

    # get the scatterplot
    x_rand, p_rand = getDistribution(rand)
    
    
    # fit and plot the powerlaw   
    results = powerlaw.Fit(rand, xmin = min(x_rand))
    fit = powerlaw.Fit(rand)
    alpha = fit.power_law.alpha
    xmin  = fit.power_law.xmin
    sk_results_pow = stats.kstest(results.pdf()[1], 'powerlaw', [alpha, xmin]) 
       
    results.power_law.plot_pdf(color='r', ax = ax,  linestyle='-', linewidth = 3, label = '$\\alpha$= ' + str(round(alpha,2)) + ', $x_{min}$=' + str(round(xmin,2)) + '\n$D$='+str(round(sk_results_pow[0], 2))+ ', $p$='+str(round(sk_results_pow[1],2)) )     

    
            
    # fit and plot the lognormal
    print 'Fitting lognormal...'
    #param = stats.lognorm.fit(rand)

    #pdf_fitted = stats.lognorm.pdf(x_rand, param[0], loc=param[1], scale=param[2])#
    #mu =  np.log(param[2])
    #sigma = param[0]

    counts, bins, bars = ax.hist(rand, normed = True, bins = 10 ** np.linspace(np.log10(min(x_rand)), np.log10(max(x_rand)), 15), log=True,alpha=0.0)#,   histtype='step', linewidth = 0)
    #sk_results_norm = stats.kstest(counts, 'lognorm', param)   
    ax.plot((bins[1:] + bins[:-1])/2, counts, 's-', color = 'royalblue',alpha = 0.5, markersize = 12, linewidth = 2)
    #ax.plot(x_rand,pdf_fitted,'r-', label = '$\\mu$=' + str(round(mu,2)) + ' $\\sigma$=' + str(round(sigma, 2)) + '\n$D$='+str(round(sk_results_norm[0], 2))+ ', $p$='+str(round(sk_results_norm[1],2)) )

    ax.set_ylim([ min(counts), 1.1])
    ax.set_xlim([ min(x_rand),  max(bins)])

     
     
     
     
     
     
    return alpha, xmin, sk_results_pow[0], sk_results_pow[1]#, mu, sigma, sk_results_norm[0], sk_results_norm[1]





def fitSkewedNormal(rand, ax, label):


    
    ax.set_title(label, fontsize = 18)
    
    param = stats.skewnorm.fit(rand)
    x_rand, p_rand = getDistribution(rand)
    pdf_fitted = stats.skewnorm.pdf(x_rand,  param[0], loc=param[1], scale=param[2])
     
    mean = stats.skewnorm.mean( param[0], loc=param[1], scale=param[2])
    
    counts, bins, bars = ax.hist(rand, normed = True, bins = np.linspace(min(x_rand), max(x_rand), 25), alpha=0.4)
    sk_results_norm = stats.kstest(counts, 'skewnorm', param)       
    ax.plot(x_rand,pdf_fitted,'r-', linewidth = 3, label = '$\\mu$=' + str(round(mean, 2)) + '\n$D$='+str(round(sk_results_norm[0], 2))+ ', $p$='+str(round(sk_results_norm[1],2)))

    return mean, sk_results_norm[0], sk_results_norm[1]




   
def get_imapct_distr():             
            


    ''' THIS CONFIRMS THAT THE IMPACT MEASURES OF INTEREST FOLLOW LOGNORMAL NATURE '''
    ''' THEREFORE THE IMPACT DISTR COULD BE THE RESULT OF A MULTIPLICATIVE PROCESS '''
     
    dir9 = 'ProcessedDataCombined/9_impact_distributions_fit'
    if not os.path.exists(dir9):
        os.makedirs(dir9)
                    
    out_pow = open(dir9 + '/' + 'impact_distribution_power_fits.dat', 'w')
    out_pow.write('norm\tdomain\tmeasure\talpha\txmin\tD\tp\n')
    
    out_norm = open(dir9 + '/' + 'impact_distribution_normal_fits.dat', 'w')
    out_norm.write('norm\tdomain\tmeasure\tmu\tD\tp\n')
      
      
    for mode in ['', 'Normalized'][0:1]:
    

        if mode == '':
            mode_ = 'Original'
        else:   
            mode_ = mode   
    
    
        FOLDER = 'ProcessedData' + mode + 'Sample' 
       
    
        professions = [('director',     'k'), 
                       ('producer',     'b'),
                       ('writer'  ,     'r'),
                       ('composer',     'g'),
                       ('art-director', 'y')]


        num_of_bins = 20
        title_font  = 25 
        seaborn.set_style('white')   
        


        for (label, color) in professions[0:2]:
        
            print mode, label
        
            f, ax = plt.subplots(2, 3, figsize=(25, 15))
            st = f.suptitle( mode + " impact distributions", fontsize=title_font)
           
            num_car  = str(int(round(len(os.listdir('Data/Film/film-'+ label +'-simple-careers'))/1000.0))) + 'k'
          
            file_avg  = FOLDER + '/1_impact_distributions/film_average_ratings_dist_' + label + '.dat'
            file_cnt  = FOLDER + '/1_impact_distributions/film_rating_counts_dist_'   + label + '.dat'
            file_mets = FOLDER + '/1_impact_distributions/film_metascores_dist_'      + label + '.dat'
            file_crit = FOLDER + '/1_impact_distributions/film_critic_review_dist_'   + label + '.dat'
            file_user = FOLDER + '/1_impact_distributions/film_user_review_dist_'     + label + '.dat'

            average_ratings = np.asarray([float(line.strip()) for line in open(file_avg)])
            rating_counts   = np.asarray([float(line.strip()) for line in open(file_cnt)])
            metascores      = np.asarray([float(line.strip()) for line in open(file_mets)])
            critic_review   = np.asarray([float(line.strip()) for line in open(file_crit)])
            user_review     = np.asarray([float(line.strip()) for line in open(file_user)])
            
                      
            # plot avg ratings
            rating_avg_fit   = fitSkewedNormal(average_ratings, ax[0,0], 'imdb ' + label + ' (avg rating)')                     
            rating_cnt_fit   = fitPowerLaw(rating_counts,       ax[0,1], 'imdb ' + label + ' (rating counts)')
            rating_mets_fit  = fitSkewedNormal(metascores,      ax[0,2], 'imdb ' + label + ' (metascore)')          
            rating_criit_fit = fitPowerLaw(critic_review,       ax[1,0], 'imdb ' + label + ' (critic reviews)')          
            rating_user_fit  = fitPowerLaw(user_review,         ax[1,1], 'imdb ' + label + ' (user reviews)')
            
             
            out_pow.write(mode_  + '\t' + label + '\t' + 'rating_cnt'     + '\t' + '\t'.join([str(t) for t in rating_cnt_fit])   + '\n')
            out_pow.write(mode_  + '\t' + label + '\t' + 'critic_reviews' + '\t' + '\t'.join([str(t) for t in rating_criit_fit]) + '\n')
            out_pow.write(mode_  + '\t' + label + '\t' + 'user_reviews'   + '\t' + '\t'.join([str(t) for t in rating_user_fit])  + '\n')
            out_norm.write(mode_ + '\t' + label + '\t' + 'avg_rating'     + '\t' + '\t'.join([str(t) for t in rating_avg_fit])   + '\n')
            out_norm.write(mode_ + '\t' + label + '\t' + 'metascore'      + '\t' + '\t'.join([str(t) for t in rating_mets_fit])  + '\n')
            

            ''' ---------------------------------------------- '''
            '''      MUSIC YO                                  '''
        
            genres = [('electro', 'k'),
                      ('pop', 'b')]
             
             
                  
            for (genre, color) in genres[0:1]:

                num_mus  = str(int(round(len(os.listdir('Data/Music/music-'+ genre +'-simple-careers'))/1000.0))) + 'k'
                file_music = FOLDER + '/1_impact_distributions/music_rating_counts_dist_' + genre + '.dat'
                rating_counts = np.asarray([round(float(line.strip())) for line in open(file_music)])    

                rating_cnt_fit   = fitPowerLaw(rating_counts,   ax[1,2], 'electronic music (rating counts)')                
                out_pow.write(mode_ + '\t' + label + '\t' + 'rating_cnt'     + '\t' + '\t'.join([str(t) for t in rating_cnt_fit])   + '\n')
            
            
           
            align_plot(ax)
            plt.savefig('Figs/fitted_impact_distros_' + label+ mode + '_full.png')
            plt.close()
            #plt.show()  
                    


    out_pow.close()
    out_norm.close()



def get_p_without_avg():

    
    for mode in ['', 'Normalized']:
    

        if mode == '':
            mode_ = 'Original'
        else:   
            mode_ = mode   
    
    
        FOLDER = 'ProcessedData' + mode #+ 'Sample' 
        dir9   = '/9_p_without_avg'
    
        professions = [('director',     'k'), 
                       ('producer',     'b'),
                       ('writer'  ,     'r'),
                       ('composer',     'g'),
                       ('art-director', 'y')]


        num_of_bins = 20
        title_font  = 25 
        seaborn.set_style('white')   
        f, ax = plt.subplots(2, 3, figsize=(25, 15))
        st = f.suptitle("IMDb " + mode + " p (without mean distributions) -  Directors", fontsize=title_font)

        field = 'film'
        
        for (label, color) in professions[0:1]:
        
        
           
            num_car  = str(int(round(len(os.listdir('Data/Film/film-'+ label +'-simple-careers'))/1000.0))) + 'k'
          
            file_avg  = FOLDER + dir9 + '/' + field + '_p_without_mean_avg_rating_' + label + '.dat'
            file_cnt  = FOLDER + dir9 + '/' + field + '_p_without_mean_rating_cnt_' + label + '.dat'        
            file_mets = FOLDER + dir9 + '/' + field + '_p_without_mean_metascore_'  + label + '.dat'   
            file_crit = FOLDER + dir9 + '/' + field + '_p_without_mean_critic_rev_' + label + '.dat'   
            file_user = FOLDER + dir9 + '/' + field + '_p_without_mean_user_rev_'   + label + '.dat' 

            average_ratings = np.asarray([float(line.strip()) for line in open(file_avg)  if float(line.strip()) != 0  ])
            rating_counts   = np.asarray([float(line.strip()) for line in open(file_cnt)])
            metascores      = np.asarray([float(line.strip()) for line in open(file_mets)])
            critic_review   = np.asarray([float(line.strip()) for line in open(file_crit)])
            user_review     = np.asarray([float(line.strip()) for line in open(file_user)])

            #ax[0,0].set_yscale('log')
            print ax[0,0].hist(average_ratings, bins = 50)
  
   

                 
        align_plot(ax)        
        plt.show() 
        
        



    
if __name__ == '__main__':         


    if sys.argv[1] == '1':
        get_imapct_distr()
    elif sys.argv[1] == '2':
        get_p_without_avg()





