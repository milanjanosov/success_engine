import os
import sys
import matplotlib
matplotlib.use('Agg')
import seaborn
import numpy as np
import random
import powerlaw
import pylab as pl
import matplotlib.pyplot as plt
from scipy import stats
from CareerTrajectory.careerTrajectory import getDistribution
from CareerTrajectory.careerTrajectory import getBinnedDistribution
from CareerTrajectory.careerTrajectory import getLogBinnedDistribution




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
    
    
    # fit and plot the lognormal
    print 'Fitting lognormal...'
    counts, bins, bars = ax.hist(rand, normed = True, bins = 10 ** np.linspace(np.log10(min(x_rand)), np.log10(max(x_rand)), 15), log=True,alpha=0.0)#,   histtype='step', linewidth = 0)
    ax.plot((bins[1:] + bins[:-1])/2, counts, 's-', color = 'royalblue', alpha = 0.5, markersize = 12, linewidth = 2)

    
    # fit and plot the powerlaw   
    results = powerlaw.Fit(rand, xmin = min(x_rand), fit_method = 'KS')
    alpha  = results.power_law.alpha
    xmin   = results.power_law.xmin 
    D = results.power_law.KS()
 
    results.power_law.plot_pdf(marker = 'o', color='r', ax = ax,  linestyle='-', linewidth=3, label ='$\\alpha$= ' + str(round(alpha,2)) + ', $x_{min}$=' + str(round(xmin,2)) + '\n$D$='+str(round(D, 2)
    ))     

                
    ax.set_ylim([ min(counts), 1.1])
    ax.set_xlim([ min(x_rand),  max(bins)])
  
  
    return alpha, xmin, D





def fitSkewedNormal(rand, ax, label, alpha_hist  = 0.4, color_line = 'r'):
   
    ax.set_title(label, fontsize = 18)
    
    param = stats.skewnorm.fit(rand)
    x_rand, p_rand = getDistribution(rand)
    pdf_fitted = stats.skewnorm.pdf(x_rand,  param[0], loc=param[1], scale=param[2])
     
    
     
    mean = stats.skewnorm.mean( param[0], loc=param[1], scale=param[2])
    maxx = x_rand[pdf_fitted.tolist().index(max(pdf_fitted))]
    
    counts, bins, bars = ax.hist(rand, normed = True, bins = np.linspace(min(x_rand), max(x_rand), 25), alpha = alpha_hist)
    sk_results_norm = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.skewnorm.cdf(x_rand, param[0], loc=param[1], scale=param[2]))   # stats.ks_2samp(np.cumsum(p_rand), np.cumsu
    ax.plot(x_rand,pdf_fitted,'-', color = color_line, linewidth = 3, label = '$\\mu$=' + str(round(mean, 2)) + ', $\\mu^{*}$=' + str(maxx) + '\n$D$='+str(round(sk_results_norm[0], 2))+ ', $p$='+str(round(sk_results_norm[1],2)))

    return mean, sk_results_norm[0], sk_results_norm[1], param[0], param[1], param[2]







   
def get_imapct_distr():             
            

 
    dir9 = 'ProcessedDataCombined/9_impact_distributions_fit'
    if not os.path.exists(dir9):
        os.makedirs(dir9)
                    
    out_pow = open(dir9 + '/' + 'impact_distribution_power_fits.dat', 'w')
    out_pow.write('norm\tdomain\tmeasure\talpha\txmin\tD\n')
    
    out_norm = open(dir9 + '/' + 'impact_distribution_normal_fits.dat', 'w')
    out_norm.write('norm\tdomain\tmeasure\tmu\tD\tp\n')
      
      
    for mode in ['', 'Normalized']:
    

        if mode == '':
            mode_ = 'Original'
        else:   
            mode_ = mode   
    
    
        FOLDER = 'ProcessedData' + mode# + 'Sample' 
       
    
        professions = [('director',     'k'), 
                       ('producer',     'b'),
                       ('writer'  ,     'r'),
                       ('composer',     'g'),
                       ('art-director', 'y')]


        num_of_bins = 20
        title_font  = 25 
        seaborn.set_style('white')   
        


        for (label, color) in professions:
        
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
                rating_counts = np.asarray([float(line.strip()) for line in open(file_music)])    

                


                rating_cnt_fit   = fitPowerLaw(rating_counts,   ax[1,2], 'electronic music (rating counts)')                
                out_pow.write(mode_ + '\t' + label + '\t' + 'rating_cnt'     + '\t' + '\t'.join([str(t) for t in rating_cnt_fit])   + '\n')
                        
           
            align_plot(ax)
            #plt.savefig('Figs/fitted_impact_distros_' + label+ mode + '_full.png')
            plt.close()
            #plt.show()  
                    


    out_pow.close()
    out_norm.close()






def fitNormal(rand, ax, label, alpha_hist  = 0.4, color_line = 'r'):
   
    ax.set_title(label, fontsize = 18)
    
    param = stats.skewnorm.fit(rand)
    x_rand, p_rand = getDistribution(rand)
    pdf_fitted = stats.skewnorm.pdf(x_rand,  param[0], loc=param[1], scale=param[2])
     
    
     
    mean = stats.skewnorm.mean( param[0], loc=param[1], scale=param[2])
    maxx = x_rand[pdf_fitted.tolist().index(max(pdf_fitted))]
    
    counts, bins, bars = ax.hist(rand, normed = True, bins = np.linspace(min(x_rand), max(x_rand), 50), alpha = alpha_hist)
    sk_results_norm = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.skewnorm.cdf(x_rand, param[0], loc=param[1], scale=param[2]))   # stats.ks_2samp(np.cumsum(p_rand), np.cumsu
    ax.plot(x_rand,pdf_fitted,'-', color = color_line, linewidth = 3, label = '$\\mu$=' + str(round(mean, 2)) + ', $\\mu^{*}$=' + str(maxx) + '\n$D$='+str(round(sk_results_norm[0], 2))+ ', $p$='+str(round(sk_results_norm[1],2)))

    return mean, sk_results_norm[0], sk_results_norm[1], param[0], param[1], param[2]









def get_p_without_avg():

    
    for mode in ['', 'Normalized']:
    

        if mode == '':
            mode_ = 'Original'
        else:   
            mode_ = mode   
    
    
        FOLDER = 'ProcessedData' + mode + 'Sample' 
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
        st = f.suptitle( mode + "  $\log(p_{\\alpha}) + \mu_p$ distributions ", fontsize=title_font)

        field = 'film'
        
        for (label, color) in professions[0:1]:
        
        
           
          
            file_avg  = FOLDER + dir9 + '/' + field + '_p_without_mean_avg_rating_' + label + '.dat'
            file_cnt  = FOLDER + dir9 + '/' + field + '_p_without_mean_rating_cnt_' + label + '.dat'        
            file_mets = FOLDER + dir9 + '/' + field + '_p_without_mean_metascore_'  + label + '.dat'   
            file_crit = FOLDER + dir9 + '/' + field + '_p_without_mean_critic_rev_' + label + '.dat'   
            file_user = FOLDER + dir9 + '/' + field + '_p_without_mean_user_rev_'   + label + '.dat' 

            average_ratings = np.asarray([float(line.strip()) for line in open(file_avg)  if float(line.strip()) != 0  ])
            rating_counts   = np.asarray([float(line.strip()) for line in open(file_cnt)  if float(line.strip()) != 0  ])
            metascores      = np.asarray([float(line.strip()) for line in open(file_mets) if float(line.strip()) != 0  ])
            critic_review   = np.asarray([float(line.strip()) for line in open(file_crit) if float(line.strip()) != 0  ])
            user_review     = np.asarray([float(line.strip()) for line in open(file_user) if float(line.strip()) != 0  ])

            fitNormal(average_ratings, ax[0,0], label + ', avg rating')
            fitNormal(rating_counts,   ax[0,1], label + ', rating count')
            fitNormal(metascores,      ax[0,2], label + ', metascore')
            fitNormal(critic_review,   ax[1,0], label + ', critic reviews')
            fitNormal(user_review,     ax[1,1], label + ', user reviews')
            
            #ax[0,0].hist(average_ratings, bins = 50)
            #ax[0,1].hist(rating_counts, bins = 50)  
           # ax[0,2].hist(metascores, bins = 50)   
            #ax[1,0].hist(critic_review, bins = 50)   
            #ax[1,1].hist(user_review, bins = 50)   
   
   


        genres = [('electro', 'k'),
                  ('pop', 'b')]
             
        field = 'music'                     
        for (genre, color) in genres[0:1]:
            file_cnt  = FOLDER + dir9 + '/' + field + '_p_without_mean_rating_cnt_' + genre + '.dat'        
            rating_counts   = np.asarray([float(line.strip()) for line in open(file_cnt)  if float(line.strip()) != 0  ])   

            #ax[1,2].hist(rating_counts, bins = 50)   
            fitNormal(rating_counts, ax[1,2], 'DJs, playcount')
        
        
        
        align_plot(ax)        
        plt.savefig('p+mu_p_distributions' + mode + '.png')
        plt.show() 
        
        



def get_impact_fits():


    dir9 = 'ProcessedDataCombined/9_impact_distributions_fit'
 
 
    original_p   = []
    normalized_p = []
 
    original_n   = []
    normalized_n = [] 
    
    for line in open(dir9 + '/' + 'impact_distribution_power_fits.dat'):
        
        fields = line.strip().split('\t')
        if 'D' not in line:
            if 'Norm' in fields[0]:
                normalized_p.append(('pow - ' + fields[1] + ' - ' + fields[2], float(fields[-1])))
            else:
                original_p.append(('pow - ' + fields[1] + ' - ' + fields[2], float(fields[-1])))
            
            
    for line in open(dir9 + '/' + 'impact_distribution_normal_fits.dat'):
        if 'D' not in line:        
            fields = line.strip().split('\t')
            if 'Norm' in fields[0]:
                normalized_n.append(('norm - ' + fields[1] + ' - ' + fields[2], float(fields[-2])))
            else:
                original_n.append(('norm - ' + fields[1] + ' - ' + fields[2], float(fields[-2])))        
                
    
   
   
   
    seaborn.set_style('white')      
    title_font  = 25 
    f, ax = plt.subplots(1, 1, figsize=(15, 15))


    font_tick = 15   
    ax.set_xlim([0,5])
    ax.plot([1] * len(original_p)   , [n[1] for n in original_p  ], 'ko', label = 'original,   powerlaw', markersize = 12, alpha = 0.7)
    ax.plot([2] * len(normalized_n) , [n[1] for n in normalized_n], 'ys', label = 'normalized, skewnorm', markersize = 12, alpha = 0.7)
    ax.plot([3] * len(normalized_p) , [n[1] for n in normalized_p], 'r^', label = 'normalized, powerlaw', markersize = 12, alpha = 0.7)
    ax.plot([4] * len(original_n)   , [n[1] for n in original_n]  , 'b8', label = 'original,   skewnorm', markersize = 12, alpha = 0.7)

    
    ax.legend(loc = 'left', fontsize = font_tick)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ticklines  = ax.get_xticklines()  + ax.get_yticklines()
    gridlines  = ax.get_xgridlines()  + ax.get_ygridlines()
    ticklabels = ax.get_xticklabels() + ax.get_yticklabels()
    for line in ticklines:
        line.set_linewidth(1)

    for line in gridlines:
        line.set_linestyle('-.')

    ax.tick_params(labelsize = font_tick) 


    plt.show()    
      




''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                                  '''   
'''                                 DO THE R MODEL OLD                               '''
'''                                                                                  '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  ''' 


def get_r_model_curves(data_file, max_data_file, ax, label, num_of_bins, title, xlabel, ylabel, log = False):


    ax.set_title(title,   fontsize = 19)
    ax.set_xlabel(xlabel, fontsize = 17)
    ax.set_ylabel(ylabel, fontsize = 17)


    data = [float(line.strip()) for line in open(data_file)]
    (data_max, career_len) = zip(*[[float(num) for num in line.strip().split('\t')] for line in open(max_data_file)  if 'nan' not in line])
    ax.plot(career_len, data_max, marker = 'o', color = 'lightgrey', alpha = 0.15,linewidth = 0)
    
    
    career_max = []
    career_max_dict = {}
    
    
    for i in range(100):

        data_new = data[:]
        random.shuffle(data_new)
      
        for leng in career_len:
            #career_max.append(max( data_new[0:int(leng)]))
            
            if leng not in career_max_dict:
                career_max_dict[leng] = [max( data_new[0:int(leng)])]
            else:
                career_max_dict[leng].append( max( data_new[0:int(leng)]))          
            
            del data_new[0:int(leng)]


    sorted_len = sorted(list(set(career_len)))
    career_max = []
    for s in sorted_len:
       career_max.append(np.mean(career_max_dict[s]))
    
    print len(sorted_len), len(career_max)
      
      
      
    if not log:
        xb_data, pb_data, pberr_data = getBinnedDistribution(np.asarray(career_len),  np.asarray(data_max), num_of_bins)         
        xb_gen, pb_gen, pberr_gen    = getBinnedDistribution(np.asarray(sorted_len),  np.asarray(career_max), num_of_bins)
        ax.errorbar((xb_data[1:] + xb_data[:-1])/2, pb_data, yerr = pberr_data, fmt = 'o-', color = 'grey', label = 'data', alpha = 0.9)
        ax.errorbar((xb_gen[1:]  + xb_gen[:-1])/2, pb_gen, yerr = pberr_gen, fmt = '-', color = 'r', label = 'R-model', alpha = 0.9)
    else:
        ax.set_xscale('log')
        ax.set_yscale('log')
        xb_data, pb_data, pberr_data = getLogBinnedDistribution(np.asarray(career_len),  np.asarray(data_max), num_of_bins)         
        xb_gen, pb_gen, pberr_gen    = getLogBinnedDistribution(np.asarray(sorted_len),  np.asarray(career_max), num_of_bins)
        ax.errorbar(xb_data, pb_data, yerr = pberr_data, fmt = 'o-', color = 'grey', label = 'data', alpha = 0.9)
        ax.errorbar(xb_gen,  pb_gen,  yerr = pberr_gen, fmt = '-', color = 'r', label = 'R-model', alpha = 0.9)        
    



def do_the_r_model():


    title_font  = 25 
    num_of_bins = 8
    seaborn.set_style('white')  


    f, ax = plt.subplots(2, 3, figsize=(25, 15))
    st = f.suptitle( "R - model vs data (movie directors and DJs)", fontsize=title_font)
    
    
    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y'),]    
       



    FOLDER = 'ProcessedData'#Sample' # mode# + 'Sample' 
    field  = 'film'
    
    for (label, color) in professions[0:1]:


        file_avg  = FOLDER + '/1_impact_distributions/film_average_ratings_dist_' + label + '.dat'
        file_cnt  = FOLDER + '/1_impact_distributions/film_rating_counts_dist_'   + label + '.dat'
        file_meta = FOLDER + '/1_impact_distributions/film_metascores_dist_'      + label + '.dat'
        file_crit = FOLDER + '/1_impact_distributions/film_critic_review_dist_'   + label + '.dat'
        file_user = FOLDER + '/1_impact_distributions/film_user_review_dist_'     + label + '.dat'
        
        
        max_avg_rat  = FOLDER + '/10_career_length_max_impact/career_length_max_avg_rat'    + field + '_' + label + '.dat'
        max_rat_cnt  = FOLDER + '/10_career_length_max_impact/career_length_max_rat_cnt'    + field + '_' + label + '.dat'
        max_meta     = FOLDER + '/10_career_length_max_impact/career_length_max_metascore'  + field + '_' + label + '.dat'     
        max_crit_rev = FOLDER + '/10_career_length_max_impact/career_length_max_crit_rev'   + field + '_' + label + '.dat'
        max_user_rev = FOLDER + '/10_career_length_max_impact/career_length_max_user_rev'   + field + '_' + label + '.dat'


        get_r_model_curves(file_avg,  max_avg_rat,  ax[0,0], label, num_of_bins, 'Average rating vs career length' , 'Career length', 'Average rating' )
        get_r_model_curves(file_meta, max_meta,     ax[0,2], label, num_of_bins, 'Metascore vs career length'      , 'Career length', 'Metascore'           )
        get_r_model_curves(file_cnt,  max_rat_cnt,  ax[0,1], label, num_of_bins, 'Rating count vs career length'   , 'Career length', 'Rating count'  , True)            
        get_r_model_curves(file_crit, max_crit_rev, ax[1,0], label, num_of_bins, 'Critic reviews vs career length' , 'Career length', 'Critic reviews', True)    
        get_r_model_curves(file_user, max_user_rev, ax[1,1], label, num_of_bins, 'User reviews vs career length'   , 'Career length', 'User reviews'  , True)            
        
   
       
    
    field  =   'music'
    genres = [('electro', 'k'),
              ('pop'    , 'b')]
                           
    for (genre, color) in genres[0:1]:   
        file_music = FOLDER + '/1_impact_distributions/music_rating_counts_dist_' + genre + '.dat'
        max_music  = FOLDER + '/10_career_length_max_impact/career_length_max_rat_cntmusic_' + genre + '.dat'
    
        get_r_model_curves(file_music, max_music, ax[1,2], genre, num_of_bins, 'Rating count vs career length', 'Career length', 'Rating count', True)   
   
          
           
    align_plot(ax) 
    plt.savefig('R-model.png')       
    #plt.show()                 
    
       
       
       
       
       
     








    
if __name__ == '__main__':         


    if sys.argv[1] == '1':
        get_imapct_distr()
    elif sys.argv[1] == '2':
        get_impact_fits()
    elif sys.argv[1] == '4':
        get_p_without_avg()

    elif sys.argv[1] == '9':
        do_the_r_model()



