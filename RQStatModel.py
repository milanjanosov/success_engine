import os
import sys
#import matplotlib
#matplotlib.use('Agg')
import seaborn
import numpy as np
import random
import powerlaw
import matplotlib.pyplot as plt
from scipy import stats
from CareerTrajectory.careerTrajectory import getDistribution
from CareerTrajectory.careerTrajectory import getBinnedDistribution
from CareerTrajectory.careerTrajectory import getLogBinnedDistribution





''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''                           GENERAL HELPERS                      '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''  


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





''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''          GET THE DISTRIBUTION OF ALL SUCCESS MEASURES          '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''  



def fitPowerLaw(rand, ax, label):

 
    # get the scatterplot
    x_rand, p_rand = getDistribution(rand)
    
    
    # fit and plot the lognormal
    print 'Fitting lognormal...'
    counts, bins, bars = ax.hist(rand, normed = True, bins = 10 ** np.linspace(np.log10(min(x_rand)), np.log10(max(x_rand)), 15), log=True,alpha=0.0)
    ax.plot((bins[1:] + bins[:-1])/2, counts, 's-', color = 'royalblue', alpha = 0.7, markersize = 12, linewidth = 2)

    
    # fit and plot the powerlaw   
    fit   = powerlaw.Fit(rand, xmin = min(x_rand), fit_method = 'KS')
    alpha = fit.power_law.alpha
    xmin  = fit.power_law.xmin 
    D = fit.power_law.KS()
    
    fit.power_law.plot_pdf(color='r',ax=ax,linestyle='-',linewidth=3,label='$\\alpha$= '+str(round(alpha,2))+', $x_{min}$='+str(round(xmin,2))+'\n$D$='+str(round(D, 2)))     

         
    ax.set_title(label, fontsize = 18)               
    ax.set_ylim([ min(counts), 1.1])
    ax.set_xlim([ min(x_rand),  max(bins)])
  
 
    return alpha, xmin, D



def fitSkewedNormal(rand, ax, label, alpha_hist  = 0.2, color_line = 'r'):
   


    ax.set_title(label, fontsize = 18)
    
    param = stats.skewnorm.fit(rand)
    x_rand, p_rand = getDistribution(rand)
    pdf_fitted = stats.skewnorm.pdf(x_rand,  param[0], loc=param[1], scale=param[2])
       
     
    mean = stats.skewnorm.mean( param[0], loc=param[1], scale=param[2])
    maxx = str(x_rand[pdf_fitted.tolist().index(max(pdf_fitted))])
    
    counts, bins, bars = ax.hist(rand, normed = True, bins = np.linspace(min(x_rand), max(x_rand), 25), alpha = alpha_hist)
    sk_results = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.skewnorm.cdf(x_rand, param[0], loc=param[1], scale=param[2]))   # stats.ks_2samp(np.cumsum(p_rand), np.cumsu
    ax.plot(x_rand,pdf_fitted,'-', color = color_line, linewidth = 3, label = '$\\mu$=' + str(round(mean, 2)) + ', $\\mu^{*}$=' +maxx+'\n$D$='+str(round(sk_results[0], 2))+ ', $p$='+str(round(sk_results[1],2)))

    return mean, sk_results[0], sk_results[1], param[0], param[1], param[2]



def get_imapct_distr():             
            

 
    outdir = 'ProcessedDataCombined/9_impact_distributions_fit'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
                    
    out_pow = open(outdir + '/' + 'impact_distribution_power_fits.dat', 'w')
    out_pow.write('norm\tdomain\tmeasure\talpha\txmin\tD\n')
    
    out_norm = open(outdir + '/' + 'impact_distribution_normal_fits.dat', 'w')
    out_norm.write('norm\tdomain\tmeasure\tmu\tD\tp\n')
     
    
    num_of_bins = 20
    title_font  = 25 
    seaborn.set_style('white')   
        
    
    for mode in ['', 'Normalized']:
    

        #if mode == '':
        #    mode_ = 'Original'
        #else:   
        #
        #    mode_ = mode   
        mode_  = 'Original' if mode == '' else 'Normalized'
        FOLDER = 'ProcessedData' + mode #+ 'Sample' 
       
    
    
        ''' ---------------------------------------------- '''
        ''' MOVIES   '''
        
        professions = [('director',     'k'), 
                       ('producer',     'b'),
                       ('writer'  ,     'r'),
                       ('composer',     'g'),
                       ('art-director', 'y')]


        for (label, color) in professions[0:1]:
        
            print mode, label
        
            f, ax = plt.subplots(4, 3, figsize=(23, 33))
            st = f.suptitle( mode + " impact distributions", fontsize=title_font)
           
            
            num_car  = str(int(round(len(os.listdir('Data/Film/film-'+ label +'-simple-careers'))/1000.0))) + 'k'
          
            file_avg  = FOLDER + '/1_impact_distributions/film_average_ratings_dist_' + label + '.dat'
            file_cnt  = FOLDER + '/1_impact_distributions/film_rating_counts_dist_'   + label + '.dat'
            file_mets = FOLDER + '/1_impact_distributions/film_metascores_dist_'      + label + '.dat'
            file_crit = FOLDER + '/1_impact_distributions/film_critic_review_dist_'   + label + '.dat'
            file_user = FOLDER + '/1_impact_distributions/film_user_review_dist_'     + label + '.dat'
            file_gros = FOLDER + '/1_impact_distributions/film_gross_dist_'           + label + '.dat'

            average_ratings = np.asarray([float(line.strip()) for line in open(file_avg)])
            rating_counts   = np.asarray([float(line.strip()) for line in open(file_cnt)])
            metascores      = np.asarray([float(line.strip()) for line in open(file_mets)])
            critic_review   = np.asarray([float(line.strip()) for line in open(file_crit)])
            user_review     = np.asarray([float(line.strip()) for line in open(file_user)])
            gross           = np.asarray([float(line.strip()) for line in open(file_gros)])  
 

            # plot avg ratings
            rating_avg_fit   = fitSkewedNormal(average_ratings, ax[0,0], 'Movie ' + label + ', average ratings')
            ax[0,0].set_xlabel('$\overline{r}_{\\rm IMDb}$', fontsize  = 17)
            ax[0,0].set_ylabel('$P(\overline{r}_{\\rm IMDb}$)', fontsize  = 17)


            rating_cnt_fit   = fitPowerLaw(rating_counts,       ax[1,0], 'imdb ' + label + ' (rating counts)')
            #ax[,].set_xlabel('')
            #ax[,].set_ylabel('')

            rating_mets_fit  = fitSkewedNormal(metascores,      ax[0,1], 'Movie ' + label + ', metascores')          
            ax[0,1].set_xlabel('$\overline{m}_{\\rm Discogs + LFM}$', fontsize  = 17)
            ax[0,1].set_ylabel('$P(\overline{m}_{\\rm Discogs + LFM}$)', fontsize  = 17)
           

            rating_criit_fit = fitPowerLaw(critic_review,       ax[2,0], 'imdb ' + label + ' (critic reviews)')          
            #ax[,].set_xlabel('')
            #ax[,].set_ylabel('')

            rating_user_fit  = fitPowerLaw(user_review,         ax[2,1], 'imdb ' + label + ' (user reviews)')
            #ax[,].set_xlabel('')
            #ax[,].set_ylabel('')

            gross_fit        = fitPowerLaw(gross,               ax[3,0], 'imdb ' + label + ' (gross revenue)')
            #ax[,].set_xlabel('')
            #ax[,].set_ylabel('')

            
             
            out_pow.write(mode_  + '\t' + label + '\t' + 'rating_cnt'     + '\t' + '\t'.join([str(t) for t in rating_cnt_fit])   + '\n')
            out_pow.write(mode_  + '\t' + label + '\t' + 'critic_reviews' + '\t' + '\t'.join([str(t) for t in rating_criit_fit]) + '\n')
            out_pow.write(mode_  + '\t' + label + '\t' + 'user_reviews'   + '\t' + '\t'.join([str(t) for t in rating_user_fit])  + '\n')
            out_norm.write(mode_ + '\t' + label + '\t' + 'avg_rating'     + '\t' + '\t'.join([str(t) for t in rating_avg_fit])   + '\n')
            out_norm.write(mode_ + '\t' + label + '\t' + 'metascore'      + '\t' + '\t'.join([str(t) for t in rating_mets_fit])  + '\n')
            

               
            ''' ---------------------------------------------- '''
            ''' MOVIES   '''
            
            
            genres = [('electro', 'k'),
                      ('pop', 'b')]
                 
                            
            for (genre, color) in genres[0:1]:

                num_mus  = str(int(round(len(os.listdir('Data/Music/music-'+ genre +'-simple-careers'))/1000.0))) + 'k'
                file_music = FOLDER + '/1_impact_distributions/music_rating_counts_dist_' + genre + '.dat'
                rating_counts = np.asarray([float(line.strip()) for line in open(file_music)])    

                rating_cnt_fit   = fitPowerLaw(rating_counts,   ax[1,1], 'Electronic music artists, rating count')
                ax[1,1].set_xlabel('$N_{r, IMDb}$',    fontsize  = 17)
                ax[1,1].set_ylabel('$P(N_{r, IMDb}$)', fontsize  = 17)
            
                out_pow.write(mode_ + '\t' + label + '\t' + 'rating_cnt'     + '\t' + '\t'.join([str(t) for t in rating_cnt_fit])   + '\n')
                          

            
            #num_mus    = str(int(round(len(os.listdir('Data/Book/book-authors-simple-careers'))/1000.0))) + 'k'
            book_avg    = FOLDER + '/1_impact_distributions/book_average_ratings_dist_authors.dat'
            avg_rating  = np.asarray([float(line.strip()) for line in open(book_avg)])    
            avg_rat_fit = fitSkewedNormal(avg_rating, ax[0,2], 'Book authors, average rating')
            ax[0,2].set_xlabel('$\overline{r}_{\\rm Goodreads}$', fontsize  = 18)
            ax[0,2].set_ylabel('$P(\overline{r}_{\\rm Goodreads}$)', fontsize  = 18)    



            book_cnt    = FOLDER + '/1_impact_distributions/book_rating_counts_dist_authors.dat'
            rating_cnt  = np.asarray([float(line.strip()) for line in open(book_cnt)])    
            rat_cnt_fit = fitPowerLaw(rating_cnt, ax[1,2], 'goodreads ' + label + ' (rating count)')  

            book_ed   = FOLDER + '/1_impact_distributions/book_metascores_dist_authors.dat'
            editions  = np.asarray([float(line.strip()) for line in open(book_ed)])    
            ed_fit    = fitPowerLaw(editions, ax[2,2], 'goodreads ' + label + ' (#editions)')  
            


            #rating_cnt_fit   = fitPowerLaw(rating_counts,   ax[1,2], 'electronic music (rating counts)')                
            #out_pow.write(mode_ + '\t' + label + '\t' + 'rating_cnt'     + '\t' + '\t'.join([str(t) for t in rating_cnt_fit])   + '\n')

  
        
        
        align_plot(ax)
        plt.savefig('Figs/fitted_impact_distros_' + label+ mode + '_full.png')
        #plt.close()
        #plt.show()  
                    


    #out_pow.close()
    #out_norm.close()
    





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
    
    
    for mode in ['', 'Normalized'][0:2]:
    
    
       
        
    
        for (label, color) in professions[0:1]:
            

            f, ax = plt.subplots(3, 3, figsize=(25, 25))
            st = f.suptitle("IMDb impact correlations - " + mode + ' - ' + label , fontsize=title_font)


            impacts = zip(*[ [float(aaa)  if 'tt' not in aaa else aaa for aaa in line.strip().split('\t')]   for line in open('ProcessedData'+mode+'Sample/7_multiple_impacts/film_multiple_impacts_' + label + '.dat')])
            
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
    


            ax[2,0].set_xlabel('Rating count', fontsize = 20)
            ax[2,0].set_ylabel('Gross revenue', fontsize = 20)
            cnt, gross = get_rid_of_zeros(impacts[2], impacts[3])


            print impacts[6][0:100]

            xb_cnt, pb_gross, pberr_gross = getLogBinnedDistribution(np.asarray(cnt), np.asarray(gross), num_of_bins)    
            ax[2,0].loglog(cnt, gross, 'o', color = color,  alpha = Alpha, label = label) 
            ax[2,0].errorbar(xb_cnt, pb_gross, yerr = pberr_gross, fmt = '^-', color = 'r')                  
        
            
            
            impacts = zip(*[ [abs(float(aaa.replace(',',''))) for aaa in line.strip().split('\t')[1:]]   for line in open('ProcessedData'+mode+'Sample/7_multiple_impacts/book_multiple_impacts_authors.dat')])


            
            ax[2,1].set_ylabel('Goodreads avg rating', fontsize = 20)
            ax[2,1].set_xlabel('Goodreads rating cnt', fontsize = 20)
            ax[2,1].set_xscale('log')            
            #ax[2,1].set_yscale('log')            
            avg, cnt = get_rid_of_zeros(impacts[0], impacts[1])
 
            ax[2,1].plot(cnt, avg, 'o', color = color, alpha = Alpha, label = label)
            xb_avg, pb_avg, pberr_avg = getLogBinnedDistribution(np.asarray(cnt), np.asarray(avg), num_of_bins)    
            ax[2,1].errorbar(xb_avg, pb_avg, yerr = pberr_avg, fmt = '^-', color = 'r')             
            
            



            ax[2,2].set_ylabel('Goodreads avg rating', fontsize = 20)
            ax[2,2].set_xlabel('Goodreads rating cnt', fontsize = 20)
            ax[2,2].set_xscale('log')            
            ax[2,2].set_yscale('log')            
            ed, cntt = get_rid_of_zeros(impacts[2], impacts[1])
    
            

            ax[2,2].plot(cntt, ed, 'o', color = color, alpha = Alpha, label = label)
            xb_avgc, pb_ed, pberr_ed = getLogBinnedDistribution(np.asarray(cntt), np.asarray(ed), num_of_bins)    


            

            ax[2,2].errorbar(xb_avgc, pb_ed, yerr = pberr_ed, fmt = '^-', color = 'r')             
            





            align_plot(ax)
            plt.savefig('Figs/correlations_'+ mode + '_' + label +'.png')
            plt.close()
            #plt.show()














    
    
    
if __name__ == '__main__':         


    if sys.argv[1] == '1':
        get_imapct_distr()
    elif sys.argv[1] == '2':
        get_impact_correlations()
    
    '''
    elif sys.argv[1] == '4':
        get_p_without_avg()

    elif sys.argv[1] == '9':
        do_the_r_model()
    '''
    
    
    
    
    
    
    
