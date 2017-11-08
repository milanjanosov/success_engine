import os
import sys
import matplotlib
matplotlib.use('Agg')
import seaborn
import numpy as np
import random
import powerlaw
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from CareerTrajectory.careerTrajectory import getDistribution
from CareerTrajectory.careerTrajectory import getBinnedDistribution
from CareerTrajectory.careerTrajectory import getLogBinnedDistribution




''' TODO '''
'''





OK 1. impact distr for
    -  min rating 0
    -  original
    -  normalized

OK 2. inflation curve for everyone and all product 
    -  min rating 0
    -  original

3. N*/N
    -  normalized
    -  N > 15
    -  min rating 0
    -  min rating 15

4. r-model predictions
    -  amire az N*/N jo

5. p distribution 
    -  amire az N*/N jo

6. career len distribution
    -  amire az N*/N jo


- lack of correlation between papers
- 




-- ELETKORT AHOL LEHET BELEVENNI A 100+ eves dolgok mittt 



'''







''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''                           GENERAL HELPERS                      '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''  


def align_ax(ax, font_tick):


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

    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15   
    ax.tick_params(labelsize = font_tick)    


def align_plot(ax):

    font_tick = 21   

    if len(ax.shape)> 1:
        for i in range(len(ax)):
            for j in range(len(ax[0])):
                align_ax(ax[i,j], font_tick)  
 
    else:
        for i in range(len(ax)):
            align_ax(ax[i], font_tick)
       

def write_row(filename, data):

    f = open(filename, 'w')
    [f.write(str(dat)+'\n') for dat in data ]
    f.close()    



''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''          GET THE DISTRIBUTION OF ALL SUCCESS MEASURES          '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''  


def fitPowerLaw(filename, ax, label):

    rand = np.asarray([float(line.strip()) for line in open(filename)]) 

    x_rand, p_rand = getDistribution(rand)                  
    ax.set_title(label, fontsize = 18)               
    
    # histogram
    counts, bins, bars = ax.hist(rand, normed = True, bins = 10 ** np.linspace(np.log10(min(x_rand)), np.log10(max(x_rand)), 15), log=True,alpha=0.0)
    ax.plot((bins[1:] + bins[:-1])/2, counts, 's-', color = 'royalblue', alpha = 0.7, markersize = 12, linewidth = 2)
    ax.set_ylim([ min(counts), 1.1])
    ax.set_xlim([ min(x_rand),  max(bins)])

    
    # fit and plot the powerlaw   
    print 'Fit and plot the powerlaw...'
    results  = powerlaw.Fit(rand, xmin = min(x_rand), fit_method = 'KS')
    alpha    = results.power_law.alpha
    xmin     = results.power_law.xmin 
    D        = results.power_law.KS()  
    parassms = results.power_law.plot_pdf(color='r',ax=ax,linestyle='-',linewidth=3,label='$\\alpha$= '+str(round(alpha,2))+', $x_{min}$='+str(round(xmin,2))+'\n$D$='+str(round(D, 2)))     
  
    
    # fit and plot the powerlaw   
    print 'Fit and plot the lognormal...'
    param = stats.lognorm.fit(rand)
    pdf_fitted = stats.lognorm.pdf(x_rand, param[0], loc=param[1], scale=param[2])#
    mu =  np.log(param[2])
    sigma = param[0]
    sk_results_norm = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.lognorm.cdf(x_rand, param[0], loc=param[1], scale=param[2]))   # stats.ks_2samp(np.cumsum(p_rand), np.cumsu
    ax.plot(x_rand,pdf_fitted,'k-', linewidth = 4, label = 'Lognormal fit, $\\mu$=' + str(round(mu,2)) + '\n$\\sigma$=' + str(round(sigma, 2)) + ', $D$='+str(round(sk_results_norm[0], 2)))
   
   
    xfit = parassms.lines[1].get_xdata()
    yfit = parassms.lines[1].get_ydata()     
    write_row('Viz/1_impact_fit/powerlaw_hist' + label + '.dat', rand)
    write_row('Viz/1_impact_fit/powerlaw_fit'  + label + '.dat', [str(xfit[i]) + '\t' + str(yfit[i]) for i in range(len(xfit))] )   
    
    
  
def fitSkewedNormal(filename, ax, label, alpha_hist  = 0.2, color_line = 'r'):
   

    rand = np.asarray([float(line.strip()) for line in open(filename)])

    print 'Fitting normal...'
    param = stats.skewnorm.fit(rand)
    x_rand, p_rand = getDistribution(rand)
    pdf_fitted = stats.skewnorm.pdf(x_rand,  param[0], loc=param[1], scale=param[2])
            
    mean = stats.skewnorm.mean( param[0], loc=param[1], scale=param[2])
    maxx = str(x_rand[pdf_fitted.tolist().index(max(pdf_fitted))])
    
    counts, bins, bars = ax.hist(rand, normed = True, bins = np.linspace(min(x_rand), max(x_rand), 25), alpha = alpha_hist)
    sk_results = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.skewnorm.cdf(x_rand, param[0], loc=param[1], scale=param[2]))   # stats.ks_2samp(np.cumsum(p_rand), np.cumsu
    ax.plot(x_rand,pdf_fitted,'-', color = color_line, linewidth = 3, label = '$\\mu$=' + str(round(mean, 2)) + ', $\\mu^{*}$=' +maxx+'\n$D$='+str(round(sk_results[0], 2))+ ', $p$='+str(round(sk_results[1],2)))
    ax.set_title(label, fontsize = 18)  
  
    write_row('Viz/1_impact_fit/normal_hist' + label + '.dat', rand)   
    write_row('Viz/1_impact_fit/normal_fit' + label + '.dat',  [str(x_rand[i]) + '\t' + str(pdf_fitted[i]) for i in range(len(x_rand))])      


   
def get_imapct_distr():             
            
              
    num_of_bins = 20
    title_font  = 25 
    seaborn.set_style('white')   
        
    
    for mode in ['', 'Normalized']:
    

        mode_    = 'Original' if mode == '' else 'Normalized'
        FOLDER_S = 'ProcessedDataSample/ProcessedData_0_' + mode + 'Sample' 
        FOLDER   = 'ProcessedData/ProcessedData_0_' + mode# + 'Sample'        
  
        ''' ---------------------------------------------- '''
        ''' MOVIES   '''
        
        professions = ['_MERGED', 'director', 'producer', 'writer', 'composer', 'art-director']
   
        for label in professions:
            
            print 'PROCESSING -- ' + label
            f, ax = plt.subplots(3, 2, figsize=(23, 23))
            st = f.suptitle( mode + "IMDb impact distributions for " + label, fontsize=title_font)

   
            file_avg  = FOLDER + '/1_impact_distributions/film_average_rating_dist_' + label + '.dat'           
            file_cnt  = FOLDER + '/1_impact_distributions/film_rating_count_dist_'   + label + '.dat'
            file_meta = FOLDER + '/1_impact_distributions/film_metascore_dist_'      + label + '.dat'
            file_crit = FOLDER + '/1_impact_distributions/film_critic_reviews_dist_' + label + '.dat'
            file_user = FOLDER + '/1_impact_distributions/film_user_reviews_dist_'   + label + '.dat'
            file_gros = FOLDER + '/1_impact_distributions/film_gross_revenue_dist_'  + label + '.dat'

            fitSkewedNormal(file_avg,   ax[0,0], 'IMDb, average ratings' + label)
            fitPowerLaw    (file_cnt,   ax[1,0], 'IMDb, rating counts'   + label)
            fitSkewedNormal(file_meta,  ax[0,1], 'IMDb, metascores'      + label)             
            fitPowerLaw    (file_crit,  ax[1,1], 'IMDb, critic reviews'  + label)                    
            fitPowerLaw    (file_user,  ax[2,1], 'IMDb, user reviews'    + label)
            fitPowerLaw    (file_gros,  ax[2,0], 'IMDb, gross revenue'   + label)

            savefig_nice(ax, 'Figs/1_impact_distributions/'+ mode_ +'_IMDB_fitted_impact_distros_' + label + '_full.png')
            
            #plt.tight_layout(pad=5, w_pad=5, h_pad=5)          
            #align_plot(ax)
            #plt.savefig(
            #plt.close()
       
                      
        ''' ---------------------------------------------- '''
        ''' MUSIC   '''
        
        genres = ['electro', 'pop']
         
        f, muax = plt.subplots(3, 3, figsize=(25, 25))
        st = f.suptitle( mode + "Music impact distributions", fontsize=title_font)
                    
        for i in range(3): 
            for j in range(3):
                genre_ind = i*3 + j
                if genre_ind < len(genres):                   
                    genre = genres[genre_ind]          
                    print 'PROCESSING -- ' + genre
                    file_music = FOLDER + '/1_impact_distributions/music_play_count_dist_' + genre + '.dat'
                    rating_cnt_fit   = fitPowerLaw(file_music, muax[i,j], genre + ' music, play counts')

        savefig_nice(ax, 'Figs/1_impact_distributions/'+ mode_ +'_Music_fitted_impact_distros_full.png')
        #plt.tight_layout(pad=5, w_pad=5, h_pad=5)          
        #align_plot(ax)
        #plt.savefig('Figs/1_impact_distributions/'+ mode_ +'_Music_fitted_impact_distros_full.png')
        #plt.close()
       
       
        ''' ---------------------------------------------- '''
        ''' BOOKS   '''      
             
        f, bax = plt.subplots(1, 3, figsize=(25, 12))
        st = f.suptitle( mode + "Books impact distributions", fontsize=title_font)

        print 'PROCESSING --  books' 
        
        book_avg = FOLDER + '/1_impact_distributions/book_average_rating_dist_authors.dat'
        book_cnt = FOLDER + '/1_impact_distributions/book_rating_count_dist_authors.dat'
        book_ed  = FOLDER + '/1_impact_distributions/book_edition_count_dist_authors.dat'            
              
        fitSkewedNormal(book_avg, bax[0], 'Goodreads, average rating'    )
        fitPowerLaw    (book_cnt, bax[1], 'Goodreads, rating count'      )  
        fitPowerLaw    (book_ed,  bax[2], 'Goodreads, number of editions')  

        savefig_nice(bax, 'Figs/1_impact_distributions/'+ mode_ +'_Books_fitted_impact_distros_full.png')        
        #plt.tight_layout(pad=5, w_pad=5, h_pad=5)          
        #align_plot(ax)
        #plt.savefig('Figs/1_impact_distributions/'+ mode_ +'_Books_fitted_impact_distros_full.png')
        #plt.close()       
           
    

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
    

def get_random_sample(mylist, N):

    return [ mylist[i] for i in sorted(random.sample(xrange(len(mylist)), N)) ]
    

def get_corr_segments(cnt, avg, limit = 10000):
            
    avg_1 = []
    cnt_1 = []
    
    avg_2 = []
    cnt_2 = []
    
    for i in range(len(cnt)):
        if cnt[i] < limit:
            avg_1.append(avg[i])
            cnt_1.append(cnt[i])
        else:
            avg_2.append(avg[i])
            cnt_2.append(cnt[i])        
        
    correl1 = str(round(stats.pearsonr(avg_1, cnt_1)[0], 3))
    correl2 = str(round(stats.pearsonr(avg_2, cnt_2)[0], 3))   

    return correl1, correl2


def plot_correlation_and_trend(xdata, ydata, ax, xlabel, ylabel, labelsize, Alpha, num_of_bins, label):

    ax.set_ylabel(xlabel, fontsize = labelsize)
    ax.set_xlabel(ylabel, fontsize = labelsize)
    ax.set_xscale('log')
    avg, cnt = get_rid_of_zeros(xdata, ydata)      
    avg_r, cnt_r = zip(*get_random_sample([(impacts[2][i],  impacts[1][i]) for i in range(len(impacts[2]))], 200000))  # avg, cnt

    corr1, corr2 = get_corr_segments(avg, cnt)

    ax.plot(avg_r, cnt_r, 'o', color = 'skyblue', alpha = Alpha, label =  ' $c_{<10^5}$ = ' + corr1 + ', $c_{>10^5}$ = ' + corr2 )
    xb_avg, pb_avg, pberr_avg = getLogBinnedDistribution(np.asarray(avg), np.asarray(cnt), num_of_bins)    
    
    corr1, corr2 = get_corr_segments(xb_avg, pb_avg)
    ax.errorbar(xb_avg, pb_avg, yerr = pberr_avg, fmt = 'o-',  color = 'midnightblue',  capsize = 3, elinewidth=3, linewidth = 4)#, label =  ' $c_{<10^5}$ = ' + corr1 + ', $c_{>10^5}$ = ' + corr2 )               
     
    write_row('Viz/2_correlations/scatter_data' + xlabel + '_'+ylabel +'_'+label+'.dat', [str(avg_r[i]) + '\t' + str(cnt_r[i]) for i in range(len(avg_r))])   
    write_row('Viz/2_correlations/binned_trend' + xlabel + '_'+ylabel +'_'+label+'.dat', [str(xb_avg[i])+'\t' + str(pb_avg[i]) + '\t' + str(pberr_avg[i]) for i in range(len(pberr_avg))  ])  


def get_impact_correlations():

    num_of_bins = 12
    title_font  = 25 
    seaborn.set_style('white')   
    
    professions = [('director',     'royalblue'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]    
    
    
  
    for mode in ['', 'Normalized']:#[0:1]:
    
        mode_  = 'Original' if mode == '' else 'Normalized'    
        FOLDER = 'ProcessedData'+mode + 'Sample'      
        
        markers = 12
    
        for (label, color) in professions:#[0:1]:
            

            f, ax = plt.subplots(3, 3, figsize=(25, 25))
            st = f.suptitle("IMDb impact correlations - " + mode + ' - ' + label , fontsize=title_font)
            labelsize = 18
            Alpha = 0.1
            
                    
            impacts = zip(*[[float(aaa) if 'tt' not in aaa else aaa for aaa in line.strip().split('\t')] for line in open(FOLDER+'/7_multiple_impacts/film_multiple_impacts_'+label+'.dat')])
    
            plot_correlation_and_trend(impacts[2], impacts[1], ax[0,0], 'IMDb, Average rating', 'IMDb, Rating count', labelsize, Alpha, num_of_bins, label)
            plot_correlation_and_trend(impacts[2], impacts[1], ax[0,0], 'IMDb, Average rating', 'IMDb, Rating count', labelsize, Alpha, num_of_bins, label)
            plot_correlation_and_trend(impacts[1], impacts[3], ax[0,1], 'IMDb,  metascore',     'IMDb, Average rating', labelsize, Alpha, num_of_bins, label)
            plot_correlation_and_trend(impacts[4], impacts[5], ax[1,1], 'IMDb, #critic review', 'IMDb, #user review', labelsize, Alpha, num_of_bins, label)           
            plot_correlation_and_trend(impacts[2], impacts[3], ax[1,0], 'IMDb, Rating count',   'IMDb, Metascore', labelsize, Alpha, num_of_bins, label)            
            plot_correlation_and_trend(impacts[2], impacts[4], ax[1,2], 'IMDb, Rating count',   'IMDb, #critic review', labelsize, Alpha, num_of_bins, label)            
            plot_correlation_and_trend(impacts[2], impacts[5], ax[0,2], 'IMDb, Rating count',   'IMDb, #user review', labelsize, Alpha, num_of_bins, label)
            plot_correlation_and_trend(impacts[2], impacts[6], ax[2,0], 'IMDb, Rating count',   'IMDb, Gross revenue (USD)', labelsize, Alpha, num_of_bins, label)
                        
            ax[0,1].set_xscale('linear')
            ax[0,1].set_yscale('linear')
            ax[1,1].set_yscale('log')
            ax[1,2].set_yscale('log')
            ax[0,2].set_yscale('log')
            ax[2,0].set_yscale('log')
    
            
           
            bimpacts = zip(*[[abs(float(a.replace(',',''))) for a in line.strip().split('\t')[1:]] for line in open(FOLDER+'/7_multiple_impacts/book_multiple_impacts_authors.dat') if 'www' not in line ])

            plot_correlation_and_trend(bimpacts[1], bimpacts[0], ax[2,2], 'Goodreads, Average rating', 'Goodreads, rating counts', labelsize, Alpha, num_of_bins, 'books')
            plot_correlation_and_trend(bimpacts[1], bimpacts[2], ax[2,1], 'Goodreads, Number of editions', 'Goodreads, rating counts', labelsize, Alpha, num_of_bins, 'books')
            ax[2,1].set_yscale('log')
            

            align_plot(ax)
            plt.tight_layout(pad=5, w_pad=5, h_pad=5)
            plt.savefig('Figs/2_correlations/' + mode_ + 'correlations_'+ label +'.png')
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

    ax.set_title(title, fontsize = 25)
    

    ax.fill_between((bx_average_ratings_year[1:] + bx_average_ratings_year[:-1])/2, bp_average_ratings_year-bperr_average_ratings_year, bp_average_ratings_year+bperr_average_ratings_year, alpha = 0.2, color = color)



    if music:
        ax.errorbar(x_average_ratings_year, y_average_ratings_year, yerr=yerr_average_ratings_year, fmt=color + '-', alpha = 0.5, capsize = 3, elinewidth=1, linewidth = 2)
        ax.errorbar((bx_average_ratings_year[1:] + bx_average_ratings_year[:-1])/2, bp_average_ratings_year, yerr=bperr_average_ratings_year, fmt= 'o-', color = color, alpha = 0.9, capsize = 3, elinewidth=1, linewidth = 3, label = label)
    else:
        ax.errorbar((bx_average_ratings_year[1:] + bx_average_ratings_year[:-1])/2, bp_average_ratings_year, yerr=bperr_average_ratings_year, fmt= 'o-', color = color, alpha = 0.9, capsize = 3, elinewidth=1, linewidth = 3)#, label = label)
    
    ax.set_xlim([1880, 2020])


def savefig_nice(ax, figname):

    plt.tight_layout(pad=5, w_pad=5, h_pad=5)          
    align_plot(ax)
    plt.savefig(figname)



def get_inflation_curves():


    num_of_bins = 20
    title_font = 25 
    seaborn.set_style('white')   


    FOLDER = 'ProcessedData/ProcessedData_0_'
    YEAR_MIN = 1900

    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]    
    
    
    
    ''' ---------------------------------------------- '''
    ''' MOVIES   '''
    
    professions = ['_MERGED', 'director', 'producer', 'writer', 'composer', 'art-director']

    for label in professions:
        
        print 'PROCESSING -- ' + label
        f, ax = plt.subplots(3, 2, figsize=(23, 23))
        st = f.suptitle( "Film inflation curves, " + label, fontsize=title_font)


        file_meta_year  = FOLDER + '/3_inflation_curves/film_yearly_average_rating_dist_'+label+'.dat'
        avg_rating_year = np.asarray([line.strip() for line in open(file_meta_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
        plot_measure(avg_rating_year, 'Movie director, Metascores', num_of_bins, ax[0,0], 'royalblue', '')    

        file_meta_year  = FOLDER + '/3_inflation_curves/film_yearly_rating_count_dist_'+label+'.dat'
        rating_cnt_year = np.asarray([line.strip() for line in open(file_meta_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
        plot_measure(rating_cnt_year, 'Movie director, Rating count', num_of_bins, ax[0,1], 'royalblue', '')    

        file_meta_year  = FOLDER + '/3_inflation_curves/film_yearly_metascore_dist_'+label+'.dat'
        metascores_year = np.asarray([line.strip() for line in open(file_meta_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
        plot_measure(metascores_year, 'Movie director, Metascore', num_of_bins, ax[1,0], 'royalblue', '')    

        file_meta_year  = FOLDER + '/3_inflation_curves/film_yearly_critic_reviews_dist_'+label+'.dat'
        critic_reviews  = np.asarray([line.strip() for line in open(file_meta_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
        plot_measure(critic_reviews, 'Movie director, Critic reviews', num_of_bins, ax[1,1], 'royalblue', '')    

        file_meta_year = FOLDER + '/3_inflation_curves/film_yearly_user_reviews_dist_'+label+'.dat'
        user_reviews   = np.asarray([line.strip() for line in open(file_meta_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
        plot_measure(user_reviews, 'Movie director, User reviews', num_of_bins, ax[2,0], 'royalblue', '')    

        file_meta_year = FOLDER + '/3_inflation_curves/film_yearly_gross_revenue_dist_'+label+'.dat'
        user_reviews   = np.asarray([line.strip() for line in open(file_meta_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
        plot_measure(user_reviews, 'Movie director, Gross_revenue', num_of_bins, ax[2,1], 'royalblue', '')    
        
        savefig_nice(ax, 'Figs/2_inflation_curves/Film_inflation_curves'+label+'.png')
        plt.close()
      
    
    ''' ---------------------------------------------- '''
    ''' MUSIC   '''
    
    genres = ['electro', 'pop']
     
    f, muax = plt.subplots(3, 3, figsize=(25, 25))
    st = f.suptitle( "Music inflation curves", fontsize=title_font)
                
    for i in range(3): 
        for j in range(3):
            genre_ind = i*3 + j
            if genre_ind < len(genres):                   
                genre = genres[genre_ind]          
                print 'PROCESSING -- ' + genre
                file_meta_year  = FOLDER + '/3_inflation_curves/music_yearly_play_count_dist_'+genre+'.dat'
                play_cnt_year = np.asarray([line.strip() for line in open(file_meta_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
                plot_measure(play_cnt_year, genre + ', Play count', num_of_bins, muax[i,j], 'forestgreen', '')    

    savefig_nice(muax, 'Figs/2_inflation_curves/Music_inflation_curves.png')
    plt.close()
   
   
    ''' ---------------------------------------------- '''
    ''' BOOKS   '''      
   
    f, bax = plt.subplots(1, 3, figsize=(25, 12))
    st = f.suptitle( "Books inflation curves", fontsize=title_font)

    print 'PROCESSING --  books' 
    file_cnt_year  = FOLDER + '/3_inflation_curves/book_yearly_rating_count_dist_authors.dat'
    rating_cnt_book = np.asarray([line.strip() for line in open(file_cnt_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
    plot_measure(rating_cnt_book,   'Book, rating counts',  num_of_bins, bax[0], 'Firebrick', '')        

    file_cnt_year  = FOLDER + '/3_inflation_curves/book_yearly_average_rating_dist_authors.dat'
    rating_cnt_book = np.asarray([line.strip() for line in open(file_cnt_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
    plot_measure(rating_cnt_book,   'Book, average rating ',  num_of_bins, bax[1], 'Firebrick', '') 

    file_cnt_year  = FOLDER + '/3_inflation_curves/book_yearly_edition_count_dist_authors.dat'
    rating_cnt_book = np.asarray([line.strip() for line in open(file_cnt_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
    plot_measure(rating_cnt_book,   'Book, edition count',  num_of_bins, bax[2], 'Firebrick', '') 

    savefig_nice(bax, 'Figs/2_inflation_curves/Books_inflation_curves.png')
    plt.close()



''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''                       GET NUMBER OF PRODUCTS                   '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   


def get_num_per_year(impacts):

    
    years = {} 
    for i in impacts:
        field = i.split('\t')
        year  = int(round(float((field[0])), -1))
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


  

def get_length_plots():


    FOLDER = 'ProcessedData'#Sample'


    title_font  = 22 
    num_of_bins = 20
    seaborn.set_style('white')  
    f, ax = plt.subplots(1, 3, figsize=(25, 8))
    st = f.suptitle("Number of products over time", fontsize=title_font)
    


    
    ''' movie '''
    file_avg_year  =  FOLDER + '/3_inflation_curves/film_yearly_average_ratings_dist_director.dat'       
    average_ratings_year = np.asarray([line.strip() for line in open(file_avg_year)])
    x_average_ratings_year, y_average_ratings_year = get_num_per_year(average_ratings_year)  
    
    movies = pd.DataFrame({'year': x_average_ratings_year, 'count': y_average_ratings_year })
    seaborn.barplot(x = movies['year'], y = movies['count'], ax = ax[0],  palette="Blues_d", linewidth = 0.0)

    #ax[0].set_title('IMDb #movies', fontsize = 25)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('')
    

    print 'MOVIES', sum(y_average_ratings_year)
    

    ''' book '''
    file_avg_year_electro = FOLDER + '/3_inflation_curves/music_yearly_rating_counts_dist_electro.dat' 
    file_avg_year_pop     = FOLDER + '/3_inflation_curves/music_yearly_rating_counts_dist_pop.dat'   
   
    average_ratings_year_electro = [line.strip() for line in open(file_avg_year_electro)]
    average_ratings_year_pop     = [line.strip() for line in open(file_avg_year_pop)]   
    
    x_average_ratings_year_electro, y_average_ratings_year_electro = get_num_per_year(np.asarray(average_ratings_year_electro + average_ratings_year_pop))  
    
    books = pd.DataFrame({'year': x_average_ratings_year_electro, 'count': y_average_ratings_year_electro })
    seaborn.barplot(x = books['year'], y = books['count'], ax = ax[1],  palette="Reds_d", linewidth = 0.0)

    #ax[1].set_title('Discogs & LastFM  #tracks', fontsize = title_font)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')

    print 'SONGS', sum(y_average_ratings_year_electro)




    ''' book '''
    file_avg_year  =  FOLDER + '/3_inflation_curves/book_yearly_average_ratings_dist_authors.dat'       
    average_ratings_year = np.asarray([line.strip() for line in open(file_avg_year)])
    x_average_ratings_year, y_average_ratings_year = get_num_per_year(average_ratings_year)  
    
    books = pd.DataFrame({'year': x_average_ratings_year, 'count': y_average_ratings_year })
    seaborn.barplot(x = books['year'], y = books['count'], ax = ax[2],  palette="Greens_d", linewidth = 0.0)

    #ax[2].set_title('Goodreads  #books', fontsize = title_font)
    ax[2].set_xlabel('')
    ax[2].set_ylabel('')
    
    print 'BOOKS', sum(y_average_ratings_year)    
  
    for axax in ax:
        axax.set_yscale('log') 
        i = 0
        for tick in axax.get_xticklabels():
            i += 1
            tick.set_rotation(45)   
            if i % 2 == 0:
                tick.set_visible(False)

    align_plot(ax)
    plt.tight_layout(pad=5, w_pad=10, h_pad=20)
    plt.savefig('1Fig_num_of_products.png')
    plt.show()





''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''                       GET CAREER LENGTH DISTR                  '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''  



def fitPowerLawLogNormal(rand, ax, label):


    # histogram 
    ax.set_xscale('log')
    print 'Fitting lognormal...'
    x_rand, p_rand = getDistribution(rand)    
    counts, bins, bars = ax.hist(rand, normed = True, bins = 10 ** np.linspace(np.log10(min(x_rand)), np.log10(max(x_rand)), 15), log=True,alpha=0.0)
    ax.plot((bins[1:] + bins[:-1])/2, counts, 's-', color = 'royalblue', alpha = 0.6, markersize = 12, linewidth = 2, label = 'Data')


    # get the lognormal
    param = stats.lognorm.fit(rand)
    pdf_fitted = stats.lognorm.pdf(x_rand, param[0], loc=param[1], scale=param[2])#
    mu =  np.log(param[2])
    sigma = param[0]
    sk_results_norm = stats.kstest(np.asarray(pdf_fitted), lambda x: stats.lognorm.cdf(x_rand, param[0], loc=param[1], scale=param[2]))   # stats.ks_2samp(np.cumsum(p_rand), np.cumsu
    ax.plot(x_rand,pdf_fitted,'k-', linewidth = 4, label = 'Lognormal fit, $\\mu$=' + str(round(mu,2)) + '\n$\\sigma$=' + str(round(sigma, 2)) + ', $D$='+str(round(sk_results_norm[0], 2)))

         
    ax.set_ylim([ min(counts), 1.1])
    ax.set_xlim([ min(x_rand),  max(bins)])
    ax.set_title(label, fontsize = 18) 



def get_career_length():


    title_font  = 25 
    num_of_bins = 20
    seaborn.set_style('white')  

    f, ax = plt.subplots(1, 3, figsize=(25, 8))
    st = f.suptitle("Career length distributions", fontsize=title_font)    


    ''' movie '''
    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y'),]    
       

    for (label, color) in professions[0:1]:
        career_length  = [float(line.strip()) for line in open('ProcessedData/5_career_length/film_career_length_' + label + '.dat')]
        fitPowerLaw([ c for c in career_length if c > 1], ax[0], label)
        


    ''' music '''
    professions = [('pop',     'k'), 
                   ('electro', 'b')]    
       
    for (label, color) in professions[1:2]:
        career_length = [float(line.strip()) for line in open('ProcessedData/5_career_length/music_career_length_' + label + '.dat')]   
        fitPowerLaw([ c for c in career_length if c > 1], ax[1], label)
        
        
    ''' books '''
    career_length_book = [float(line.strip()) for line in open('ProcessedData/5_career_length/book_career_length_authors.dat')]   
    fitPowerLaw([ c for c in career_length_book if c > 0], ax[2], label)
        

    

    ax[0].set_title('Movie director careers', fontsize = 18)
    ax[1].set_title('DJ careers', fontsize = 18) 
    ax[2].set_title('Author careers', fontsize = 18)  

    align_plot(ax)
    plt.tight_layout(pad=5, w_pad=8, h_pad=20)
    #plt.savefig('2Fig_career_length.png')
    #plt.close()
    plt.show()







''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''           GET THE N*/N STUFF OF ALL SUCCESS MEASURES           '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''   


def plot_red_lines(ax, x):

    #plt.grid() 
    if len(ax.shape)> 1:
        for i in range(len(ax)):
            for j in range(len(ax[0])):     
                ax[i,j].set_xlim(-0.05, 1.05)
                ax[i,j].set_ylim(-0.05, 1.05)
                yyy = [1 - y0 for y0 in x]
                ax[i,j].plot(x, yyy, '-', linewidth=2, alpha = 0.9, color = 'r') 
                ax[i,j].set_xlabel('$N^{*}/N$', fontsize=17)
                ax[i,j].set_ylabel( r'$P( \geq  N^{*}/N)$' , fontsize=17)
                
    else:
        for j in range(len(ax)):     
            ax[j].set_xlim(-0.05, 1.05)
            ax[j].set_ylim(-0.05, 1.05)
            yyy = [1 - y0 for y0 in x]
            ax[j].plot(x, yyy, '-', linewidth=2, alpha = 0.9, color = 'r') 
            ax[j].set_xlabel('$N^{*}/N$', fontsize=17)
            ax[j].set_ylabel( r'$P( \geq  N^{*}/N)$' , fontsize=17)



def plot_ccdf(file_avg_all, num_of_bins, ax, color, label, Nmin, title, marker):


    x_Nstar_avg_all, p_Nstar_avg_all, len_career, r_square, numInd = parse_N_star_N_data(file_avg_all, Nmin)
    

    #if 'orig' in label:
    ax.set_title(str(numInd) + ' ' + title, fontsize = 19)   
        
    bx_average_ratings, bp_average_ratings, bperr_average_ratings = getBinnedDistribution(np.asarray(x_Nstar_avg_all), np.asarray(p_Nstar_avg_all), num_of_bins)
 
 
    binss = (bx_average_ratings[1:] + bx_average_ratings[:-1])/2
    dataf = open('Viz/NN/NNstar_data_' + title + '.dat', 'w')
    for i in range(len(binss)):
        dataf.write(str(binss[i]) + '\t' + str(bp_average_ratings[i]) + '\t' + str(bperr_average_ratings[i] )+ '\n')
    dataf.close()
         
         
    ax.fill_between((bx_average_ratings[1:] + bx_average_ratings[:-1])/2, bp_average_ratings-bperr_average_ratings, bp_average_ratings+bperr_average_ratings, alpha = 0.2, color = color)         
         
         
    #ax.plot(x_Nstar_avg_all, p_Nstar_avg_all, color = color,  marker = 'o', linewidth = 0, markersize = 5, alpha= 0.5, label = label + ', ' + str(len_career) + ' $R^2=$' + str(round(r_square, 4)),)  
    ax.errorbar((bx_average_ratings[1:] + bx_average_ratings[:-1])/2, bp_average_ratings, yerr=bperr_average_ratings, fmt=color + '-', linewidth = 2,  markersize = 0,marker = marker, alpha = 0.9) # label = label + ' $R^2 = $' + str(round(r_square, 5)), 
            
    return r_square            



def parse_N_star_N_data(filename, cutoff_N1):

    try:
        N_star_N = []
        
        for line in open(filename):
        

            fields   = line.strip().split('\t')    
            best_id  = float(fields[0])
            career_N = float(fields[1])
            if career_N >= cutoff_N1:
                N_star_N.append(best_id/career_N)


        x_stat = np.linspace(0,1, len(N_star_N))
        maxy = max(N_star_N)
        y_stat = np.asarray([1-yy/maxy for yy in sorted(N_star_N)])      
    
        slope, intercept, r_square, p_value, std_err = stats.linregress(x_stat,[1 - aaa for aaa in y_stat])
  
        numInd = len(N_star_N)
  
        return x_stat, y_stat, len(N_star_N), r_square, numInd

    except ValueError:
    
        return [], [], 0, 0, 0
        



def get_r_test():


    ''' ---------------------------------------------- '''
    '''      MOVIE YO                                  '''
    
    '''
       -  normalized
    -  N > 15
    -  min rating 0
    -  min rating 15
    '''
    
    
    
    num_of_bins = 15
    title_font  = 25 
    Nmin = 15
    seaborn.set_style('white')   

    xxx = np.arange(0,1, 1.0/20)

    
    
    min_rating = 0    
    folder = 'ProcessedData/ProcessedData_' + str(min_rating)
    
    
    
    print folder, '\n\n'


    professions = [('_MERGED',      'b',  'o'),
                   ('director',     'k',  'o'), 
                   ('producer',     'b',  's'),
                   ('writer'  ,     'r',  '^'),
                   ('composer',     'g',  '8'),
                   ('art-director', 'y',  'x')]

    ''' 
    for (label, color, marker) in professions:

        
        f, ax = plt.subplots(3, 2, figsize=(23, 23))
        st = f.suptitle( "Relative rank of the best, " + label, fontsize=title_font)

        #for (mode, colorm, markerm) in [('', 'k', 'o'), ('_Normalized', 'b', '^'), ('_NormalizedRandomized', 'r', 's')][1:]:
           

       
        file_avg_all  = folder +  '_Normalized' + '/4_NN_rank_N/film_best_product_NN_ranks_all_average_rating_' + label + '.dat'
        file_cnt_all  = folder +  '_Normalized' + '/4_NN_rank_N/film_best_product_NN_ranks_all_rating_count_'   + label + '.dat'
        file_mets_all = folder +  '_Normalized' + '/4_NN_rank_N/film_best_product_NN_ranks_all_metascore_'      + label + '.dat'
        file_crit_all = folder +  '_Normalized' + '/4_NN_rank_N/film_best_product_NN_ranks_all_critic_reviews_' + label + '.dat'
        file_user_all = folder +  '_Normalized' + '/4_NN_rank_N/film_best_product_NN_ranks_all_user_reviews_'   + label + '.dat'
        file_gross    = folder +  '_Normalized' + '/4_NN_rank_N/film_best_product_NN_ranks_all_gross_revenue_'  + label + '.dat'          
      
 
            
        colorm = 'b'
        mode = ''           
        markerm = 'o'                                         
                                                                                                   
        r_square_avg   = plot_ccdf(file_avg_all,  num_of_bins, ax[0,0], colorm, mode, Nmin, 'Individual, IMDb average ratings' , markerm)
        r_square_cnt   = plot_ccdf(file_cnt_all,  num_of_bins, ax[0,1], colorm, mode, Nmin, 'Individual, IMDb rating counts'   , markerm)
        r_square_meta  = plot_ccdf(file_mets_all, num_of_bins, ax[1,0], colorm, mode, Nmin, 'Individual, IMDb metascores'      , markerm)
        r_square_crit  = plot_ccdf(file_crit_all, num_of_bins, ax[1,1], colorm, mode, Nmin, 'Individual, IMDb #critic reviews' , markerm)
        r_square_user  = plot_ccdf(file_user_all, num_of_bins, ax[2,0], colorm, mode, Nmin, 'Individual, IMDb #user reviews'   , markerm)
        r_square_gross = plot_ccdf(file_gross,    num_of_bins, ax[2,1], colorm, mode, Nmin, 'Individual, IMDb gross revenue '  , markerm)   
            

        plot_red_lines(ax, xxx)        
        #plt.show()
        savefig_nice(ax, 'Figs/3_best_rank_distr/Film_NN_stat_' + str(min_rating) + '_' + label + '.png')          

    '''
    
            
            
    ''' ---------------------------------------------- '''
    ''' MUSIC   '''
    
     
    genres  = ['_MERGED', 'electro', 'pop']
    markerm = 'o'
     
    f, muax = plt.subplots(3, 3, figsize=(25, 25))
    st = f.suptitle( "Music inflation curves", fontsize=title_font)
                
    for i in range(3): 
        for j in range(3):
            genre_ind = i*3 + j
            if genre_ind < len(genres):                   
                genre = genres[genre_ind]          
                print 'PROCESSING -- ' + genre
                file_music   = folder + '_Normalized' + '/4_NN_rank_N/music_best_product_NN_ranks_all_play_count_'+genre+'.dat'
                r_square_cnt = plot_ccdf(file_music,  num_of_bins, muax[i,j], 'g', genre, Nmin, 'Play count, ' + genre, markerm)

    plot_red_lines(muax, xxx)
    savefig_nice(muax, 'Figs/3_best_rank_distr/Music_NN_stat_' + str(min_rating) + '_' + genre + '.png') 
    
    

    ''' ---------------------------------------------- '''
    ''' BOOKS   '''      
    
    
    f, bax = plt.subplots(1, 3, figsize=(25, 12))
    st = f.suptitle( "Books inflation curves", fontsize=title_font)
    markerm = 'o'

    print 'PROCESSING --  books', folder
    
     
    file_avg_all  = folder + '_Normalized' + '/4_NN_rank_N/book_best_product_NN_ranks_all_average_rating_' + 'authors' + '.dat'
    file_cnt_all  = folder + '_Normalized' + '/4_NN_rank_N/book_best_product_NN_ranks_all_rating_count_'   + 'authors' + '.dat'
    file_mets_all = folder + '_Normalized' + '/4_NN_rank_N/book_best_product_NN_ranks_all_edition_count_'  + 'authors' + '.dat'

    r_square_avg  = plot_ccdf(file_avg_all,  num_of_bins, bax[0], 'r', '', Nmin, 'Individual, Goodreads average ratings' , markerm)
    r_square_cnt  = plot_ccdf(file_cnt_all,  num_of_bins, bax[1], 'r', '', Nmin, 'Individual, Goodreads rating count'    , markerm)
    r_square_meta = plot_ccdf(file_mets_all, num_of_bins, bax[2], 'r', '', Nmin, 'Individual, Goodreads edition count'   , markerm)

    plot_red_lines(bax, xxx)
    savefig_nice(bax, 'Figs/3_best_rank_distr/Book_NN_stat_' + str(min_rating) + 'png') 
   
    




















''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                                  '''   
'''                                 DO THE R MODEL OLD                               '''
'''                                                                                  '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  ''' 




def get_r_model_curves(data_file, max_data_file, ax, label, num_of_bins, title, xlabel, ylabel, log = False):


    ax.set_title(title,   fontsize = 19)
    ax.set_xlabel(xlabel, fontsize = 17)
    ax.set_ylabel(ylabel, fontsize = 17)


    data = [float(line.strip()) for line in open(data_file) if 'nan' not in line]
    (data_max, career_len) = zip(*[[float(num) for num in line.strip().split('\t')] for line in open(max_data_file)  if 'nan' not in line])
    ax.plot(career_len, data_max, marker = 'o', color = 'lightgrey', alpha = 0.15,linewidth = 0)
    
    
    fdata = open('Viz/rmodel_data_' + label + '.dat', 'w')
    for i in range(len(career_len)):
        fdata.write(str(career_len[i]) + '\t' + str(data_max[i]) + '\n')
    fdata.close
    
    
    
    print max(career_len), len(data)
    
    career_max = []
    career_max_dict = {}
    
    
    for i in range(1):

        data_new = data[:]
        random.shuffle(data_new)
        i= 1
        for leng in career_len:
            #career_max.append(max( data_new[0:int(leng)]))
        
            i += 1

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
        
        
        
        fdata = open('Viz/rmodel_bin_data_' + label + '.dat', 'w')
        xb_data = (xb_data[1:] + xb_data[:-1])/2
        xb_gen  = (xb_gen[1:]  + xb_gen[:-1])/2
        for i in range(len(xb_data)):
            fdata.write(str(xb_data[i]) + '\t' + str(pb_data[i]) + '\t' + str(pberr_data[i]) + '\n')
        fdata.close
        
        fdata = open('Viz/rmodel_bin_gen_' + label + '.dat', 'w')
        xb_data = (xb_data[1:] + xb_data[:-1])/2
        xb_gen  = (xb_gen[1:]  + xb_gen[:-1])/2
        for i in range(len(xb_gen)):
            fdata.write(str(xb_gen[i]) + '\t' + str(pb_gen[i]) + '\t' + str(pberr_gen[i]) + '\n')
        fdata.close
        
    else:
        ax.set_xscale('log')
        ax.set_yscale('log')
        xb_data, pb_data, pberr_data = getLogBinnedDistribution(np.asarray(career_len),  np.asarray(data_max), num_of_bins)         
        xb_gen, pb_gen, pberr_gen    = getLogBinnedDistribution(np.asarray(sorted_len),  np.asarray(career_max), num_of_bins)
        ax.errorbar(xb_data, pb_data, yerr = pberr_data, fmt = 'o-', color = 'grey', label = 'data', alpha = 0.9)
        ax.errorbar(xb_gen,  pb_gen,  yerr = pberr_gen, fmt = '-', color = 'r', label = 'R-model', alpha = 0.9)        
    
        fdata = open('Viz/rmodel_bin_data_' + label + '.dat', 'w')
        for i in range(len(xb_data)):
            fdata.write(str(xb_data[i]) + '\t' + str(pb_data[i]) + '\t' + str(pberr_data[i]) + '\n')
        fdata.close
        
        fdata = open('Viz/rmodel_bin_gen_' + label + '.dat', 'w')
        for i in range(len(xb_gen)):
            fdata.write(str(xb_gen[i]) + '\t' + str(pb_gen[i]) + '\t' + str(pberr_gen[i]) + '\n')
        fdata.close




def do_the_r_model():


    title_font  = 25 
    num_of_bins = 8
    seaborn.set_style('white')  

    
    min_rating = 0       
    folder = 'ProcessedData/ProcessedData_' + str(min_rating)
    mode   = '_Normalized'
    field  = 'film'


    ''' ---------------------------------------------- '''
    ''' FILM   '''
    '''
    professions = ['_MERGED',      
               'director',     
               'producer',     
               'writer'  ,     
               'composer',     
                   'art-director']
     
    for label in professions[1:3]:

        f, ax = plt.subplots(3, 2, figsize=(23, 23))
        st = f.suptitle( "Film - R - model vs data (movie directors and DJs), " + label, fontsize=title_font)

        file_avg   = folder +  mode + '/1_impact_distributions/film_average_rating_dist_'  + label + '.dat'
        file_cnt   = folder +  mode + '/1_impact_distributions/film_rating_count_dist_'    + label + '.dat'
        file_meta  = folder +  mode + '/1_impact_distributions/film_metascore_dist_'       + label + '.dat'
        file_crit  = folder +  mode + '/1_impact_distributions/film_critic_reviews_dist_'  + label + '.dat'
        file_user  = folder +  mode + '/1_impact_distributions/film_user_reviews_dist_'    + label + '.dat'
        file_gross = folder +  mode + '/1_impact_distributions/film_gross_revenue_dist_'   + label + '.dat'        
        
        max_avg_rat   = folder + mode + '/7_career_length_max_impact/film_career_length_max_average_rating_' + label + '.dat'
        max_rat_cnt   = folder + mode + '/7_career_length_max_impact/film_career_length_max_rating_count_'    + label + '.dat'
        max_meta      = folder + mode + '/7_career_length_max_impact/film_career_length_max_metascore_'       + label + '.dat'     
        max_crit_rev  = folder + mode + '/7_career_length_max_impact/film_career_length_max_critic_reviews_'    + label + '.dat'
        max_user_rev  = folder + mode + '/7_career_length_max_impact/film_career_length_max_user_reviews_'    + label + '.dat'
        max_gross_rev = folder + mode + '/7_career_length_max_impact/film_career_length_max_gross_revenue_'    + label + '.dat'




        get_r_model_curves(file_avg,  max_avg_rat,  ax[0,0], label, num_of_bins, 'Average rating vs career length' , 'Career length', 'Average rating' )
        #get_r_model_curves(file_meta, max_meta,     ax[0,1], label, num_of_bins, 'Metascore vs career length'      , 'Career length', 'Metascore'           )
        get_r_model_curves(file_cnt,  max_rat_cnt,  ax[1,0], label, num_of_bins, 'Rating count vs career length'   , 'Career length', 'Rating count'  , True)            
        get_r_model_curves(file_crit, max_crit_rev, ax[1,1], label, num_of_bins, 'Critic reviews vs career length' , 'Career length', 'Critic reviews', True)    
        get_r_model_curves(file_user, max_user_rev, ax[2,0], label, num_of_bins, 'User reviews vs career length'   , 'Career length', 'User reviews'  , True)            
        get_r_model_curves(file_user, max_user_rev, ax[2,1], label, num_of_bins, 'Gross revenue vs career length'  , 'Career length', 'Gross revenue' , True)            

        #plt.show()
        savefig_nice(ax, 'Figs/4_r_model/Film_R_model_test_' + str(min_rating) + '_' + label + '.png')        
   '''
     
     
     
    ''' ---------------------------------------------- '''
    ''' MUSIC   '''
     
    '''       
    genres  = ['electro', 'pop']#['_MERGED', 'electro', 'pop']
    markerm = 'o'
     
    f, muax = plt.subplots(3, 3, figsize=(25, 25))
    st = f.suptitle( "Music -R - model vs data", fontsize=title_font)
                
    for i in range(3): 
        for j in range(3):
            genre_ind = i*3 + j
            if genre_ind < len(genres):                   
                genre = genres[genre_ind]          
                print 'PROCESSING -- ' + genre
                file_music = folder + mode + '/1_impact_distributions/music_play_count_dist_' + genre + '.dat'
                max_music  = folder + mode + '/7_career_length_max_impact/music_career_length_max_play_count_' + genre + '.dat'           
                get_r_model_curves(file_music, max_music, muax[i,j], genre, num_of_bins, 'Rating count vs career length', 'Career length', 'Rating count', True)

    plt.show()
    savefig_nice(muax, 'Figs/4_r_model/Music_R_model_test_' + str(min_rating) + '_' + genre + '.png')       
    ''' 
     
     

    ''' ---------------------------------------------- '''
    ''' BOOKS   '''  
       
    f, bax = plt.subplots(1, 3, figsize=(25, 12))
    st = f.suptitle( "Books, R - model vs data", fontsize=title_font) 
        
    file_book = folder +  mode + '/1_impact_distributions/book_average_rating_dist_authors.dat'
    max_book  = folder +  mode + '/7_career_length_max_impact/book_career_length_max_average_rating_authors.dat' 
    get_r_model_curves(file_book, max_book, bax[0], 'book', num_of_bins, 'Average rating vs career length', 'Career length', 'Average rating', True)  
    
    
    file_book = folder +  mode + '/1_impact_distributions/book_rating_count_dist_authors.dat'
    max_book  = folder +  mode + '/7_career_length_max_impact/book_career_length_max_rating_count_authors.dat' 
    get_r_model_curves(file_book, max_book, bax[1], 'book', num_of_bins, 'Rating_count vs career length', 'Career length', 'Rating_count', True)
    
    
    file_book = folder +  mode + '/1_impact_distributions/book_edition_count_dist_authors.dat'
    max_book  = folder +  mode + '/7_career_length_max_impact/book_career_length_max_edition_count_authors.dat' 
    get_r_model_curves(file_book, max_book, bax[2], 'book', num_of_bins, 'Average rating vs career length', 'Career length', 'Edition count', True)         
  
  
  
    plt.show()
    savefig_nice(bax, 'Figs/4_r_model/Book_R_model_test_' + str(min_rating) + 'png')   
           

    
      
       


''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                                  '''   
'''                               WHEN COMES THE BEST?                               '''
'''                                                                                  '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  ''' 


def get_time_distr(filename, ax, num_of_bins, label):

    Alpha = 0.5
    data = np.asarray([round(float(line.strip())) for line in open(filename) if float(line.strip()) > 0]) 
    x_data, p_data = getDistribution(data)
    xb_data, pb_data, pberr_data = getBinnedDistribution(np.asarray(x_data), np.asarray(p_data), num_of_bins)         





    ax.plot(x_data, p_data, 'o', linewidth = 3,  alpha = Alpha )
    ax.errorbar((xb_data[1:] + xb_data[:-1])/2, pb_data, yerr = pberr_data, fmt = 'o-', color = 'r', label = label, alpha = 0.9 )
    
    
    
    write_row('Viz/3_best_timing/normal_fit_best_times'  + label + '.dat', [str(x_data[i])  + '\t' + str(p_data[i])  for i in range(len(x_data))])   
    write_row('Viz/3_best_timing/normal_hist_best_times' + label + '.dat', [str(xb_data[i]) + '\t' + str(pb_data[i])  + '\t' + str(pberr_data[i])  for i in range(len(pberr_data))])       

 
    
   
    
    

def get_best_times():



    title_font  = 25 
    num_of_bins = 12
    seaborn.set_style('white')  

 
    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y'),]    
       



    FOLDER = 'ProcessedDataNormalized' + '/11_time_of_the_best' #Normalized' # mode# + 'Sample' 
    field  = 'film'
     
   
   
    ''' ---------------------------------------------- '''
    ''' MOVIES   '''  
    
    for (label, color) in professions:

        f, ax = plt.subplots(2, 3, figsize=(25, 15))
        st = f.suptitle( 'MOVIES - Time distribution of the best product ($P(t^*)$)', fontsize=title_font)

        #get_time_distr(FOLDER + '/film_time_of_the_best_avg_rating_' + label + '.dat', ax[0,0], num_of_bins, 'IMDb avg rating, ' + label)
        #get_time_distr(FOLDER + '/film_time_of_the_best_rating_cnt_' + label + '.dat', ax[0,1], num_of_bins, 'IMDb rating cnt, ' + label)
        #get_time_distr(FOLDER + '/film_time_of_the_best_metascore_'  + label + '.dat', ax[0,2], num_of_bins, 'IMDb metascore, '  + label)
        #get_time_distr(FOLDER + '/film_time_of_the_best_critic_rev_' + label + '.dat', ax[1,0], num_of_bins, 'IMDb critic rev, ' + label)
        #get_time_distr(FOLDER + '/film_time_of_the_best_user_rev_'   + label + '.dat', ax[1,1], num_of_bins, 'IMDb user rev, '   + label)
        #get_time_distr(FOLDER + '/film_time_of_the_best_gross_'      + label + '.dat', ax[1,2], num_of_bins, 'IMDb gross, '      + label)               
     
        align_plot(ax)   
        plt.savefig('Figs/3_best_time_distribution/IMDb_best_times_distr_' + label + '.png')      
        #plt.show()
        plt.close()
    

    
    ''' ---------------------------------------------- '''
    ''' MUSIC   '''
   
    genres = ['electro', 'pop']
         
    f, ax = plt.subplots(3, 3, figsize=(25, 25))
    st = f.suptitle( 'MUSIC - Time distribution of the best product ($P(t^*)$)', fontsize=title_font)
                 
    for i in range(3): 
        for j in range(3):
            genre_ind = i*3 + j
            if genre_ind < len(genres):                   
                genre = genres[genre_ind]               
                get_time_distr(FOLDER + '/music_time_of_the_best_rating_cnt_' + genre + '.dat', ax[i,j], num_of_bins, 'Music play count, ' + genre)

    plt.tight_layout(pad=5, w_pad=5, h_pad=5)          
    align_plot(ax)   
    plt.savefig('Figs/3_best_time_distribution/Music_best_times_distr.png')      
    #plt.show()
    plt.close()

    

    ''' ---------------------------------------------- '''
    ''' BOOKS   '''
    
    f, ax = plt.subplots(1, 3, figsize=(25, 8))
    st = f.suptitle( 'BOOKS - Time distribution of the best product ($P(t^*)$)', fontsize=title_font)
 
    num_of_bins = 21
    
    get_time_distr(FOLDER + '/book_time_of_the_best_avg_rating_authors.dat', ax[0], num_of_bins, 'Book avg rating, '   + genre) 
    #get_time_distr(FOLDER + '/book_time_of_the_best_rating_cnt_authors.dat', ax[1], num_of_bins, 'Book rating count, ' + genre) 
    #get_time_distr(FOLDER + '/book_time_of_the_best_metascore_authors.dat',  ax[2], num_of_bins, 'Book #editions, '    + genre)
                              
    plt.tight_layout(pad=5, w_pad=5, h_pad=5)          
    align_plot(ax)   
    plt.savefig('Figs/3_best_time_distribution/Books_best_times_distr.png')      
    plt.show()
    #plt.close() 
 
    
    
    
    
    
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                                  '''   
'''                               p - mu_p distribution                              '''
'''                                                                                  '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''    

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

    

   
    title_font  = 25 
    num_of_bins = 8
    seaborn.set_style('white')  

    
    min_rating = 0       
    folder = 'ProcessedData/ProcessedData_' + str(min_rating)
    mode   = '_Normalized'
    field  = 'film'


    ''' ---------------------------------------------- '''
    ''' FILM   '''
 
    professions = ['_MERGED',      
                   'director',     
                   'producer',     
                   'writer'  ,     
                   'composer',     
                   'art-director']
     
    for label in professions[1:]:

        f, ax = plt.subplots(3, 2, figsize=(23, 23))
        st = f.suptitle( mode + "  $\log(p_{\\alpha}) + \mu_p$ distributions ", fontsize=title_font)
    
  
        file_avg   = folder + mode + '/9_p_without_avg/' + field + '_p_without_mean_average_rating_'     + label + '.dat'
        file_cnt   = folder + mode + '/9_p_without_avg/' + field + '_p_without_mean_rating_count_'       + label + '.dat'        
        file_mets  = folder + mode + '/9_p_without_avg/' + field + '_p_without_mean_metascore_'          + label + '.dat'   
        file_crit  = folder + mode + '/9_p_without_avg/' + field + '_p_without_mean_critic_reviews_'     + label + '.dat'   
        file_user  = folder + mode + '/9_p_without_avg/' + field + '_p_without_mean_user_reviews_'       + label + '.dat' 
        file_gross = folder + mode + '/9_p_without_avg/' + field + '_p_without_mean_user_gross_revenue_' + label + '.dat' 

        average_ratings = np.asarray([float(line.strip()) for line in open(file_avg)  if float(line.strip()) != 0  ])
        rating_counts   = np.asarray([float(line.strip()) for line in open(file_cnt)  if float(line.strip()) != 0  ])
        metascores      = np.asarray([float(line.strip()) for line in open(file_mets) if float(line.strip()) != 0  ])
        critic_review   = np.asarray([float(line.strip()) for line in open(file_crit) if float(line.strip()) != 0  ])
        user_review     = np.asarray([float(line.strip()) for line in open(file_user) if float(line.strip()) != 0  ])
        gross_revenue   = np.asarray([float(line.strip()) for line in open(file_user) if float(line.strip()) != 0  ])

        fitNormal(average_ratings, ax[0,0], label + ', avg rating')
        fitNormal(rating_counts,   ax[0,1], label + ', rating count')
  #      fitNormal(metascores,      ax[1,0], label + ', metascore')
   #     fitNormal(critic_review,   ax[1,1], label + ', critic reviews')
    #    fitNormal(user_review,     ax[2,0], label + ', user reviews')
     #   fitNormal(gross_revenue,   ax[2,1], label + ', gross revenue')            
 

        #plt.show()
        savefig_nice(ax, 'Figs/5_p_nomean_distr/Film_p_no_mean_distr_' + str(min_rating) + '_' + label + '.png')        
 
      

    ''' ---------------------------------------------- '''
    ''' MUSIC   '''
     
           
    genres  = ['electro', 'pop']#['_MERGED', 'electro', 'pop']
    markerm = 'o'
     
    f, muax = plt.subplots(3, 3, figsize=(25, 25))
    st = f.suptitle( "Music -R - model vs data", fontsize=title_font)
                
    for i in range(3): 
        for j in range(3):
            genre_ind = i*3 + j
            if genre_ind < len(genres):                   
                genre = genres[genre_ind]          
                print 'PROCESSING -- ' + genre
                file_cnt = folder + mode + '/9_p_without_avg/music_p_without_mean_play_count_'   + genre + '.dat'        
                rating_counts = np.asarray([float(line.strip()) for line in open(file_cnt)  if float(line.strip()) != 0  ])
      #          fitNormal(rating_counts, muax[i,j], genre + ', play count')

    #plt.show()
    savefig_nice(muax, 'Figs/5_p_nomean_distr/Music_p_no_mean_distr_' + str(min_rating) + '_' + genre + '.png')       
     
     
    
 
    ''' ---------------------------------------------- '''
    ''' BOOKS   '''  
       
    f, bax = plt.subplots(1, 3, figsize=(25, 12))
    st = f.suptitle( "Books, R - model vs data", fontsize=title_font) 
 
    file_avg = folder + mode + '/9_p_without_avg/' + 'book_p_without_mean_average_rating_authors.dat'
    file_cnt = folder + mode + '/9_p_without_avg/' + 'book_p_without_mean_rating_count_authors.dat'        
    file_ed  = folder + mode + '/9_p_without_avg/' + 'book_p_without_mean_edition_count_authors.dat'   

#    average_ratings = np.asarray([float(line.strip()) for line in open(file_avg)  if float(line.strip()) != 0  ])
#    rating_counts   = np.asarray([float(line.strip()) for line in open(file_cnt)  if float(line.strip()) != 0  ])
#    edition_counts  = np.asarray([float(line.strip()) for line in open(file_ed)   if float(line.strip()) != 0  ])

    fitNormal(average_ratings, bax[0], label + ', avg rating')
    fitNormal(rating_counts,   bax[1], label + ', rating count')
    fitNormal(edition_counts,  bax[2], label + ', edition count')


      
  
    #plt.show()
    savefig_nice(bax, 'Figs/5_p_nomean_distr/Book_p_no_mean_distr_' + str(min_rating) + 'png')   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':         


    if sys.argv[1] == '1':
        get_imapct_distr()
        
    elif sys.argv[1] == '2':
        get_impact_correlations()
        
    elif sys.argv[1] == '3': 
        get_length_plots()
        
    elif sys.argv[1] == '4':
        get_career_length()

    elif sys.argv[1] == '5':
        get_inflation_curves()
    
    elif sys.argv[1] == '6':
        get_r_test()   

    elif sys.argv[1] == '9':
        do_the_r_model()
        
    elif sys.argv[1] == '11':
        get_best_times()
        
    elif sys.argv[1] == '12':
        get_p_without_avg()
  
    
    
    
    
    
    
    
