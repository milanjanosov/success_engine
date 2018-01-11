execfile("0_imports.py")


def plot_red_lines(ax, x):

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
    
    ax.set_title(str(numInd) + ' ' + title, fontsize = 19)   
        
    bx_average_ratings, bp_average_ratings, bperr_average_ratings = binning.getPercentileBinnedDistribution(np.asarray(x_Nstar_avg_all), np.asarray(p_Nstar_avg_all), num_of_bins)
    
    bx_average_ratings    = np.asarray(bx_average_ratings)  #(bx_average_ratings[1:] + bx_average_ratings[:-1])/2  
    bp_average_ratings    = np.asarray(bp_average_ratings)
    bperr_average_ratings = np.asarray(bperr_average_ratings) 


    D = stats.kstest(bp_average_ratings, 'uniform' )[0]


    ax.fill_between(bx_average_ratings, bp_average_ratings-bperr_average_ratings, bp_average_ratings+bperr_average_ratings, alpha = 0.2, color = color)                  
    ax.errorbar(bx_average_ratings, bp_average_ratings, yerr=bperr_average_ratings, fmt=color + '-', linewidth = 2,  markersize = 0,marker = marker, alpha = 0.9, label = label + ' $R^2 = $' + str(round(r_square, 5)) + ', D = ' + str(round(D, 2))) 
            
    legend = ax.legend(loc='left', shadow=True, fontsize = 20)


    outdir = '2_r_rule_data'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dataf = open(outdir + '/' + title + 'NNstar_data.dat', 'w')
    for i in range(len(bx_average_ratings)):
        dataf.write(str(bx_average_ratings[i]) + '\t' + str(bp_average_ratings[i]) + '\t' + str(bperr_average_ratings[i] )+ '\n')
    dataf.close()

     




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



    ''' FILM '''
    
    num_of_bins = 15
    title_font  = 25 
    Nmin = 15
    seaborn.set_style('white')   

    xxx    = np.arange(0,1, 1.0/20)
    folder = '../ProcessedData/ProcessedData_0_Normalized'    
    
    professions = ['director', 'producer', 'writer', 'composer', 'art-director']
   
    for label in professions:

        f, ax = plt.subplots(3, 2, figsize=(23, 23))
        st = f.suptitle( "Relative rank of the best, " + label, fontsize=title_font)
   
        file_cnt_all = folder + '/4_NN_rank_N/film_best_product_NN_ranks_all_rating_count_'   + label + '.dat'           
        r_square_cnt = plot_ccdf(file_cnt_all,  num_of_bins, ax[0,1], 'b',  label, Nmin, label , 'o' )

        plot_red_lines(ax, xxx)        
        #plt.show()



    ''' MUSIC  '''
    
    genres  = ['electro', 'pop', 'rock', 'classical', 'hiphop', 'funk', 'folk', 'jazz'] 
    f, muax = plt.subplots(3, 3, figsize=(25, 25))
    st = f.suptitle( "Music inflation curves", fontsize=title_font)
                
    for i in range(3): 
        for j in range(3):
            genre_ind = i*3 + j
            if genre_ind < len(genres):                   
                genre = genres[genre_ind]          
                print 'PROCESSING -- ' + genre
                file_music   = folder + '/4_NN_rank_N/music_best_product_NN_ranks_all_play_count_'+genre+'.dat'
                r_square_cnt = plot_ccdf(file_music,  num_of_bins, muax[i,j], 'g', genre, Nmin, 'Play count, ' + genre, 'o'  )

    plot_red_lines(muax, xxx)
    #plt.show()

    
    ''' BOOKS   '''      

    f, bax = plt.subplots(1, 3, figsize=(22, 7))
    st = f.suptitle( "Books inflation curves", fontsize=title_font)

    file_cnt_all  = folder + '/4_NN_rank_N/book_best_product_NN_ranks_all_rating_count_'   + 'authors' + '.dat'
    r_square_cnt  = plot_ccdf(file_cnt_all,  num_of_bins, bax[1], 'r', '', Nmin, 'Individual, Goodreads rating count'    , 'o'  )

    plot_red_lines(bax, xxx)
    plt.show()

  

get_r_test()



