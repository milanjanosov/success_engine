execfile("0_imports.py")



def plot_ccdf(folder, fn, num_of_bins, ax, label):

    file_avg_all = folder + '/4_NN_rank_N/' + fn + label + '.dat'           

    x_Nstar_avg_all, p_Nstar_avg_all, len_career, r_square, numInd = parse_N_star_N_data(file_avg_all)
    
    bx_average_ratings, bp_average_ratings, bperr_average_ratings = binning.getPercentileBinnedDistribution(np.asarray(x_Nstar_avg_all), np.asarray(p_Nstar_avg_all), num_of_bins)
    
    bx_average_ratings    = np.asarray(bx_average_ratings)  #(bx_average_ratings[1:] + bx_average_ratings[:-1])/2  
    bp_average_ratings    = np.asarray(bp_average_ratings)
    bperr_average_ratings = np.asarray(bperr_average_ratings) 


    D = stats.kstest(bp_average_ratings, 'uniform' )[0]

    ax.fill_between(bx_average_ratings, bp_average_ratings-bperr_average_ratings, bp_average_ratings+bperr_average_ratings, alpha = 0.2, color = 'b')                  
    ax.errorbar(bx_average_ratings, bp_average_ratings, yerr=bperr_average_ratings, fmt='b' + '-', linewidth = 2,  markersize = 0,marker = 'o', alpha = 0.9) 
            
  

    outdir = 'ResultData/2_r_rule_data'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dataf = open(outdir + '/' + label + '_NNstar_data.dat', 'w')
    for i in range(len(bx_average_ratings)):
        dataf.write(str(bx_average_ratings[i]) + '\t' + str(bp_average_ratings[i]) + '\t' + str(bperr_average_ratings[i] )+ '\n')
    dataf.close()

     

def parse_N_star_N_data(filename):

    try:
        N_star_N = []
        
        for line in open(filename):
        
            fields   = line.strip().split('\t')    
            best_id  = float(fields[0])
            career_N = float(fields[1])
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


    num_of_bins = 10
    f, ax       = plt.subplots(1, 2, figsize=(23, 23)) 
    folder      = '../ProcessedData/ProcessedData_0_Normalized'    
    

    fields = [    ('director',     'film_best_product_NN_ranks_all_rating_count_'),
                  ('producer',     'film_best_product_NN_ranks_all_rating_count_'),  
                  ('writer',       'film_best_product_NN_ranks_all_rating_count_'),
                  ('composer',     'film_best_product_NN_ranks_all_rating_count_'),
                  ('art-director', 'film_best_product_NN_ranks_all_rating_count_'),
                  ('electro',      'music_best_product_NN_ranks_all_play_count_'),      
                  ('pop',          'music_best_product_NN_ranks_all_play_count_'),      
                  ('rock',         'music_best_product_NN_ranks_all_play_count_'),      
                  ('funk',         'music_best_product_NN_ranks_all_play_count_'), 
                  ('folk',         'music_best_product_NN_ranks_all_play_count_'), 
                  ('classical',    'music_best_product_NN_ranks_all_play_count_'), 
                  ('jazz',         'music_best_product_NN_ranks_all_play_count_'), 
                  ('hiphop',       'music_best_product_NN_ranks_all_play_count_'), 
                  ('authors',      'book_best_product_NN_ranks_all_rating_count_')]


    for (label, fn) in fields:
        print label
        plot_ccdf(folder, fn, num_of_bins, ax[0], label)



 
get_r_test()



