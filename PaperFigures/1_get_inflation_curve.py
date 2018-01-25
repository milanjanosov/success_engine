execfile("0_imports.py")



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
            
    x, y, err = zip(*[(year, np.mean(impacts), np.std(impacts)) for year, impacts in years_impacts.items() ])

    return np.asarray(x), np.asarray(y), np.asarray(err)




def plot_measure(average_ratings_year, label, num_of_bins, ax, color, aa, music = False):
    
  
    x_average_ratings_year, y_average_ratings_year, yerr_average_ratings_year = get_yearly_avg_data(average_ratings_year)    
    bx_average_ratings_year, bp_average_ratings_year, bperr_average_ratings_year = binning.getBinnedDistribution(x_average_ratings_year, y_average_ratings_year, num_of_bins)

    bx_average_ratings_year = (bx_average_ratings_year[1:] + bx_average_ratings_year[:-1])/2

  
   # write_row('ResultData/1_impact_distributions/' + label + '_inflation.dat', [str(bx_average_ratings_year[i]) + '\t' + str(bp_average_ratings_year[i]) + '\t' + str(bperr_average_ratings_year[i]) for i in range(len(bx_average_ratings_year))] )   
  
    write_row('FinalData/2_' + label + '_inflation.dat', [str(bx_average_ratings_year[i]) + '\t' + str(bp_average_ratings_year[i]) + '\t' + str(bperr_average_ratings_year[i]) for i in range(len(bx_average_ratings_year))] )   


    ax.fill_between(bx_average_ratings_year, bp_average_ratings_year-bperr_average_ratings_year, bp_average_ratings_year+bperr_average_ratings_year, alpha = 0.2, color = color)



    if music:
        ax.errorbar(x_average_ratings_year, y_average_ratings_year, yerr=yerr_average_ratings_year, fmt=color + '-', alpha = 0.5, capsize = 3, elinewidth=1, linewidth = 2)
        ax.errorbar(bx_average_ratings_year, bp_average_ratings_year, yerr=bperr_average_ratings_year, fmt= 'o-', color = color, alpha = 0.9, capsize = 3, elinewidth=1, linewidth = 3, label = label)
    else:
        ax.errorbar(bx_average_ratings_year, bp_average_ratings_year, yerr=bperr_average_ratings_year, fmt= 'o-', color = color, alpha = 0.9, capsize = 3, elinewidth=1, linewidth = 3)#, label = label)
    
    ax.set_xlim([1880, 2020])




def get_inflation_curves():


    num_of_bins = 20
    title_font = 25 
  	

    FOLDER = '../ProcessedData_all_years/ProcessedDataNormalized_no'
    YEAR_MIN = 1900

    professions = [('director',     'k'), 
                   ('producer',     'b'),
                   ('writer'  ,     'r'),
                   ('composer',     'g'),
                   ('art-director', 'y')]    
    
    
    
    ''' ---------------------------------------------- '''
    ''' MOVIES   '''
    
    professions = ['director', 'producer', 'writer', 'composer', 'art-director']

    for label in professions[0:1]:
        
        print 'PROCESSING -- ' + label
        f, ax = plt.subplots(3, 2, figsize=(23, 23))
        st = f.suptitle( "Film inflation curves, " + label, fontsize=title_font)


        file_meta_year  = FOLDER + '/3_inflation_curves/film_yearly_rating_count_dist_'+label+'.dat'
        rating_cnt_year = np.asarray([line.strip() for line in open(file_meta_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
        plot_measure(rating_cnt_year, label, num_of_bins, ax[0,1], 'royalblue', '')    

    #plt.show()
    
    ''' ---------------------------------------------- '''
    ''' MUSIC   '''
    
    genres = ['electro']#, 'pop']
     
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
                plot_measure(play_cnt_year, genre , num_of_bins, muax[i,j], 'forestgreen', '')    
    
   

    ''' ---------------------------------------------- '''
    ''' BOOKS   '''      
   
    f, bax = plt.subplots(1, 3, figsize=(25, 12))
    st = f.suptitle( "Books inflation curves", fontsize=title_font)

    print 'PROCESSING --  books' 
    file_cnt_year  = FOLDER + '/3_inflation_curves/book_yearly_rating_count_dist_authors.dat'
    rating_cnt_book = np.asarray([line.strip() for line in open(file_cnt_year)  if float(line.strip().split('\t')[0]) > YEAR_MIN ] )
    plot_measure(rating_cnt_book,   'author' ,  num_of_bins, bax[2], 'Firebrick', '') 
     

get_inflation_curves()


