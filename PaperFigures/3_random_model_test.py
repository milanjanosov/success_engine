execfile("0_imports.py")




def divideUnequal(list1, list2):


    counter=0
    step=0
    divided=[]
    for count in list1:
        count = int(count)
        step= counter+ count
        sublist= list2[counter: step]
        counter= step
        divided.append(sublist)
        

    return divided



def get_averaged_vectors(career_len, data, N, randomized):

    career_len_gen = []
    career_max_gen = []
     
    for i in range(100):
 
        if randomized == True: 
            random.shuffle(data)
       
        career_len_gen += career_len   
        synt_careers = divideUnequal(career_len, data)  

        for synt_career in synt_careers:
            career_max_gen.append(max(synt_career))    

    return career_len_gen, career_max_gen



def get_r_model_curves(data_file, max_data_file, ax, label, num_of_bins, title, xlabel, ylabel, log = False):


    ax.set_title(title,   fontsize = 19)
    ax.set_xlabel(xlabel, fontsize = 17)
    ax.set_ylabel(ylabel, fontsize = 17)


    data = [float(line.strip()) for line in open(data_file)]# if 'nan' not in line]
    (data_max, career_len) = zip(*[[float(num) for num in line.strip().split('\t')] for line in open(max_data_file)])#  if 'nan' not in line])
    ax.plot(career_len, data_max, marker = 'o', color = 'lightgrey', alpha = 0.15,linewidth = 0)
    
  
    career_len_gen,  career_max_gen  = get_averaged_vectors(career_len, data[:], 10, randomized = True)
    career_len_data, career_max_data = get_averaged_vectors(career_len, data[:], 10, randomized = False)
 
    xb_data, pb_data, pberr_data = binning.getPercentileBinnedDistribution(np.asarray(career_len_data),  np.asarray(career_max_data), num_of_bins)     
    xb_gen, pb_gen, pberr_gen    = binning.getPercentileBinnedDistribution(np.asarray(career_len_gen),  np.asarray(career_max_gen), num_of_bins)

    ax.errorbar(xb_data, pb_data, yerr = pberr_data, fmt = 'o-', color = 'grey', label = 'data', alpha = 0.9)
    ax.plot(xb_gen,  pb_gen, color = 'r', label = 'R-model', alpha = 0.9)                

    ax.set_ylim([min([min(pb_gen),  min(pb_data)]),  max(max(pb_data) + max(pberr_data), max(xb_gen) + max(pberr_gen))])
    ax.set_xlim([min([min(xb_gen),  min(xb_data)])-1,max(xb_data)+1])    

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')



    outdir = '3_r_model_data'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fdata = open(outdir + '/' + label + '_r_model_bin_data.dat', 'w')
    for i in range(len(xb_data)):
        fdata.write(str(xb_data[i]) + '\t' + str(pb_data[i]) + '\t' + str(pberr_data[i]) + '\n')
    fdata.close


    fdata = open(outdir + '/' + label + '_r_model_raw_data.dat', 'w')
    for i in range(len(career_len)):
        fdata.write(str(career_len[i]) + '\t' + str(data_max[i]) + '\n')
    fdata.close

    
    fdata = open(outdir + '/' + label + '_r_model_bin_gen.dat', 'w')
    for i in range(len(xb_gen)):
        fdata.write(str(xb_gen[i]) + '\t' + str(pb_gen[i]) + '\n')
    fdata.close





def do_the_r_model():


    title_font  = 25 
    num_of_bins = 12
    seaborn.set_style('white')  

    
    folder = '../ProcessedData/ProcessedData_0_Normalized' 



    ''' FILM   '''
    
    professions = ['director', 'producer', 'writer', 'composer', 'art-director']
    
    for label in professions[0:1]:
        
        f, ax = plt.subplots(3, 2, figsize=(23, 23))
        st = f.suptitle( "Film - R - model vs data (movie directors and DJs), " + label, fontsize=title_font)

        file_cnt    = folder + '/7_career_length_max_impact/film_rating_count_dist_'    + label + '.dat'
        max_rat_cnt = folder + '/7_career_length_max_impact/film_career_length_max_rating_count_'    + label + '.dat'  
        get_r_model_curves(file_cnt,  max_rat_cnt,    ax[0,0], label, num_of_bins, 'Rating count vs career length'   , 'Career length', 'Rating count'  , True)            

        plt.show()









               
  
    ''' MUSIC   '''
    '''
    genres  = ['electro', 'pop', 'rock', 'jazz', 'folk', 'funk', 'hiphop', 'classical']

     
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

    #plt.show()
    '''
    

    ''' BOOKS   '''  
    '''
    f, bax = plt.subplots(1, 3, figsize=(25, 12))
    st = f.suptitle( "Books, R - model vs data", fontsize=title_font) 
        


    file_book = folder +  mode + '/1_impact_distributions/book_rating_count_dist_authors.dat'
    max_book  = folder +  mode + '/7_career_length_max_impact/book_career_length_max_rating_count_authors.dat' 
    get_r_model_curves(file_book, max_book, bax[1], 'book', num_of_bins, 'Rating_count vs career length', 'Career length', 'Rating_count', True)
    

  
    #plt.show()

                
    '''
    
      


do_the_r_model()


        
