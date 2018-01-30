execfile("0_imports.py")




def divideUnequal(list1, list2):

    counter = 0
    step    = 0
    divided = []
    for count in list1:
        count = int(count)
        step = counter+ count
        sublist = list2[counter: step]
        counter = step
        divided.append(sublist)
        
    return divided



def get_averaged_vectors(career_len, data, N, randomized):

    career_len_gen = []
    career_max_gen = []
     
    for i in range(1):
 
        if randomized == True: 
            random.shuffle(data)
       
        career_len_gen += career_len   
        synt_careers = divideUnequal(career_len, data)  

        for synt_career in synt_careers:
            career_max_gen.append(max(synt_career))    

    return career_len_gen, career_max_gen



def get_r_model_curves(data_file, max_data_file, ax, label, num_of_bins):


    data = [float(line.strip()) for line in open(data_file)]# if 'nan' not in line]
    (data_max, career_len) = zip(*[[float(num) for num in line.strip().split('\t')] for line in open(max_data_file)])#  if 'nan' not in line])
     
    career_len_gen,  career_max_gen  = get_averaged_vectors(career_len, data[:], 10, randomized = True)
    career_len_data, career_max_data = get_averaged_vectors(career_len, data[:], 10, randomized = False)
 
    xb_data, pb_data, pberr_data = binning.getPercentileBinnedDistribution(np.asarray(career_len_data),  np.asarray(career_max_data), num_of_bins)     
    xb_gen, pb_gen, pberr_gen    = binning.getPercentileBinnedDistribution(np.asarray(career_len_gen),  np.asarray(career_max_gen), num_of_bins)

  
    outdir = 'ResultData/4_r_model_data'
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


    num_of_bins = 12
    
    folder = '../ProcessedData_1960_careers/ProcessedDataNormalized_years_all' 
    f, ax  = plt.subplots(3, 2, figsize=(23, 23))


    fields = [    ('director',     'film_rating_count_dist_',        'film_career_length_max_rating_count_'),
                  ('producer',     'film_rating_count_dist_',        'film_career_length_max_rating_count_'),  
                  ('writer',       'film_rating_count_dist_',        'film_career_length_max_rating_count_'),
                  ('composer',     'film_rating_count_dist_',        'film_career_length_max_rating_count_'),
                  ('art-director', 'film_rating_count_dist_',        'film_career_length_max_rating_count_'),
                  ('electro',      'music_play_count_dist_' ,        'music_career_length_max_play_count_'),      
                  ('pop',          'music_play_count_dist_' ,        'music_career_length_max_play_count_'),      
                  ('rock',         'music_play_count_dist_' ,        'music_career_length_max_play_count_'),      
                  ('funk',         'music_play_count_dist_',         'music_career_length_max_play_count_'), 
                  ('folk',         'music_play_count_dist_',         'music_career_length_max_play_count_'), 
                  ('classical',    'music_play_count_dist_',         'music_career_length_max_play_count_'), 
                  ('jazz',         'music_play_count_dist_',         'music_career_length_max_play_count_'), 
                  ('hiphop',       'music_play_count_dist_',         'music_career_length_max_play_count_'), 
                  ('authors',      'book_rating_count_dist_', 'book_career_length_max_rating_count_')]


    for (label, fn1, fn2) in fields[13:]:

        file_cnt    = folder + '/1_impact_distributions/'     + fn1 + label + '.dat'
        max_rat_cnt = folder + '/7_career_length_max_impact/' + fn2 + label + '.dat'  
        get_r_model_curves(file_cnt,  max_rat_cnt,    ax[0,0], label, num_of_bins)


do_the_r_model()


        
