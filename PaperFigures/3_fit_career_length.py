execfile("0_imports.py")



def get_career_length():


    FOLDER   = '../ProcessedData/ProcessedData_0_Normalized'    
    f, ax    = plt.subplots(2, 2, figsize=(23, 23))

    distancefile = 'ResultData/3_q_model_stat/KS_distances_career_length.dat'
    outfolder    = 'ResultData/3_q_model_stat'

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    f_Ddata  = open(distancefile, 'w')
    f_Ddata.write('Field\tPowerlaw\tLognormal\n')
    f_Ddata.close()

    fields = [    ('director',     'film_career_length_rating_count_'),
                  ('producer',     'film_career_length_rating_count_'),   
                  ('writer',       'film_career_length_rating_count_'),
                  ('composer',     'film_career_length_rating_count_'),
                  ('art-director', 'film_career_length_rating_count_'),
                  ('electro',      'music_career_length_play_count_'),      
                  ('pop',          'music_career_length_play_count_'),      
                  ('rock',         'music_career_length_play_count_'),      
                  ('funk',         'music_career_length_play_count_'), 
                  ('folk',         'music_career_length_play_count_'), 
                  ('classical',    'music_career_length_play_count_'), 
                  ('jazz',         'music_career_length_play_count_'), 
                  ('hiphop',       'music_career_length_play_count_'), 
                  ('authors',      'book_career_length_rating_count_')]


    Pros = []
    
    for (label, fn) in fields:

        file_cnt  = FOLDER + '/8_career_length/' + fn + label + '.dat'              
        p = Process(target = fit.fitPowerLaw, args=(file_cnt,   ax[1,0], label, 'ResultData/3_q_model_stat', 'career_length', 9, True, True, distancefile, ))

        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()
    


get_career_length()

   
