execfile("0_imports.py")



def get_p_without_avg():


    FOLDER       = '../ProcessedData/ProcessedData_0_Normalized'    
    f, ax        = plt.subplots(2, 2, figsize=(23, 23))
    distancefile = 'ResultData/3_q_model_stat/KS_distances_log_Q.dat'
    outfolder    = 'ResultData/3_q_model_stat'

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    f_Ddata  = open(distancefile, 'w')
    f_Ddata.write('Field\tPowerlaw\tLognormal\n')
    f_Ddata.close()

    
    fields = [    ('director',     'film_log_Q_wout_mean_rating_count_'),
                  ('producer',     'film_log_Q_wout_mean_rating_count_'),   
                  ('writer',       'film_log_Q_wout_mean_rating_count_'),
                  ('composer',     'film_log_Q_wout_mean_rating_count_'),
                  ('art-director', 'film_log_Q_wout_mean_rating_count_'),
                  ('electro',      'music_log_Q_wout_mean_play_count_'),      
                  ('pop',          'music_log_Q_wout_mean_play_count_'),      
                  ('rock',         'music_log_Q_wout_mean_play_count_'),      
                  ('funk',         'music_log_Q_wout_mean_play_count_'), 
                  ('folk',         'music_log_Q_wout_mean_play_count_'), 
                  ('classical',    'music_log_Q_wout_mean_play_count_'), 
                  ('jazz',         'music_log_Q_wout_mean_play_count_'), 
                  ('hiphop',       'music_log_Q_wout_mean_play_count_'), 
                  ('authors',      'book_log_Q_wout_mean_rating_count_')]

    Pros = []
    
    for (label, fn) in fields:
        file_cnt  = FOLDER + '/11_log_Q_wout_means/' + fn + label + '.dat'              
        p = Process(target = fit.fitPowerLaw, args=(file_cnt, ax[1,0], label, 'ResultData/3_q_model_stat', 'log_Q', -sys.maxint, True, 0, distancefile,  ))

        Pros.append(p)
        p.start()
      
    for t in Pros:
        t.join()



get_p_without_avg()



