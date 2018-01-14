execfile("0_imports.py")


def get_norm_log_p():

    
    FOLDER   = '../ProcessedData/ProcessedData_0_Normalized/'    
    f, ax    = plt.subplots(1, 2, figsize=(20, 6))

    statfile = 'ResultData/5_pQ_fit/STAT_log_Q.dat'
    outfolder    = 'ResultData/5_pQ_fit'

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    fout = open(statfile, 'w')
    fout.write('label\tD\tmean\tvariance\tskewness\tkurtosity\n')
    fout.close()

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
        filenm  = FOLDER + '/11_log_Q_wout_means/' + fn + label + '.dat'              
        p = Process(target = fit.fitAndStatsSkewedNormal, args=(filenm, ax[0], label, outfolder, 'log_Q', statfile, 10,))

        Pros.append(p)
        p.start()
      
    for t in Pros:
        t.join()  


 

get_norm_log_p()
