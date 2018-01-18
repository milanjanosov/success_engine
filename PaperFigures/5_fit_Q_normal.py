execfile("0_imports.py")



def do_the_fits(filenm, ax, label, outfolder, statfile, statfile_t, statfile_f, N, var = 'log_Q'):

    fit.fitAndStatsSkewedNormal(filenm, ax, label, outfolder, var, statfile,   N)
    fit.fitAndStatsNormal(filenm, ax, label, outfolder, var, statfile_f, N)
    fit.fitAndStatsTransformedNormal(filenm, ax, label, outfolder, var, statfile_t, N)


def get_norm_log_p():

    
    FOLDER   = '../ProcessedData/ProcessedData_0_Normalized/'    
    f, ax    = plt.subplots(2, 2, figsize=(20, 12))

    statfile     = 'ResultData/5_pQ_fit/STAT_log_Q.dat'
    statfile_f   = 'ResultData/5_pQ_fit/STAT_log_Q_fnorm.dat'
    statfile_t   = 'ResultData/5_pQ_fit/STAT_log_Q_tnorm.dat'
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
        p = Process(target = do_the_fits, args=(filenm, ax[0,0], label, outfolder, statfile, statfile_t, statfile_f, 10, 'log_Q', ))

        Pros.append(p)
        p.start()
      
    for t in Pros:
        t.join()  
    '''
    
    filenm  = FOLDER + '/11_log_Q_wout_means/' + 'film_log_Q_wout_mean_rating_count_' + 'director' + '.dat'              
    fit.fitAndStatsNormal(filenm, ax[0,0], 'director', outfolder, 'log_Q', statfile, 10)
    fit.fitAndStatsSkewedNormal(filenm, ax[0,1], 'director', outfolder, 'log_Q', statfile, 10)
    fit.fitAndStatsTransformedNormal(filenm, ax[1,0], 'director', outfolder, 'log_Q', statfile, 10)

    plt.savefig('5_different_Q_normal_fits.png')
    plt.show() 
    '''


get_norm_log_p()
