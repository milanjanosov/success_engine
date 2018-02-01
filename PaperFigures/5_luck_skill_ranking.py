execfile("0_imports.py")


def do_the_fits(filenm, ax, label, outfolder, statfile, statfile_t, statfile_f, N, var = 'log_p'):

    fit.fitAndStatsSkewedNormal(filenm, ax, label, outfolder, var, statfile,   N)
    fit.fitAndStatsNormal(filenm, ax, label, outfolder, var, statfile_f, N)
    fit.fitAndStatsTransformedNormal(filenm, ax, label, outfolder, var, statfile_t, N)


def fit_everything():

   
    FOLDER   = '../ProcessedData/ProcessedDataNormalized_no'    
    f, ax    = plt.subplots(1, 2, figsize=(20, 6))

    statfile     = 'ResultData/5_pQ_fit/STAT_log_p.dat'
    statfile_f   = 'ResultData/5_pQ_fit/STAT_log_p_fnorm.dat'
    statfile_t   = 'ResultData/5_pQ_fit/STAT_log_p_tnorm.dat'
    outfolder    = 'ResultData/5_pQ_fit'

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    fout = open(statfile, 'w')
    fout.write('label\tD\tmean\tvariance\tskewness\tkurtosity\n')
    fout.close()

    fields = [    ('director',     'film_p_without_mean_rating_count_'),
                  ('producer',     'film_p_without_mean_rating_count_'),   
                  ('writer',       'film_p_without_mean_rating_count_'),
                  ('composer',     'film_p_without_mean_rating_count_'),
                  ('art-director', 'film_p_without_mean_rating_count_'),
                  ('electro',      'music_p_without_mean_play_count_'),      
                  ('pop',          'music_p_without_mean_play_count_'),      
                  ('rock',         'music_p_without_mean_play_count_'),      
                  ('funk',         'music_p_without_mean_play_count_'), 
                  ('folk',         'music_p_without_mean_play_count_'), 
                  ('classical',    'music_p_without_mean_play_count_'), 
                  ('jazz',         'music_p_without_mean_play_count_'), 
                  ('hiphop',       'music_p_without_mean_play_count_'), 
                  ('authors',      'book_p_without_mean_rating_count_')]


    Pros = []
    


    for (label, fn) in fields:
        filenm  = FOLDER + '/11_log_Q_wout_means/' + fn + label + '.dat'              
        
        #p = Process(target = do_the_fits, args=(filenm, ax[0,0], label, outfolder, statfile, statfile_t, statfile_f, 10, 'log_Q', ))
        #Pros.append(p)
        #p.start()


        p = Process(target = fit.fitAndStatsSkewedNormal, args=(filenm, ax[0], label, outfolder, 'log_Q', statfile, 10, ))
        Pros.append(p)
        p.start()


        p = Process(target = fit.fitAndStatsNormal, args=(filenm, ax[0], label, outfolder, 'log_Q', statfile_t, 10, ))
        Pros.append(p)
        p.start()


        p = Process(target = fit.fitAndStatsTransformedNormal, args=(filenm, ax[0], label, outfolder, 'log_Q', statfile_f, 10, ))
        Pros.append(p)
        p.start()

        

        fn = fn.replace('log_Q', 'log_p')
        filenm  = FOLDER + '/9_p_without_avg/' + fn + label + '.dat'              

        p = Process(target = fit.fitAndStatsSkewedNormal, args=(filenm, ax[0], label, outfolder, 'log_p', statfile, 100, ))
        Pros.append(p)
        p.start()


        p = Process(target = fit.fitAndStatsNormal, args=(filenm, ax[0], label, outfolder, 'log_p', statfile_t, 100, ))
        Pros.append(p)
        p.start()


        p = Process(target = fit.fitAndStatsTransformedNormal, args=(filenm, ax[0], label, outfolder, 'log_p', statfile_f, 100, ))
        Pros.append(p)
        p.start()



      
    for t in Pros:
        t.join()  



fit_everything()
