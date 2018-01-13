execfile("0_imports.py")




def get_imapct_distr():             
            

    FOLDER   = '../ProcessedData/ProcessedData_0_Normalized'    
    f, ax    = plt.subplots(2, 2, figsize=(23, 23))

    fields = [    ('director',     'film_rating_count_dist_'),
                  ('producer',     'film_rating_count_dist_'),  
                  ('writer',       'film_rating_count_dist_'),
                  ('composer',     'film_rating_count_dist_'),
                  ('art-director', 'film_rating_count_dist_'),
                  ('rock',         'music_play_count_dist_'),      
                  ('funk',         'music_play_count_dist_'), 
                  ('folk',         'music_play_count_dist_'), 
                  ('classical',    'music_play_count_dist_'), 
                  ('jazz',         'music_play_count_dist_'), 
                  ('hiphop',       'music_play_count_dist_'), 
                  ('authors',      'book_rating_count_dist_')]

    Pros = []
    
    for (label, fn) in fields:

        file_cnt  = FOLDER + '/1_impact_distributions/film_rating_count_dist_'   + label + '.dat'              

        p = Process(target = fit.fitPowerLaw, args=(file_cnt,   ax[1,0], label, '1_impact_distributions', ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()



        
get_imapct_distr()
        
        
