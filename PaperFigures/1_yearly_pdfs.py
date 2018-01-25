execfile("0_imports.py")


def get_pdf(rand, ax, label = '', out_folder = '', name = '', writeout = True, norm = 'no'):

     

    x_rand, p_rand = getDistribution(rand, True)                  
    



    nbins = 20

    print 'xrand', len(x_rand)

    counts, bins, bars = ax.hist(rand, normed = True, bins = 10 ** np.linspace(np.log10(min(x_rand)), np.log10(max(x_rand)), nbins), log=True, alpha=0.0, cumulative=0) 
    ax.plot((bins[1:] + bins[:-1])/2, counts, 'o', alpha = 0.7, markersize = 4, linewidth = 2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    bins = (bins[1:] + bins[:-1])/2    


    #plt.plot(x_rand[0::100], p_rand[0::100], 'bo')
    #plt.show()

    write_row(out_folder + '/' + label + '_' + name + '_lognormal_hist_pdf_' + str(nbins) + '_'  + norm + '_nonorm' + '.dat', [str(bins[i])   + '\t' + str(counts[i]) for i in range(len(counts)) if counts[i] > 0] )

   



def get_yearly_pdfs():


    for normalize in ['no', 'yearly_avg', 'years_all', 'field_avg' ,  'fields_all'][2:3]:

        

        #FOLDE    = '../ProcessedData/ProcessedDataNormalized_' + normalize    
        FOLDER    = '../ProcessedData_all_years/ProcessedDataNormalized_' + normalize    
        f, ax     = plt.subplots(2, 2, figsize=(23, 23))
        outfolder = 'ResultData/1_impact_distributions_all_years/yearly_distr'
        
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)



        '''fields = [    ('director',     'film_rating_count_dist_'),
                      ('producer',     'film_rating_count_dist_'),  
                      ('writer',       'film_rating_count_dist_'),
                      ('composer',     'film_rating_count_dist_'),
                      ('art-director', 'film_rating_count_dist_'),
                      ('electro',      'music_play_count_dist_'),      
                      ('pop',          'music_play_count_dist_'),      
                      ('rock',         'music_play_count_dist_'),      
                      ('funk',         'music_play_count_dist_'), 
                      ('folk',         'music_play_count_dist_'), 
                      ('classical',    'music_play_count_dist_'), 
                      ('jazz',         'music_play_count_dist_'), 
                      ('hiphop',       'music_play_count_dist_'), 
                      ('authors',      'bookyearly_valuesrating_count_')]
        '''



        fields = [    ('director',     'filmyearly_valuesrating_count_'),                    
                      ('composer',     'filmyearly_valuesrating_count_'),                     
                      ('classical',    'musicyearly_valuesplay_count_'), 
                      ('authors',      'bookyearly_valuesrating_count_')]
        

        all_values = []
        
        for (label, fn) in fields[3:]:

            file_cnt  = FOLDER + '/12_yearly_values/' + fn + label + '.dat'    



            for index, line in enumerate(open(file_cnt)):
          
                

                fields = line.strip().split('\t')
                year   = int(fields[0].split('.')[0])
        
                if year > 0:

                    values = [float(v) for v in fields[1:]]


                    all_values += values

                get_pdf(values, ax[0,0], label = str(year), out_folder = outfolder, name = label, writeout = True, norm = 'no')
            #p = Process(target = fit.fitLognormal, args=(file_cnt, ax[1,0], label, 'ResultData/1_impact_distributions', 'impact_distributions', -sys.maxint, True, 0, normalize, ))
 

            get_pdf(all_values, ax[0,0], label = 'all_years', out_folder = outfolder, name = label, writeout = True, norm = 'no')



    #plt.show()

get_yearly_pdfs()
