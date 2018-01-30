execfile("0_imports.py")



def get_yearly_pdfs():


    for normalize in ['no', 'yearly_avg', 'years_all', 'field_avg' ,  'fields_all'][2:3]:

        

        #FOLDE    = '../ProcessedData/ProcessedDataNormalized_' + normalize    
        FOLDER    = '../ProcessedData_all_years/ProcessedDataNormalized_' + normalize    
        f, ax     = plt.subplots(1, 2, figsize=(23, 6))
        outfolder = 'ResultData/1_impact_distributions_all_years/yearly_distr'
        
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)


        fields = [    ('director',     'filmyearly_valuesrating_count_'),                    
                      ('composer',     'filmyearly_valuesrating_count_'),                     
                      ('classical',    'musicyearly_valuesplay_count_'), 
                      ('authors',      'bookyearly_valuesrating_count_')]
        
    
        times   = []
        lengths = []
        
        for (label, fn) in fields[3:]:

            file_cnt  = FOLDER + '/12_yearly_values/' + fn + label + '.dat'    

            for index, line in enumerate(open(file_cnt)):
          
                fields = line.strip().split('\t')
                year   = int(fields[0].split('.')[0])
        
                if year > 0:

                    values = [float(v) for v in fields[1:]]

                    lengths.append(len(values))
                    times.append(year)

      
        ax[0].set_yscale('log')
        ax[0].plot(times, lengths, 'bo')
        ax[1].plot(times, lengths, 'bo')
        align_plot(ax)
        plt.show()


    

get_yearly_pdfs()
