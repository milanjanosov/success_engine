execfile("0_imports.py")




       




''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''
'''                                                                '''   
'''          GET THE DISTRIBUTION OF ALL SUCCESS MEASURES          '''
'''                                                                '''
''' -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  '''  


def optimize_xmin():



    cutoffs = []
    Dlogs = []
    Dpows = []

    title_font = 12 
    
    FOLDER_S = 'ProcessedDataSample/ProcessedData_0_Normalized_Sample' 
    FOLDER   = 'ProcessedData/ProcessedData_0_Normalized'# + '_Sample'        


    N = 30 
    
    
    outfolder = 'ResultData/1_impact_distributions/'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder) 
    
    '''
    professions = [ 'director', 'producer', 'writer', 'composer', 'art-director']
 

   
    for label in professions:
    
    
        file_cnt  = FOLDER + '/1_impact_distributions/film_rating_count_dist_'   + label + '.dat'   
        data =  np.asarray(sorted([float(line.strip()) for line in open(file_cnt) ]))
        min_data = min( data )
        num_data = len( data )
        max_data = max( data[0:int(0.9*num_data)] )
        cutoffs = np.logspace(np.log10(min_data),np.log10(max_data), N)


        for cutoff in cutoffs:
        
            f, ax = plt.subplots(1, 2, figsize=(23, 23))

            print cutoff
            Dlog, Dpow = fit.fitPowerLaw(file_cnt,   ax[0], 'IMDb, rating counts director', cutoff, writeout = False)
            Dlogs.append(Dlog)
            Dpows.append(Dpow)
           

     
        fff = open(outfolder + label +'_xmin_D.dat', 'w')
        for i in range(N):
            fff.write(str(cutoffs[i]) + '\t' + str(Dlogs[i]) + '\t' + str(Dpows[i]) + '\n' )
        fff.close()  
       
    '''

    
    Dlogs = []
    Dpows = []

    genres = ['rock', 'funk', 'folk', 'classical', 'jazz', 'hiphop']#['electro', 'pop']
       
    f, muax = plt.subplots(3, 3, figsize=(25, 25))
    
                
    for i in range(3): 
        for j in range(3):
            genre_ind = i*3 + j
            if genre_ind < len(genres):                   
                genre = genres[genre_ind]          
                print 'PROCESSING -- ' + genre
                
                file_music = FOLDER + '/1_impact_distributions/music_play_count_dist_' + genre + '.dat'
                
                
                data =  np.asarray(sorted([float(line.strip()) for line in open(file_music) ]))
                min_data = min( data )
                num_data = len( data )
                max_data = max( data[0:int(0.9*num_data)] )
                cutoffs = np.logspace(np.log10(min_data),np.log10(max_data), N)


                for cutoff in cutoffs:
                
                    f, ax = plt.subplots(1, 2, figsize=(23, 23))

                    print cutoff
                    Dlog, Dpow = fit.fitPowerLaw(file_music,   ax[0], 'IMDb, rating counts director', cutoff, writeout = False)
                    Dlogs.append(Dlog)
                    Dpows.append(Dpow)                
                   

                print genre

                fff = open(outfolder + genre +'_xmin_D.dat', 'w')
                for ijk in range(N):
                    fff.write(str(cutoffs[ijk]) + '\t' + str(Dlogs[ijk]) + '\t' + str(Dpows[ijk]) + '\n' )
                fff.close()  
     
           
       
    ''' ---------------------------------------------- '''
    ''' BOOKS   '''      
    '''f, bax = plt.subplots(1, 3, figsize=(25, 12))

    Dlogs = []
    Dpows = []    
    print 'PROCESSING --  books'       
    book_cnt = FOLDER + '/1_impact_distributions/book_rating_count_dist_authors.dat'
          

    data =  np.asarray(sorted([float(line.strip()) for line in open(book_cnt) ]))
    min_data = min( data )
    num_data = len( data )
    max_data = max( data[0:int(0.9*num_data)] )
    cutoffs = np.logspace(np.log10(min_data),np.log10(max_data), N)


    for cutoff in cutoffs:
    
        f, ax = plt.subplots(1, 2, figsize=(23, 23))

        print cutoff
        Dlog, Dpow = fit.fitPowerLaw(book_cnt,   ax[0], 'IMDb, rating counts director', cutoff, writeout = False)
        Dlogs.append(Dlog)
        Dpows.append(Dpow)
       

 
    fff = open(outfolder + 'book_authors_xmin_D.dat', 'w')
    for i in range(N):
        fff.write(str(cutoffs[i]) + '\t' + str(Dlogs[i]) + '\t' + str(Dpows[i]) + '\n' )
    fff.close()  
            
    '''






   
def get_imapct_distr():             
            
              
    num_of_bins = 20
    title_font  = 25 
    seaborn.set_style('white')  

    cut = 'nocutoff' 

 

    for mode in ['', 'Normalized'][1:]:
    
        
        mode_    = 'Original' if mode == '' else 'Normalized'
        FOLDER_S = 'ProcessedDataSample/ProcessedData_0_' + mode + '_Sample' 
        FOLDER   = 'ProcessedData/ProcessedData_0_' + mode# + '_Sample'        
  
        ''' ---------------------------------------------- '''
        ''' MOVIES   '''
        
        professions = ['_MERGED', 'director', 'producer', 'writer', 'composer', 'art-director']
   
        for label in professions[1:]:
            
            print 'PROCESSING -- ' + label
            #f, ax = plt.subplots(3, 2, figsize=(23, 23))
            f, ax = plt.subplots(2, 2, figsize=(23, 23))
            st = f.suptitle( mode + "IMDb impact distributions for " + label, fontsize=title_font)

   

            file_cnt  = FOLDER + '/1_impact_distributions/film_rating_count_dist_'   + label + '.dat'
       
                          
            fit.fitPowerLaw(file_cnt,   ax[1,0], 'IMDb, rating counts'   + label )#), 0.01)
            
            #pltplot(ax)
            plt.show()

          
        
        
                      
        ''' ---------------------------------------------- '''
        ''' MUSIC   '''
        
        genres = ['electro', 'pop', 'rock', 'funk', 'folk', 'classical', 'jazz', 'hiphop']#['electro', 'pop']
        
        f, muax = plt.subplots(3, 3, figsize=(25, 25))
        st = f.suptitle( mode + "Music impact distributions", fontsize=title_font)
                    
        for i in range(3): 
            for j in range(3):
                genre_ind = i*3 + j
                if genre_ind < len(genres):                   
                    genre = genres[genre_ind]          
                    print 'PROCESSING -- ' + genre
                    file_music = FOLDER + '/1_impact_distributions/music_play_count_dist_' + genre + '.dat'
                    rating_cnt_fit   = fit.fitPowerLaw(file_music, muax[i,j], genre + ' music, play counts') #, 0.01)



             
       
        ''' ---------------------------------------------- '''
        ''' BOOKS   '''      
        f, bax = plt.subplots(1, 3, figsize=(25, 12))
        st = f.suptitle( mode + "Books impact distributions", fontsize=title_font)
        
        print 'PROCESSING --  books'       
        book_avg = FOLDER + '/1_impact_distributions/book_average_rating_dist_authors.dat'
        book_cnt = FOLDER + '/1_impact_distributions/book_rating_count_dist_authors.dat'
        book_ed  = FOLDER + '/1_impact_distributions/book_edition_count_dist_authors.dat'            
              

        fit.fitPowerLaw    (book_cnt, bax[1], 'Goodreads, rating count' )#     , 0.01)  



        
        
get_imapct_distr()
        
        
