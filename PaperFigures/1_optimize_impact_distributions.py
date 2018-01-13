execfile("0_imports.py")
    

def get_KS_results(args):


    fn, N, ax, label, outfolder, FOLDER = args  

    cutoffs = []
    Dlogs   = []
    Dpows   = []

    file_cnt = FOLDER + '/1_impact_distributions/' + fn + label + '.dat'   
    data     =  np.asarray(sorted([float(line.strip()) for line in open(file_cnt) ]))
    min_data = min( data )
    num_data = len( data )
    max_data = max( data[0:int(0.9*num_data)] )
    cutoffs = np.logspace(np.log10(min_data),np.log10(max_data), N)


    for index, cutoff in enumerate(cutoffs):
    
        print label, '  ', index+1, '/', N
        Dlog, Dpow = fit.fitPowerLaw(file_cnt, ax, '',  '', '', cutoff = -   sys.maxint, writeout = False, noise = False, distancefile = '')

        Dlogs.append(Dlog)
        Dpows.append(Dpow)
       

    fff = open(outfolder + label +'_xmin_D.dat', 'w')
    fff.write('cutoff\tlognormal\tpowerlaw\n')
    for i in range(N):
        fff.write(str(cutoffs[i]) + '\t' + str(Dlogs[i]) + '\t' + str(Dpows[i]) + '\n' )
    fff.close()  



def optimize_xmin():

   
    FOLDER    = '../ProcessedData/ProcessedData_0_Normalized'       
    outfolder = 'ResultData/1_impact_distributions/'
    f , ax    = plt.subplots(1, 2, figsize=(23, 23)) 
    if not os.path.exists(outfolder):
        os.makedirs(outfolder) 

    N = 3
    
 
    fields = [    ('director',     'film_rating_count_dist_'),
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
                  ('authors',      'book_rating_count_dist_')]


    Pros = []
    
    for (label, fn) in fields[0:1]:
        p = Process(target = get_KS_results, args=([fn, N, ax[0], label, outfolder, FOLDER], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()



optimize_xmin()


