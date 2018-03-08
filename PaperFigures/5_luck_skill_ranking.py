


''' TODO 



    - Nstar stat with release level stuff
    - rankings for release level stuff as well (maybe...? sure not mixed up)
    - run by ProcessCareerStat.py w a switch

    

'''



execfile("0_imports.py")


def do_the_fits(filenm, ax, label, outfolder, statfile, statfile_t, statfile_f, N, var = 'log_p'):

    fit.fitAndStatsSkewedNormal(filenm, ax, label, outfolder, var, statfile,   N)
    fit.fitAndStatsNormal(filenm, ax, label, outfolder, var, statfile_f, N)
    fit.fitAndStatsTransformedNormal(filenm, ax, label, outfolder, var, statfile_t, N)


def fit_everything(extra):

   
    norms = ['no', 'yearly_avg', 'field_avg', 'fields_all', 'years_all']

    for norm in norms:


        FOLDER   = '../ProcessedData/ProcessedDataNormalized_' + norm    
        f, ax    = plt.subplots(1, 2, figsize=(20, 6))

        statfile     = FOLDER + '/13_pQ_fit/STAT_skewnorm_'        + norm + '.dat'
        statfile_f   = FOLDER + '/13_pQ_fit/STAT_forcednorm_'      + norm + '.dat'
        statfile_t   = FOLDER + '/13_pQ_fit/STAT_transformednorm_' + norm + '.dat'
        outfolder    = FOLDER + '/13_pQ_fit'

        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        if extra == '':

            fout = open(statfile, 'w')
            fout.write('label\tvariable\tD\tmean\tvariance\tskewness\tkurtosity\n')
            fout.close()

            fout = open(statfile_f, 'w')
            fout.write('label\tvariable\tD\tmean\tvariance\tskewness\tkurtosity\n')
            fout.close()

            fout = open(statfile_t, 'w')
            fout.write('label\tvariable\tD\tmean\tvariance\tskewness\tkurtosity\n')
            fout.close()


        fields = [ ('director',     'film_p_without_mean_rating_count_'),
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
        




        #fields = [    ('jazz',     'music_p_without_mean_play_count_')]


       

        Pros = []
        

        for (label, fn) in fields:
            

            try:

                filenm  = FOLDER + '/11_log_Q_wout_means/' + fn.replace('_p_', '_log_Q_').replace('without', 'wout') + label + extra + '.dat'              
                
                p = Process(target = fit.fitAndStatsSkewedNormal, args=(filenm, ax[0], label + extra, outfolder, 'log_Q', statfile, 10, ))
                Pros.append(p)
                p.start()
            


            
                p = Process(target = fit.fitAndStatsNormal, args=(filenm, ax[0], label + extra, outfolder, 'log_Q', statfile_t, 10, ))
                Pros.append(p)
                p.start()



           
                p = Process(target = fit.fitAndStatsTransformedNormal, args=(filenm, ax[0], label + extra, outfolder, 'log_Q', statfile_f, 10, ))
                Pros.append(p)
                p.start()
           

                fn = fn.replace('log_Q', 'log_p')
                filenm = FOLDER + '/9_p_without_avg/' + fn + label + extra + '.dat'  
             

               
                p = Process(target = fit.fitAndStatsSkewedNormal, args=(filenm, ax[0], label + extra, outfolder, 'log_p', statfile, 100, ))
                Pros.append(p)
                p.start()
          

                p = Process(target = fit.fitAndStatsNormal, args=(filenm, ax[0], label + extra, outfolder, 'log_p', statfile_t, 100, ))
                Pros.append(p)
                p.start()
            
                p = Process(target = fit.fitAndStatsTransformedNormal, args=(filenm, ax[0], label + extra, outfolder, 'log_p', statfile_f, 100, ))
                Pros.append(p)
                p.start()
        
            
            except:
                pass

          
        for t in Pros:
            t.join()  
        





def logit(a):
    
    return 1.0/(1.0 + math.exp(-a))



def rank_stuff(labelQ, varianceQ, labelp, variancep):


    Qvars = {}
    pvars = {}

    for index, label in enumerate(labelQ):
        Qvars[label] = float(varianceQ[index])

    for index, label in enumerate(labelp):
        pvars[label] = float(variancep[index])



    distances     = [] 
    percentages   = [] 
    distances_log = []


    for label in labelQ:
        distances.append((label,     (Qvars[label] - pvars[label]) / math.sqrt(2)))
        distances_log.append((label, 100 * logit((Qvars[label] - pvars[label]) / math.sqrt(2))))
        percentages.append((label,   100 * (Qvars[label] / (pvars[label] + Qvars[label]))))

    print distances
    print percentages
    print distances_log



def get_rankings():


    # for norm in ['no', ...
    # for fitversion in ['skewnorm', '...

    infolder    = '../ProcessedData/ProcessedDataNormalized_no/13_pQ_fit'

    labelQ, var, DQ, meanQ, varianceQ, skewnessQ, kurtosityQ = zip(*[[ff for ff in line.strip().split('\t') ] for line in open(infolder + '/STAT_skewnorm.dat') if 'log_Q' in line])
    labelp, var, Dp, meanp, variancep, skewnessp, kurtosityp = zip(*[[ff for ff in line.strip().split('\t') ] for line in open(infolder + '/STAT_skewnorm.dat') if 'log_p' in line])

    rank_stuff(labelQ, varianceQ, labelp, variancep)




    
    

       


    ''' varQ - varp plot '''
        
    '''ff, ax = plt.subplots(1, 1, figsize=(14, 14))

    print Qvars
    print pvars

    ax.plot(Qvars.values(), pvars.values(), 'bo', markersize = 12)
    ax.set_xlabel('P(Q) variances', fontsize = 26)
    ax.set_ylabel('P(p) variances', fontsize = 26)
    ax.plot([1, 1.1*max(Qvars.values() + pvars.values())], [1, 1.1*max(Qvars.values() + pvars.values())], color = 'r')    

    for label in labelQ:
        ax.annotate(label, (Qvars[label], pvars[label]), fontsize = 17)

    ax.set_title('Artistic fields in the p-Q variance space', fontsize = 30)
    align_ax(ax, 18)
    #plt.savefig('5_pQvar_plane.png')
    #plt.close()       
    plt.show()    
    '''

    ''' distance from the 50:50 case on the varQ var p plane '''
        
    #for label in labelQ:
    #    print label, '\t', (Qvars[label] - pvars[label]) / math.sqrt(2) ,'\t', pvars[label] / (pvars[label] + Qvars[label])
        #print Qvars[label] * pvars[label]



    '''ff, ax = plt.subplots(1, 1, figsize=(24, 4))
    for index, dist in enumerate(distances): 
        ax.hlines(1,1,dist)     
        
        
    ax.hlines(1,0,max(distances)) 

    ax.set_xlim(min(distances),1.05*max(distances))
    ax.annotate('0', (0.01, 1.01), fontsize = 22, color = 'r')

    #ax.set_ylim(0.5,1.5)

    ax.plot([0],[1],'o',ms = 20, color = 'red')  

    y = np.ones(np.shape(distances))
    ax.plot(distances,y,'|',ms = 60, color = 'blue')


    ax.axis('off')
    for i, txt in enumerate(labelQ[0::3]):
        ax.annotate(txt, (distances[labelQ.index(txt)], 0.97), fontsize = 17,rotation=-30)

    for i, txt in enumerate(labelQ[1::3]):
        ax.annotate(txt, (distances[labelQ.index(txt)], 1.03), fontsize = 17,rotation=30)
     
    for i, txt in enumerate(labelQ[2::3]):
        ax.annotate(txt, (distances[labelQ.index(txt)], 1.03), fontsize = 17,rotation=30)
            

    #plt.savefig('5_luck_skill_variances.png')
    #plt.close()       
        
    plt.show()
    '''
        
    '''  fraction of luck - varp/(varp+varQ)  '''     
       

    '''
    ff, ax = plt.subplots(1, 1, figsize=(24, 4))
    for index, dist in enumerate(percentages): 
        ax.hlines(1,1,dist)     
          
    ax.hlines(1,0,1)     
    #ax.hlines(1,min(percentages),min(percentages)) 


    ax.plot([0.5],[0],'o',ms = 20, color = 'red')  
    ax.annotate('0%', (0, 1), fontsize = 22, color = 'r')
    ax.annotate('100%', (1, 1), fontsize = 22, color = 'r')
    ax.annotate('50%', (0.51, 0.98), fontsize = 22, color = 'r')


    #ax.annotate('37%', (min(percentages), 0.98), fontsize = 22, color = 'r')
    #ax.annotate('73%', (max(percentages), 0.98), fontsize = 22, color = 'r')




    ax.set_xlim([min(percentages),max(percentages)])
    ax.set_xlim([0,1])
    y = np.ones(np.shape(distances))
    ax.plot(percentages,y,'|',ms = 60, color = 'blue')


    ax.axis('off')
    for i, txt in enumerate(labelQ[0::3]):
        ax.annotate(txt, (percentages[labelQ.index(txt)], 0.97), fontsize = 17,rotation=-30)

    for i, txt in enumerate(labelQ[1::3]):
        ax.annotate(txt, (percentages[labelQ.index(txt)], 1.03), fontsize = 17,rotation=30)
     
    for i, txt in enumerate(labelQ[2::3]):
        ax.annotate(txt, (percentages[labelQ.index(txt)], 1.03), fontsize = 17,rotation=30)     
         
         
    #plt.savefig('5_luck_skill_percentage.png')
         
    #plt.close()     
            
            
    plt.show()
    '''









if __name__ == '__main__':  

    if sys.argv[1] == 'fit':
        fit_everything('')
        fit_everything('_release-max')
    elif sys.argv[1] == 'ranking':
        get_rankings()









