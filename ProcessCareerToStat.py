import os 
import numpy as np
from multiprocessing import Process
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory
from CareerTrajectory.careerTrajectory import MultipleImpactCareerTrajectory
import time
import gzip
import random


def add_max_impact(lista, maxvalue):

    if 'nan' != maxvalue: lista.append(maxvalue)
 
 
def get_dict_data(impacts):

    return [str(year) + '\t' + str(i) for year, impact in impacts.items() for i in impact ]
   

def create_folder(folder):

    if not os.path.exists(folder): os.makedirs(folder)    
    

def add_time_series(impacts, time_series):

    for year, events in time_series.items():
        if year not in impacts:
            impacts[year] = events
        else:
            impacts[year] += events  


def write_yearly_avgs(dict_data, filename):

    if len(dict_data) >  0:
        f = open(filename, 'w')      
        [f.write(str(year) + '\t' + str(np.mean(values)) + '\t' + str(np.std(values)) + '\n' )  for year, values in  dict_data.items()] 
        f.close()


def write_yearly_values(dict_data, filename):

    if len(dict_data) >  0:
        f = open(filename, 'w')      
        [f.write(str(year) + '\t' + '\t'.join([str(v) for v in values]) + '\n' )  for year, values in  dict_data.items()] 
        f.close()


def parse_norm_factors(filename, norm = 1):
                
    norm_factors = {}
    for line in open(filename):   
        fields = line.strip().split('\t')
        norm_factors[int(float(fields[0]))] = float(fields[1]) / norm
                        
    return norm_factors
    
        
def write_distr_data(data, filename):
        
    if len(data) > 0:
        f = open(filename, 'w')
        [f.write(str(a) + '\n') for a in data if a > -100000]
        f.close()
   

def write_pairs(data, filename):

    if len(data) > 0:

        f = open(filename, 'w')
        for d in data:
            if 'nan' != d[0]:
                f.write(str(d[0]) + '\t' + str(d[1]) + '\n')
        f.close()  
        

def write_NN_rank(NNdata_all, NNdata_rand, filename1, filename2, random = False):
             
    mode = 'w'   
    if random:
        if os.path.exists(filename1):
            mode = 'a'


    if len(NNdata_all) > 0:     
    
        f = open(filename1, mode)
        [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NNdata_all]
        f.close() 
        f = open(filename2, mode)
        [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NNdata_rand]
        f.close()      


def chunkIt(seq, num):

    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def process_simple_career_trajectories(args):


    input_data  = args[0]
    normalize   = args[1]
    randomized  = args[2]
    data_folder = args[3]
    impact_measures  = args[4]
    min_rating_count = args[5]
    input_fields     = args[6]
    frac             = args[7]



    


 
    
    for (files, field, label) in [input_data]:

        ijk = 0


        if frac < 1.0:        
            files = random.sample(files, int(len(files) * frac + 100*random.random()    ))


        nnn = len(files)
        #print nnn


        ''' init variables and stuff '''           
        # initialize a dict to store all the impact measures' values
        # the values of the maxes of each individual, the yearly impacts for the inflation curves... etc
        impact_values   = {}
        impact_values_R = {}
        max_impacts     = {}
        yearly_impacts  = {}
        career_lengths  = {}
        best_products_rank_rand = {}
        best_products_rank_all  = {}
        best_products_time      = {}
        best_value_careerlength = {}
        p_without_mean  = {}     
        multi_impacts   = []
        norm_factors    = {}
        log_Q_wout_mean = {}
           
        for impact_measure in impact_measures[field]:
            
            # measures for everyone
            impact_values   [impact_measure] = []
            impact_values_R [impact_measure] = []
            max_impacts     [impact_measure] = []    
            yearly_impacts  [impact_measure] = {}   
            norm_factors    [impact_measure] = {}   
   
            # measures for the random ipact rule
            best_products_rank_rand[impact_measure] = []  
            best_products_rank_all[impact_measure]  = []  
            best_products_time[impact_measure]      = [] 
            best_value_careerlength[impact_measure] = [] 
                    
            # Q model stuff
            career_lengths[impact_measure]  = []  
            p_without_mean[impact_measure]  = []  
            log_Q_wout_mean[impact_measure] = []  


        # read the normalization vectors if we want to work with normalized impact measures     
        '''if 'yearly_avg' in normalize: 
            for impact_measure in impact_measures[field]:
                norm_factors[impact_measure] = parse_norm_factors('ProcessedData/ProcessedDataNormalized_no/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + '.dat' )
   
                #print norm_factors    


        elif 'years_all' in normalize: 

            for impact_measure in impact_measures[field]:
                norm_factors[impact_measure] = np.mean(parse_norm_factors('ProcessedData/ProcessedDataNormalized_yearly_avg/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + '.dat' ).values())

        '''

        '''if 'field_avg' in normalize: 

            for impact_measure in impact_measures[field]:
                #norm_factors[impact_measure] = np.mean(parse_norm_factors('ProcessedData/ProcessedDataNormalized_years_all/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + '.dat' ).values())
                norm_factors[impact_measure] = np.mean(parse_norm_factors('ProcessedData/ProcessedDataNormalized_no/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + '.dat' ).values())
                #print norm_factors    


        elif 'fields_all' in normalize: 


            for impact_measure in impact_measures[field]:

                all_stuff = []
                for (fffiles, fffield, fflabel) in input_fields:

                    if 'music' in fffield:
                        impact_measuree = 'play_count'
                    else:
                        impact_measuree = 'rating_count'
                    all_stuff += parse_norm_factors('ProcessedData/ProcessedDataNormalized_no/6_yearly_averages/' + fffield + '_yearly_average_' + impact_measuree + '_' + fflabel + '.dat' ).values()

                norm_factors[impact_measure] = np.mean(all_stuff)
                

        elif 'yearly_avg' in normalize: 
            for impact_measure in impact_measures[field]:
                norm_factors[impact_measure] = parse_norm_factors('ProcessedData/ProcessedDataNormalized_field_avg/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + '.dat' )
   
                #print norm_factors    


        elif 'years_all' in normalize: 

            for impact_measure in impact_measures[field]:
                norm_factors[impact_measure] = np.mean(parse_norm_factors('ProcessedData/ProcessedDataNormalized_no/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + '.dat' ).values())
        '''

        if 'no' not in normalize: 


            for impact_measure in impact_measures[field]:
           
                all_stuff = []
                for (fffiles, fffield, fflabel) in input_fields:

                    if 'music' in fffield: impact_measuree = 'play_count'
                    else: impact_measuree = 'rating_count'
                    
                    all_stuff += parse_norm_factors('ProcessedData/ProcessedDataNormalized_no/6_yearly_averages/' + fffield + '_yearly_average_' + impact_measuree + '_' + fflabel + '.dat' ).values()

                total_avg = float(np.mean(all_stuff))
                field_avg = np.mean(parse_norm_factors('ProcessedData/ProcessedDataNormalized_no/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + '.dat' ).values())
              
                norm_const = 1.0
                if 'field_avg' in normalize:
                    norm_const = 1.0/field_avg
                elif 'fields_all' in normalize:
                    norm_const = total_avg/field_avg
                elif 'years_all' in normalize:
                    norm_const = total_avg

#                print normalize, norm_const
                norm_factors[impact_measure] = parse_norm_factors('ProcessedData/ProcessedDataNormalized_no/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + '.dat' , norm_const)




        #   print  field, label, normalize, norm_factors, '\n'


        ''' iterate over all the careers and do the job '''
        for filename in files:
                 
            ijk += 1
            #print  ijk, '/', nnn
	    

	    if 'music' == field:            
            career_type = filename.split('_')[2]
	    else:
		career_type = 'simple'
#	    print career_type, filename 
            # calc the stats of theserparated measures
            for impact_measure in impact_measures[field]:
                    
                # construct the career of the individual
                impact_id = 1#impact_measures[field].index(impact_measure)     
                
                if 'music' in field:
                    impact_id = 0


                date_of_birth = 0
                date_of_death = 9999

                if 'book' in field:
                    
                    try:
                        for line in gzip.open('Data/Book/book-authors-simple-profiles/' + filename.split('_')[0] + '_author_bio.dat.gz'):
                            if 'Year_of_birth' in line:
                                dob = int(line.strip().split('\t')[1])
                            if 'Year_of_death' in line:
                                dod = int(line.strip().split('\t')[1])
                        
                        if dob > 0:
                            date_of_birth = dob
                        if dod > 0:
                            date_of_death = dod
                    except:
                        pass
                
 

                individuals_career=SimpleCareerTrajectory(filename, data_folder+'/'+field.title()+'/'+field+'-'+label+'-' + career_type + '-careers/'+filename,impact_id, normalize, norm_factors[impact_measure], randomized, min_rating_count, date_of_birth, date_of_death) 
                       
                timestamps = individuals_career.getTimeStamps()                



                if len(timestamps) > 0 and min(timestamps) > 0:

                         
                    # save the value of all impact measures
                    impact_values[impact_measure] += individuals_career.getImpactValues()  
                    career_time_series = individuals_career.getYearlyProducts()
                    add_time_series(yearly_impacts[impact_measure], career_time_series)          
     

                    # get the yearly values for the inflation curves
                    max_impacts  [impact_measure].append(individuals_career.getMaxImpact())                                 
                    career_length = len(individuals_career.getImpactValues()  )
                    career_lengths[impact_measure] .append(career_length)                 
             
                    # do further stats if he is a good one with at least ... products
                    if career_length > 4:

                        individuals_name = individuals_career.name.split('_')[0].split('/')[-1]

                        # get the rank and time of the best product for the random impact rule
                        (NN_all, NN_rand, N) = individuals_career.getRankOfMaxImpact() 
                        if 'nan' not in str(NN_rand):
                            best_products_rank_all[impact_measure]  += [(n, N) for n in NN_all ]
                            best_products_rank_rand[impact_measure] .append((NN_rand, N))                
                    
                        best_products_time[impact_measure].append(individuals_career.getTimeOfTheBest())

                        impact_values_R[impact_measure] += individuals_career.getImpactValues()  
                        

                        # get stuff for the R-model
                        best_value_careerlength[impact_measure].append((individuals_career.getMaxImpact(), career_length))           
                        
                        # getting things for the Qmodel

                        p_without_mean[impact_measure]  += individuals_career.getLogPwithZeroAvg()        
                        log_Q_wout_mean[impact_measure].append(individuals_name + '\t' + str(individuals_career.getLogQ()) )    




            # more than one impact measure is used - for the correlation plots
            # multiimpact_career = MultipleImpactCareerTrajectory(filename,data_folder+'/'+field.title()+'/'+field+'-'+label+'-simple-careers/'+filename, norm_factors.values(), randomized, date_of_birth, date_of_death) 
            # multi_impacts += multiimpact_career.getImpactValues()
         

#        try:
        if 2 == 2:

            avgQ = str(np.mean([float(fff.split('\t')[1]) for fff in log_Q_wout_mean.values()[0]]))
            varQ = str(np.std([float(fff.split('\t')[1]) for fff  in log_Q_wout_mean.values()[0]]))





            avgp = str(np.mean([float(fff) for fff in p_without_mean.values()[0]]))
            varp = str(np.std([float(fff) for fff  in p_without_mean.values()[0]]))



            #print label, avgQ

            if 'nan' not in avgQ and 'nan' not in varQ:

                string = label + '\t' + str(nnn) + '\t' + str(len(log_Q_wout_mean.values()[0])) + '\t' + avgQ + '\t' + varQ + '\t' + avgp + '\t' + varp
                #print string
                fQ = open('ProcessedData/SampleSize_vs_Q.dat', 'a')
                fQ.write(string + '\n')
                fQ.close()
              
  #      except:
   #         pass
          

   
      
        ''' write out the results '''                      
        '''out_root = 'ProcessedData/ProcessedData'
        if normalize:  out_root = out_root + 'Normalized_' + normalize
        if randomized: out_root = out_root + 'Randomized'            
            
                                                           
        for impact_measure in impact_measures[field]:
            
            # write impact measures
            if 'simple' in career_type:
                extra = ''
            else:
                extra = '_' + career_type

            #if randomized: extra = '_' + str(round(time.time(), 5))
            filename = out_root + '/1_impact_distributions/' + field + '_' + impact_measure + '_dist_' + label + extra + '.dat'
            write_distr_data(impact_values[impact_measure], filename)
        

            # normalizing factors
            filename = out_root + '/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + extra + '.dat' 
            write_yearly_avgs(yearly_impacts[impact_measure],  filename)
        
            filename = out_root + '/12_yearly_values/' + field + 'yearly_values' + impact_measure + '_' + label + extra + '.dat' 
            write_yearly_values(yearly_impacts[impact_measure],  filename)


          
            # write max values
            filename = out_root + '/2_max_impact_distributions/' + field + '_max_' + impact_measure + '_dist_' + label + extra + '.dat'       
            write_distr_data(max_impacts[impact_measure], filename)
        
            # inflation curves
            filename = out_root + '/3_inflation_curves/' + field + '_yearly_' + impact_measure + '_dist_' + label + extra + '.dat'       
            write_distr_data(get_dict_data(yearly_impacts[impact_measure]), filename)
                  
            # rank of the best products
            filename1 = out_root + '/4_NN_rank_N/' + field + '_best_product_NN_ranks_all_' + impact_measure + '_' + label + extra + '.dat'
            filename2 = out_root + '/4_NN_rank_N/' + field + '_best_product_NN_ranks_rand_'+ impact_measure + '_' + label + extra + '.dat'                                                
            write_NN_rank(best_products_rank_all[impact_measure], best_products_rank_rand[impact_measure], filename1, filename2, random = randomized)

            # time of the best product
            filename = out_root + '/5_time_of_the_best/' + field + '_time_of_the_best_'+ impact_measure + '_' + label + extra + '.dat'
            write_distr_data(best_products_time[impact_measure], filename)
            
            # career length and max impact for testing the r-model
            filename = out_root + '/7_career_length_max_impact/' + field + '_' + impact_measure + '_dist_' + label + extra + '.dat'
            write_distr_data(impact_values_R[impact_measure], filename)

            filename = out_root + '/7_career_length_max_impact/' + field + '_career_length_max_' + impact_measure + '_' + label + extra + '.dat'
            write_pairs(best_value_careerlength[impact_measure], filename)

            # career length distribution
            filename = out_root + '/8_career_length/'  + field + '_career_length_' + impact_measure + '_' + label + extra + '.dat'
            write_distr_data(career_lengths[impact_measure], filename)
            
            # the distribution of p - mu_p in the impact = pQ formula
            filename = out_root + '/9_p_without_avg/' + field + '_p_without_mean_' + impact_measure + '_' + label + extra + '.dat'
            write_distr_data(p_without_mean[impact_measure], filename)
            
            # write out multiple impact data
            filename = out_root + '/10_multiple_impacts/' + field + '_multiple_impacts_'  + label + extra + '.dat'
            write_distr_data(multi_impacts, filename)
            
            # write out the logQ_i + mu_p
            filename = out_root + '/11_log_Q_wout_means/' + field + '_log_Q_wout_mean_' + impact_measure + '_'  + label + extra + '.dat'
            write_distr_data(log_Q_wout_mean[impact_measure], filename)
        '''  

     
     
def process_fields(min_rating_count, normalize, frac, randomized):

    data_folder = 'Data'     
     
    impact_measures = {'film'     : ['rating_count'],#, 'average_rating',  'metascore', 'critic_reviews', 'user_reviews', 'gross_revenue'],
                       'music'    : ['play_count'  ], #,
                       'book'     : ['rating_count']}#'average_rating',, 'edition_count']  }

 
        
    '''input_fields = [(os.listdir(data_folder + '/Music/music-pop-simple-careers'),          'music',      'pop'),
                    (os.listdir(data_folder + '/Music/music-jazz-simple-careers'),         'music',      'jazz'),
                    (os.listdir(data_folder + '/Music/music-jazz-release-max-careers'),     'music',      'jazz') ]    


    input_fields = [(os.listdir(data_folder + '/Music/music-pop-simple-careers'),          'music',      'pop'),
                    (os.listdir(data_folder + '/Music/music-electro-simple-careers'),      'music',      'electro'),
                    (os.listdir(data_folder + '/Music/music-classical-simple-careers'),    'music',      'classical'),
                    (os.listdir(data_folder + '/Music/music-folk-simple-careers'),         'music',      'folk'),
                    (os.listdir(data_folder + '/Music/music-funk-simple-careers'),         'music',      'funk'),
                    (os.listdir(data_folder + '/Music/music-jazz-simple-careers'),         'music',      'jazz'),
                    (os.listdir(data_folder + '/Music/music-hiphop-simple-careers'),       'music',      'hiphop'),                   		
                    (os.listdir(data_folder + '/Music/music-rock-simple-careers'),         'music',      'rock') ]
    

    '''


    input_fields = [(os.listdir(data_folder + '/Music/music-pop-simple-careers'),          'music',      'pop'),
                    (os.listdir(data_folder + '/Music/music-electro-simple-careers'),      'music',      'electro'),
                    (os.listdir(data_folder + '/Music/music-classical-simple-careers'),    'music',      'classical'),
                    (os.listdir(data_folder + '/Music/music-folk-simple-careers'),         'music',      'folk'),
                    (os.listdir(data_folder + '/Music/music-funk-simple-careers'),         'music',      'funk'),
                    (os.listdir(data_folder + '/Music/music-jazz-simple-careers'),         'music',      'jazz'),
                    (os.listdir(data_folder + '/Music/music-hiphop-simple-careers'),       'music',      'hiphop'),                   		
                    (os.listdir(data_folder + '/Music/music-rock-simple-careers'),         'music',      'rock'),  
                    (os.listdir(data_folder + '/Film/film-director-simple-careers'),       'film',       'director'),
                    (os.listdir(data_folder + '/Film/film-producer-simple-careers'),       'film',       'producer'),   
                    (os.listdir(data_folder + '/Film/film-writer-simple-careers'),         'film',       'writer'),   
                    (os.listdir(data_folder + '/Film/film-composer-simple-careers'),       'film',       'composer'),   
                    (os.listdir(data_folder + '/Film/film-art-director-simple-careers'),   'film',       'art-director'),   
                    (os.listdir(data_folder + '/Book/book-authors-simple-careers'),        'book',       'authors') ]



    


    '''input_fields = [(os.listdir(data_folder + '/Music/music-pop-simple-careers'),          'music',      'pop'),      
                    (os.listdir(data_folder + '/Film/film-composer-simple-careers'),       'film',       'composer'),   
                    (os.listdir(data_folder + '/Book/book-authors-simple-careers'),        'book',       'authors') ]
    



    input_fields = [
                    (os.listdir(data_folder + '/Film/film-director-simple-careers'),       'film',       'director'),
                    (os.listdir(data_folder + '/Film/film-producer-simple-careers'),       'film',       'producer'),   
                    (os.listdir(data_folder + '/Film/film-writer-simple-careers'),         'film',       'writer'),   
                    (os.listdir(data_folder + '/Film/film-composer-simple-careers'),       'film',       'composer'),   
                    (os.listdir(data_folder + '/Film/film-art-director-simple-careers'),   'film',       'art-director')]#,   
                    #(os.listdir(data_folder + '/Book/book-authors-simple-careers'),        'book',       'authors') ]
    '''








    out_root = 'ProcessedData/ProcessedData'
    if normalize:  out_root = out_root + 'Normalized_' + normalize
    if randomized: out_root = out_root + 'Randomized'   

    #if 'fields_all' == normalize:         
    create_folder(out_root + '/2_max_impact_distributions')    
    create_folder(out_root + '/3_inflation_curves')    
    create_folder(out_root + '/4_NN_rank_N')    
    create_folder(out_root + '/5_time_of_the_best')    
    create_folder(out_root + '/7_career_length_max_impact')                
    create_folder(out_root + '/8_career_length')                            
    create_folder(out_root + '/9_p_without_avg')                                        
    create_folder(out_root + '/10_multiple_impacts')  
    create_folder(out_root + '/11_log_Q_wout_means')  
         
         
               
    create_folder(out_root + '/1_impact_distributions')    
    create_folder(out_root + '/6_yearly_averages')    
    create_folder(out_root + '/12_yearly_values')    
  

    Pros = []
    
    for inp in input_fields:
        p = Process(target = process_simple_career_trajectories, args=([inp, normalize, randomized, data_folder, impact_measures, min_rating_count, input_fields, frac], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()


        
if __name__ == '__main__':  

   

    min_rating_count = 0      


#    process_fields(min_rating_count, normalize = False, randomized = False)
#    process_fields(min_rating_count, normalized = True,  randomized = False)

    f = open('ProcessedData/SampleSize_vs_Q.dat', 'w')



    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for frac in fractions:

        for i in range(int(3/frac)):

            print frac, i

            process_fields(min_rating_count, normalize = 'no', frac = frac,   randomized = False )   
      


    '''process_fields(min_rating_count, normalize = 'yearly_avg',  randomized = False )
    process_fields(min_rating_count, normalize = 'field_avg' ,  randomized = False )     
    process_fields(min_rating_count, normalize = 'fields_all',  randomized = False )     
    process_fields(min_rating_count, normalize = 'years_all' ,  randomized = False )   

    for i in range(100):
        process_fields(min_rating_count, normalize = 'no',          randomized = True )   
        process_fields(min_rating_count, normalize = 'yearly_avg',  randomized = True )
        process_fields(min_rating_count, normalize = 'field_avg' ,  randomized = True )     
        process_fields(min_rating_count, normalize = 'fields_all',  randomized = True )     
        process_fields(min_rating_count, normalize = 'years_all' ,  randomized = True )   
    '''   
  

