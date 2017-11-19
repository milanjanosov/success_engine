import os 
import numpy as np
from multiprocessing import Process
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory
from CareerTrajectory.careerTrajectory import MultipleImpactCareerTrajectory



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


def parse_norm_factors(filename):
                
    norm_factors = {}
    for line in open(filename):   
        fields = line.strip().split('\t')
        norm_factors[float(fields[0])] = float(fields[1])
                        
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
        

def write_NN_rank(NNdata_all, NNdata_rand, filename1, filename2):
                
    if len(NNdata_all) > 0:     
    
        f = open(filename1, 'w')
        [f.write(str(a[0]) + '\t' + str(a[1]) + '\n') for a in NNdata_all]
        f.close() 
        f = open(filename2, 'w')
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
    normalized  = args[1]
    randomized  = args[2]
    data_folder = args[3]
    impact_measures  = args[4]
    min_rating_count = args[5]
 
    
    for (files, field, label) in [input_data]:

        ijk = 0
        nnn = len(files)

        ''' init variables and stuff '''           
        # initialize a dict to store all the impact measures' values
        # the values of the maxes of each individual, the yearly impacts for the inflation curves... etc
        impact_values  = {}
        max_impacts    = {}
        yearly_impacts = {}
        career_lengths = {}
        best_products_rank_rand = {}
        best_products_rank_all  = {}
        best_products_time      = {}
        best_value_careerlength = {}
        p_without_mean = {}     
        multi_impacts  = []
        norm_factors   = {}
           
        for impact_measure in impact_measures[field]:
            
            # measures for everyone
            impact_values [impact_measure] = []
            max_impacts   [impact_measure] = []    
            yearly_impacts[impact_measure] = {}   
            norm_factors  [impact_measure] = {}   
   
            # measures for the random ipact rule
            best_products_rank_rand[impact_measure] = []  
            best_products_rank_all[impact_measure]  = []  
            best_products_time[impact_measure]      = [] 
            best_value_careerlength[impact_measure] = [] 
                    
            # Q model stuff
            career_lengths[impact_measure] = []  
            p_without_mean[impact_measure] = []  


        # read the normalization vectors if we want to work with normalized impact measures     
        if normalized: 
            for impact_measure in impact_measures[field]:
                norm_factors[impact_measure] = parse_norm_factors('ProcessedData/ProcessedData_0/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + '.dat' )
   

        ''' iterate over all the careers and do the job '''
        for filename in files:
                 
            ijk += 1
            print ijk, '/', nnn
            
            
            # calc the stats of theserparated measures
            for impact_measure in impact_measures[field]:
                    
                # construct the career of the individual
                impact_id = impact_measures[field].index(impact_measure)     

                if 'book' in field:
                    print filename, filename.split('_')[0] + '_author_bio.dat'
                    ''' WHEN THERE IS A FUCKING WORKING MACHINE
                        REPARSE GOODREADS PROFILES, GET YOB AND YOD PROPERLY (WTF HAPPENED)
                        AND THAN READ THOSE FILES HERE...
                    '''
       
                date_of_birth = 0
                date_of_death = 9999
                
                individuals_career=SimpleCareerTrajectory(filename, data_folder+'/'+field.title()+'/'+field+'-'+label+'-simple-careers/'+filename,impact_id,norm_factors[impact_measure], randomized, min_rating_count, date_of_birth, date_of_death) 
                           
                # save the value of all impact measures
                impact_values[impact_measure] += individuals_career.getImpactValues()  
                max_impacts  [impact_measure].append(individuals_career.getMaxImpact())  
                
                # get the yearly values for the inflation curves
                career_time_series = individuals_career.getYearlyProducts()
                                
                career_length = len(career_time_series)
                add_time_series(yearly_impacts[impact_measure], career_time_series)
                

                # do further stats if he is a good one with at least ... products
                if career_length > 14:

                    # get the rank and time of the best product for the random impact rule
                    (NN_all, NN_rand, N) = individuals_career.getRankOfMaxImpact() 
                    if 'nan' not in str(NN_rand):
                        best_products_rank_all[impact_measure]  += [(n, N) for n in NN_all ]
                        best_products_rank_rand[impact_measure] .append((NN_rand, N))                
                
                    best_products_time[impact_measure].append(individuals_career.getTimeOfTheBest())

                    # get stuff for the R-model
                    best_value_careerlength[impact_measure].append((individuals_career.getMaxImpact(), career_length))           
                    
                    # getting things for the Qmodel
                    career_lengths[impact_measure] .append(career_length)
                    
                    p_without_mean[impact_measure] += individuals_career.getLogPwithZeroAvg()        


            # more than one impact measure is used - for the correlation plots
            multiimpact_career = MultipleImpactCareerTrajectory(filename,data_folder+'/'+field.title()+'/'+field+'-'+label+'-simple-careers/'+filename, norm_factors.values(), randomized, date_of_birth, date_of_death) 
            multi_impacts += multiimpact_career.getImpactValues()
            
      
        ''' write out the results '''                      
        out_root = 'ProcessedData/ProcessedData_' + str(min_rating_count) 
        if normalized: out_root = out_root + '_Normalized'
        if randomized: out_root = out_root + 'Randomized'            
            
    
        create_folder(out_root + '/1_impact_distributions')    
        create_folder(out_root + '/2_max_impact_distributions')    
        create_folder(out_root + '/3_inflation_curves')    
        create_folder(out_root + '/4_NN_rank_N')    
        create_folder(out_root + '/5_time_of_the_best')    
        create_folder(out_root + '/6_yearly_averages')    
        create_folder(out_root + '/7_career_length_max_impact')                
        create_folder(out_root + '/8_career_length')                            
        create_folder(out_root + '/9_p_without_avg')                                        
        create_folder(out_root + '/10_multiple_impacts')                                                    
        
        
        for impact_measure in impact_measures[field]:
            
            # write impact measures
            filename = out_root + '/1_impact_distributions/' + field + '_' + impact_measure + '_dist_' + label + '.dat'
            write_distr_data(impact_values[impact_measure], filename)
        
            # write max values
            filename = out_root + '/2_max_impact_distributions/' + field + '_max_' + impact_measure + '_dist_' + label + '.dat'       
            write_distr_data(max_impacts[impact_measure], filename)
        
            # inflation curves
            filename = out_root + '/3_inflation_curves/' + field + '_yearly_' + impact_measure + '_dist_' + label + '.dat'       
            write_distr_data(get_dict_data(yearly_impacts[impact_measure]), filename)
                  
            # rank of the best products
            filename1 = out_root + '/4_NN_rank_N/' + field + '_best_product_NN_ranks_all_' + impact_measure + '_' + label + '.dat'
            filename2 = out_root + '/4_NN_rank_N/' + field + '_best_product_NN_ranks_rand_'+ impact_measure + '_' + label + '.dat'                                                
            write_NN_rank(best_products_rank_all[impact_measure], best_products_rank_rand[impact_measure], filename1, filename2)

            # time of the best product
            filename = out_root + '/5_time_of_the_best/' + field + '_time_of_the_best_'+ impact_measure + '_' + label + '.dat'
            write_distr_data(best_products_time[impact_measure], filename)
            
            # normalizing factors
            filename = out_root + '/6_yearly_averages/' + field + '_yearly_average_' + impact_measure + '_' + label + '.dat' 
            write_yearly_avgs(yearly_impacts[impact_measure],  filename)
        
            # career length and max impact for testing the r-model
            filename = out_root + '/7_career_length_max_impact/' + field + '_career_length_max_' + impact_measure + '_' + label + '.dat'
            write_pairs(best_value_careerlength[impact_measure], filename)

            # career length distribution
            filename = out_root + '/8_career_length/'  + field + '_career_length_' + impact_measure + '_' + label + '.dat'
            write_distr_data(career_lengths[impact_measure], filename)
            
            # the distribution of p - mu_p in the impact = pQ formula
            filename = out_root + '/9_p_without_avg/' + field + '_p_without_mean_' + impact_measure + '_' + label + '.dat'
            write_distr_data(p_without_mean[impact_measure], filename)
            
            # write out multiple impact data
            filename = out_root + '/10_multiple_impacts/' + field + '_multiple_impacts_'  + label + '.dat'
            write_distr_data(multi_impacts, filename)
            
     
def process_fields(min_rating_count, normalized, randomized):

    data_folder = 'Data'     
     
    impact_measures = {'film' : ['average_rating', 'rating_count', 'metascore', 'critic_reviews', 'user_reviews', 'gross_revenue'],
                       'music': ['play_count'],
                       'book' : ['average_rating', 'rating_count', 'edition_count']}

 
    input_fields = [(os.listdir(data_folder + '/Music/music-pop-simple-careers'),         'music', 'pop'),
                    (os.listdir(data_folder + '/Music/music-electro-simple-careers'),     'music', 'electro'),
                    (os.listdir(data_folder + '/Film/film-director-simple-careers'),      'film',  'director'),
                    (os.listdir(data_folder + '/Film/film-producer-simple-careers'),      'film',  'producer'),   
                    (os.listdir(data_folder + '/Film/film-writer-simple-careers'),        'film',  'writer'),   
                    (os.listdir(data_folder + '/Film/film-composer-simple-careers'),      'film',  'composer'),   
                    (os.listdir(data_folder + '/Film/film-art-director-simple-careers'),  'film',  'art-director'),   
                    (os.listdir(data_folder + '/Book/book-authors-simple-careers'),       'book',  'authors') ]


  

    Pros = []
    
    for inp in input_fields:  
        p = Process(target = process_simple_career_trajectories, args=([inp, normalized, randomized, data_folder, impact_measures, min_rating_count], ))
        Pros.append(p)
        p.start()
       
    for t in Pros:
        t.join()


        
if __name__ == '__main__':  

   

    min_rating_count = 0      
    process_fields(min_rating_count, normalized = False, randomized = False)
    process_fields(min_rating_count, normalized = True,  randomized = False)
    #process_fields(min_rating_count, normalized = True,  randomized = True )

    
    

