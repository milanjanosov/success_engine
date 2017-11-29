
import os



def merge(target_file, target_folder):

    root = 'ProcessedData'
    metafolders = os.listdir(root) 

    impact_measures = {'film' : ['average_rating', 'rating_count', 'metascore', 'critic_reviews', 'user_reviews', 'gross_revenue'],
                       'music': ['play_count'],
                       'book' : ['average_rating', 'rating_count', 'edition_count']}


    for metafolder in metafolders:

        folders = [root+ '/' + metafolder + '/' + fname for fname in os.listdir(root + '/' + metafolder) if target_folder  in fname]
        
        for folder in folders:
            
            files = os.listdir(folder)
     
            for field, measures in impact_measures.items():
     
      
     
                for impact_measure in measures:
                

                    files_to_merge = set()
                    if '_' == target_file or '_yearly_' in target_file:
                        filename = field + target_file + impact_measure + '_dist_' #+ label + '.dat'
                    else:
                        filename = field + target_file + impact_measure + '_' #+ label + '.dat'
                    

                    print filename
                    for fn in files:
                        if filename in fn:
                            files_to_merge.add(fn)
                       
                    f = open(folder + '/' + filename + '_MERGED.dat', 'w')
                    
                    for fileinput in list(files_to_merge):
                        for line in open(folder + '/' + fileinput):
                            a  =2                            
                            f.write(line)
                    f.close()        
                    
            
#merge('_best_product_NN_ranks_all_', 'NN')
#merge('_career_length_max_', '7_career_length_max_impact')
#merge('_', '1_impact_distributions')
#merge('_yearly_', '3_inflation_curves')
merge('_p_without_mean_', '9_p_without_avg')
#merge('_career_length_', '8_career_length')
#merge('_max_', '2_max_impact_distributions')



