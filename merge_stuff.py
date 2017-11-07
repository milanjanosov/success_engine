
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
                    filename = field + target_file + impact_measure + '_' #+ label + '.dat'
                    
                    for fn in files:
                        if filename in fn:
                            files_to_merge.add(fn)
                        
                    f = open(folder + '/' + filename + '_MERGED.dat', 'w')
                    
                    for fileinput in list(files_to_merge):
                        for line in open(folder + '/' + fileinput):
                            f.write(line)
                    f.close()        
        
            
#merge('_best_product_NN_ranks_all_', 'NN'):
#merge('_career_length_max_', '7_career_length_max_impact')
merge('_', '1_impact_distributions')


