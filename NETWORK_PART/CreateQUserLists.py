import os
import gzip
root  = 'simple-careers/'
types = ['producer', 'director', 'composer', 'art-director', 'writer']

for ctype in types:
    

    if not os.path.exists('users_types'):
        os.makedirs('users_types')


    fout = open('users_types/Q' + ctype + '_namelist.dat', 'w')
    
    files = os.listdir(root + 'film-' + ctype + '-simple-careers')
    n = len(files)



    for ind, fn in enumerate(files):

        #if ind == 10: break

        print ind, '/', n
        with gzip.open(root + 'film-' + ctype + '-simple-careers/' + fn) as myf:


            #length= len([ll for ll in myf.read().strip().split('\n') if 'movie_id' not in ll])
            #length = len([ll for ll in myf.read().strip().split('\n') if ('movie_id' not in ll and float(ll.strip().split('\t')[3]) > 0.0)])

            length = 0

            for ll in myf.read().strip().split('\n'):

                if 'movie_id' not in ll:
    
                    rating = 0.0

                    try:
                        rating = float(ll.strip().split('\t')[3]) 

                    except:
                        pass

            
                    if rating > 0.0:
                        length += 1

            
       
 
        if length > 4:
            fout.write(fn.split('_')[0]+'\n')
            
    fout.close()    
   
