import os
import shutil
import gzip


def create_centrality_careers(ctype, sample, tipusok):



    sam       = ''
    neighbrs  = {}





    if sample: sam = '_sample'

    infolder = 'networks' + sam #+ '/' + ctype + tipus + '_' + str(yearLIMIT)




    for tipus in tipusok:

        outfolder = 'centrality-careers/' + ctype + tipus
        if os.path.exists(outfolder):
            shutil.rmtree(outfolder)
            os.makedirs(outfolder)
        else:
            os.makedirs(outfolder)

            


        files = [infolder + '/' + fo + '/' + 'Q' + ctype + '_' + ctype + tipus +'_NODE_CENTRALITIES_' + fo.split('_')[-1] + '.dat' for fo in os.listdir(infolder) if 'QQ' in fo]
        nnn   = len(files)

        for ind, fn in enumerate(files):


            year = fn.split('_')[-1].replace('.dat', '')

            print ind, '/', nnn

            for line in open(fn):
                if 'between' in line:
                    header = 'year\t' + '\t'.join(line.strip().split('\t')) + '\n'

                else:
                    user     = line.strip().split('\t')[0]
                    record   = year + '\t' + '\t'.join(line.strip().split('\t')[1:]) + '\n'
                    filename = outfolder + '/' + user + '_centrality_career.dat'

                    if not os.path.exists(filename):
                        fout = open(filename, 'w')
                        fout.write(header)
                        fout.write(record)
                        fout.close()
                    else:
                        fout = open(filename, 'a')
                        fout.write(record)
                        fout.close()




def merge_career_types(ctype, tipusok):


    outfolder = 'centrality-success-careers/'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)


    for tipus in tipusok:

        centrality_folder = 'centrality-careers/'  + ctype + tipus
        simple_folder     = 'simple-careers/film-' + ctype + '-simple-careers'


        centr_files  = [centrality_folder + '/' + ff for ff in os.listdir(centrality_folder)]
        simple_files = [simple_folder     + '/' + ff for ff in os.listdir(simple_folder)]

        for fn in  centr_files[0:10]:


            user        = fn.split('/')[-1].split('_')[0]
            simple_file = simple_folder + '/' + user + '_' + ctype + '_simple_career.gz'


            centralities = {}
            for line in open(fn):
                if 'between' in line:
                    header = line.strip()
                else:
                    year   = int(line.strip().split('\t')[0])     
                    record = '\t'.join(line.strip().split('\t')[1:])
                    
                    centralities [year] = record

            
            events = {}
            for line in gzip.open(simple_file):
                if 'rating' in line:
                    header2 = line.strip()
                else:
                    year = line.strip().split('\t')[1]

                events[year] = line



            fout = open(outfolder + user + '_centralities_success.dat', 'w')
            fout.write(header2 + '\t' + header)
            for year, event in events.items():

                print year, event, centralities[year]


            fout.close()

            



ctype     = 'director'
sample    = False
tipusok   = ['-QQ']#, '-QE', '']


create_centrality_careers(ctype, sample, tipusok)
#merge_career_types(ctype, tipusok)



'''
    - here we get the careertrajectories attaching the nw features to each event (movie)
    - next step is to see the correlation between these measures
        - general correlation plot between centr - I*
        - centr(t*) vs I* 
        - centr(t from t0 to tn) vs I* 
              --> central position leads to more successful piece?
              --> you get your big hit bc of the right collaborators (clustering, ratings - good teams? connection between those, triads?)
              --> or you get well-connected after you made your big hit out if nowhere?



source /opt/virtualenv-python2.7/bin/activate


'''





