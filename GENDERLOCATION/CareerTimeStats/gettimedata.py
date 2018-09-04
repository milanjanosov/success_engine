import sys
import os
import gzip



data_folder  = '../../Data'     
input_fields = [(data_folder + '/Film/film-art-director-simple-careers',   'film',       'art-director'),   
                (data_folder + '/Music/music-pop-simple-careers',          'music',      'pop'),
                (data_folder + '/Music/music-electro-simple-careers',      'music',      'electro'),
                (data_folder + '/Music/music-classical-simple-careers',    'music',      'classical'),
                (data_folder + '/Music/music-folk-simple-careers',         'music',      'folk'),
                (data_folder + '/Music/music-funk-simple-careers',         'music',      'funk'),
                (data_folder + '/Music/music-jazz-simple-careers',         'music',      'jazz'),
                (data_folder + '/Music/music-hiphop-simple-careers',       'music',      'hiphop'),                   		
                (data_folder + '/Music/music-rock-simple-careers',         'music',      'rock'),  
                (data_folder + '/Film/film-director-simple-careers',       'film',       'director'),
                (data_folder + '/Film/film-producer-simple-careers',       'film',       'producer'),   
                (data_folder + '/Film/film-writer-simple-careers',         'film',       'writer'),   
                (data_folder + '/Film/film-composer-simple-careers',       'film',       'composer'),   
                (data_folder + '/Book/book-authors-simple-careers',        'book',       'authors') ]



input_fields2 = {}
for (folder, field, label)  in input_fields:
    input_fields2[label] = folder
    




rootfolder =  '../../QMODELNEW/pQData'
careers    =  ['director-10', 'art_director-20', 'composer-10', 'writer-10', 'producer-10'] + ['electro-80', 'rock-80', 'pop-80', 'jazz-80', 'classical-80', 'funk-80', 'folk-80', 'hiphop-80']+  ['authors-50']


#careers    =  [ 'funk-80']


users = {}


for career in careers:
    for line in open(rootfolder + '/Q_distribution_' + career + '_0.dat'):
        users[career] = [line.strip().split('\t')[0] for ind, line in enumerate(open(rootfolder + '/Q_distribution_' + career + '_0.dat'))]



folderout = 'UsersTimeData'
if not os.path.exists(folderout):
    os.makedirs(folderout)




''' =========================================================== '''
'''                  map musician names to ids                  '''


classical = {}
for line in open('../Music/artist_ids_discogs_classical.dat'):
    idd, name = line.strip().split('\t')
    classical[name] = str(idd)

jazz = {}
for line in open('../Music/artist_ids_discogs_jazz.dat'):
    idd, name = line.strip().split('\t')
    jazz[name] = str(idd)

hiphop = {}
for line in open('../Music/artist_ids_discogs_hiphop.dat'):
    idd, name = line.strip().split('\t')
    hiphop[name] = str(idd)

funk = {}
for line in open('../Music/artist_ids_discogs_funk.dat'):
    idd, name = line.strip().split('\t')
    funk[name] = str(idd)

folk = {}
for line in open('../Music/artist_ids_discogs_folk.dat'):
    idd, name = line.strip().split('\t')
    folk[name] = str(idd)

rock= {}
for line in open('../Music/artist_ids_discogs_rock.dat'):
    if len(line.strip().split('\t')) == 2:
        idd, name = line.strip().split('\t')
        rock[name] = str(idd)


''' =========================================================== '''























for ind, (career, users) in enumerate(users.items()):    
    

    fout = open(folderout + '/' + career + '_time_data.dat', 'w')

 


    for user in users:
        years = []


        #print career, ind
    

        if 2 == 2:
        #try:

            productivity = 0

            try:

                for line in gzip.open(input_fields2[career.split('-')[0].replace('_', '-') ] + '/' + user + '_' + career.split('-')[0].replace('-', '_') + '_simple_career.dat.gz'):
                    if 'year' not in line:
                        productivity += 1
                        try:
                            year = float(line.strip().split('\t')[1])
                            years.append(year)
                        except:
                            pass

            except:

                for line in gzip.open(input_fields2[career.split('-')[0].replace('_', '-') ] + '/' + user + '_' + career.split('-')[0].replace('-', '_') + '_simple_career.dat.gz'):
                    if 'year' not in line:
                        productivity += 1
                        try:
                            year = float(line.strip().split('\t')[1])
                            years.append(year)
                        except:
                            pass



            fout.write(user + '\t' + str(min(years)) + '\t' +  str(max(years) - min(years)) + '\t' + str(productivity) + '\n')

        #except:
        #    pass

    fout.close()





