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
careers    =  ['art_director-20' ] + ['authors-50'] + ['director-10', 'composer-10', 'writer-10', 'producer-10'] + ['electro-80', 'rock-80', 'pop-80', 'jazz-80', 'classical-80', 'funk-80', 'folk-80', 'hiphop-80']


#careers    =  [ 'jazz-80' ]


users = {}





for career in ['electro-80',  'pop-80']:
    print career
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

pop= {}
for line in open('../Music/artist_ids_discogs_pop.dat'):
    if len(line.strip().split('\t')) == 2:
        idd, name = line.strip().split('\t')
        pop[name] = str(idd)




''' =========================================================== '''



music_names = {
    'classical' : classical, 
    'jazz'      : jazz, 
    'hiphop'    : hiphop, 
    'funk'      : funk, 
    'folk'      : folk, 
    'rock'      : rock,
    'pop'       : pop
    }



















for ind, (career, users) in enumerate(users.items()):    
    

    fout = open(folderout + '/' + career + '_time_data.dat', 'w')

 


    for user in users:
        years = []
        maxx  = 0

        #print career, ind
    

        if 2 == 2:
        #try:

            productivity = 0

            try:

                for line in gzip.open(input_fields2[career.split('-')[0].replace('_', '-') ] + '/' + user + '_' + career.split('-')[0].replace('-', '_') + '_simple_career.gz'):

                    if 'year' not in line:
                        productivity += 1
                        try:
                            year   = float(line.strip().split('\t')[1])
                            years.append(year)
                            
                            print line

                            impact = float(line.strip().split('\t')[3])                           
                            if impact > maxx:
                                maxx = impact


                        except:
                            pass


            except:


                try:

                    for line in gzip.open(input_fields2[career.split('-')[0].replace('_', '-') ] + '/' + user + '_' + career.split('-')[0].replace('-', '_') + '_simple_career.dat.gz'):
                        if 'year' not in line:
                            productivity += 1
                            try:
                                year = float(line.strip().split('\t')[2])
                                years.append(year)


                                print career

                                impact = float(line.strip().split('\t')[2])                           
                                if impact > maxx:
                                    maxx = impact

                            except:
                                pass

                except:
                    
          

                    for line in gzip.open(input_fields2[career.split('-')[0].replace('_', '-') ] + '/' + user + '_simple_career.dat.gz'):
                        if 'year' not in line:
                            productivity += 1
                            try:
                                year = float(line.strip().split('\t')[1])
                                years.append(year)
                                print line
                                impact = float(line.strip().split('\t')[2])                           
                                if impact > maxx:
                                    maxx = impact

                            except:
                                pass



            try:

                if career.split('-')[0].replace('_', '-') in music_names:

                    print user    
                    user = music_names[career.split('-')[0].replace('_', '-')][user]
                    print user



                fout.write(user + '\t' + str(min(years)) + '\t' +  str(max(years) - min(years)) + '\t' + str(productivity) + '\t' + str(maxx) + '\n')

            except:
                pass



    fout.close()





