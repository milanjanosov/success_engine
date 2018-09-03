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



users = {}


for career in careers[0:1]:
    for line in open(rootfolder + '/Q_distribution_' + career + '_0.dat'):
        users[career] = [line.strip().split('\t')[0] for ind, line in enumerate(open(rootfolder + '/Q_distribution_' + career + '_0.dat')) if ind < 100]



for career, users in users.items():    
    
    for user in users:
        years = []
        print user
        for line in gzip.open(input_fields2[career.split('-')[0]] + '/' + user + '_' + career.split('-')[0] + '_simple_career.gz'):
            if 'year' not in line:
                year = float(line.strip().split('\t')[0])
                years.append(year)


        print user, min(years), max(years) - min(years)




