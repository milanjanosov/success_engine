import os
import random
import numpy as np
import gzip





input_fields = [ ('music',      'pop'),
                 ('music',      'electro'),
                 ('music',      'classical'),
                 ('music',      'folk'),
                 ('music',      'funk'),
                 ('music',      'jazz'),
                 ('music',      'hiphop'),                   		
                 ('music',      'rock')]  
                # ('film',       'director'),
                 #('film',       'producer'),   
                 #('film',       'writer'),   
                # ('film',       'composer'),   
                # ('film',       'art-director'),   
                # ('book',       'authors') ]








for (field, label) in input_fields:


    #field = 'film'
    #label = 'art-director'


    cnt = 'rating'
    if 'music' in field:
        cnt = 'play'


    Qfolder = 'ProcessedData/ProcessedDataNormalized_no_0/11_log_Q_wout_means/' + field + '_log_Q_wout_mean_' + cnt + '_count_' + label + '.dat'

    outfolder = 'ProcessedData/ProcessedDataNormalized_no_0/13_luck_distribution'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    fout = open(outfolder + '/' + label + '_luck_distr.dat', 'w')

    users_Qs = {}

    for ind, line in enumerate(open(Qfolder)):

        fields = line.strip().split('\t')   

        if len(fields) == 4:

            name, N, Q, Qa = fields


            for line in gzip.open('Data/' + field.title() + '/' + field + '-' + label + '-simple-careers/' + name + '_' + label.replace('-', '_') + '_simple_career.gz'):
                if 'count' not in line:
                    #movie_id, year, rating_value, rating_count, metascore, review_count_user, review_count_critic, gross, opening_weekend = line.strip().split('\t')
                    print line
                    movie_id, rating_count, year = line.strip().split('\t')

                    try:
                        rating_count = float(rating_count)
                        Q = float(Q)
                        fout.write( name + '\t' + str(rating_count) + '\t' + str(Q) + '\t' + str(rating_count/Q) + '\n')
                    except:
                        pass


    fout.close()
