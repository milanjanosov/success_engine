import json
import os
from collections import Counter
from difflib import SequenceMatcher
import time
from multiprocessing import Process


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_grammies():

    with open('grammyData.json') as myfile:
        data = myfile.read()
        
    jdata = json.loads(data)

    names       = []
    award_types = {}

    fout = open('names.dat', 'w')

    for jd in jdata:
        
        atype = jd['awardType']
        if atype not in award_types:
            award_types[atype] = 1
        else:
            award_types[atype] += 1
            
        names.append(jd['name'] )
      
        fout.write(jd['name'] + '\t' + jd['awardFor']+ '\n')
            
    fout.close()    
   
    #return dict(Counter(names).most_common(20))
    return dict(Counter(names))



def get_names(genre):

    ids_names = {}

    for line in open('../ids/artist_ids_discogs_' + genre + '.dat'):
        fields = line.strip().split('\t')
        if len(fields) > 1:        
            idd  = fields[0]
            name = fields[1]  

            ids_names[idd] = name 

    return ids_names





def process_genre_stuff(args):

    genre    = args[0]
    limits   = args[1]
    nnnG     = args[2]
    grammies = args[3]


    for limit in limits:

        fout = open(folderout + '/' + genre + '_' + str(limit) + '.dat', 'w')


        ijk       = 1
        ids_names = get_names(genre)
        nnnI      = len(ids_names)
        nnn       = nnnG * nnnI / 1000

        for gname in grammies.keys():
            
            for idd, name in ids_names.items():

                print ('BEFORE   ', genre, '\t', ijk/1000, '/', nnn, '\t', gname, idd, name, similar(gname, name))
                ijk += 1
                simscore = similar(gname, name.strip())

                if simscore > limit:

                    fout.write(gname + '\t' + idd + '\t' + name.strip() + '\t' + str(simscore) + '\n')

                    print ('  AFTER   ', gname, idd, name, simscore)
                   # print (name.strip(), idd, gname, simscore)

        fout.close()









folderout = 'NameMatching'
if not os.path.exists(folderout):
    os.makedirs(folderout)



grammies = get_grammies()
genres   = ['electro', 'pop', 'rock', 'folk', 'funk', 'jazz', 'hiphop', 'classical']
nnnG     = len(grammies)
limits   = [0.7, 0.8, 0.9, 0.95]
Pros     = []



for genre in genres:  

    folderout = 'NameMatching/' + genre
    if not os.path.exists(folderout):
        os.makedirs(folderout)

    p = Process(target = process_genre_stuff, args=([genre, limits, nnnG, grammies],))
    Pros.append(p)
    p.start()
   

for t in Pros:
    t.join()







