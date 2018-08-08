import os
import sys
import gzip




def preproc_data(infolder, LIMIT, field, label):


    infolder  = '../Data/' + label.title() + '/' + label + '-' + field + '-simple-careers'
    files     = [infolder + '/' + fn for fn in os.listdir(infolder)]   

    id_data = {}

    nnn = len(files)

    for ind, fn in enumerate(files):
        
        imdbid = fn.split('/')[-1].split('_')[0]

        data = []
    
        #print ind, '/', nnn

        #if ind == 1000: break


        try:

            for line in gzip.open(fn):
                if 'year' not in line:

                    line = line.replace(',','')

                    if 'film' == label or 'book' in label:

                        fields = line.strip().split('\t')
                        if 'None' != fields[3] and len(fields[1]) > 0:                      
                            if '-' in fields[1]:
                                year = min([float(f) for f in fields[1].split('-')])
                            else:
                                year = float(fields[1])

                            if float(fields[3]) > 0:
                       
                                data.append( (fields[0], year, float(fields[3])) )

                    elif 'music' in label:
            
                        fields = line.strip().split('\t')
                        if 'None' != fields[2] and len(fields[1]) > 0:                      
                            if '-' in fields[1]:
                                year = min([float(f) for f in fields[1].split('-')])
                            else:
                                year = float(fields[1])

                            if float(fields[2]) > 0:   
                                data.append( (fields[0], year, float(fields[2])) )


            


            if len(data) >= LIMIT:
                id_data[imdbid] = data
            
        except:
            pass


 
    folderout = 'Data/' + field + '/' + label + '-' + field + '-simple-careers-limit-' + str(LIMIT) + '/'
    if not os.path.exists(folderout):
        os.makedirs(folderout)

    for imdbid, data in id_data.items():
        fout = open(folderout + '/' + imdbid + '.dat', 'w')
        for d in data:
            fout.write(str(d[0]) + '\t' + str(d[1]) + '\t' + str(d[2]) + '\n' )
        fout.close()


    print field, label, '\tnumfiles:  ',  len(id_data)







if __name__ == '__main__':  







    labels = ['film', 'music', 'book']

    fields = {'film'  : ['director', 'art-director', 'producer', 'composer', 'writer'],
              'music' : ['electro', 'pop', 'rock', 'funk', 'folk', 'jazz',   'hiphop', 'classical'],
              'book'  : ['authors'] }

    LIMITs = [[5, 10, 15, 20], 
             #[10, 20, 30, 40, 60, 80], 
             [ 60, 80,100], 
              [5, 10, 15, 20]]


    


    


    for ind, label in enumerate(labels):

        if 'music' == label:

            for field in fields[label]:

        
                for LIMIT in LIMITs[ind]:
                    infolder = 'Data/' + label + '-' + field + '-simple-careers'
                    preproc_data(infolder, LIMIT, field, label)
        

