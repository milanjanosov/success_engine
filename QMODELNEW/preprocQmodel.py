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
    
        print ind, '/', nnn

        #try:

        for line in gzip.open(fn):
            if 'year' not in line:


                if 'film' == label:

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
        
        #except:
        #    pass


 
    folderout = 'Data/' + field + '/' + label + '-' + field + '-simple-careers-limit-' + str(LIMIT) + '/'
    if not os.path.exists(folderout):
        os.makedirs(folderout)

    for imdbid, data in id_data.items():
        fout = open(folderout + '/' + imdbid + '.dat', 'w')
        for d in data:
            fout.write(str(d[0]) + '\t' + str(d[1]) + '\t' + str(d[2]) + '\n' )
        fout.close()


    print 'numfiles  ',  len(id_data)







if __name__ == '__main__':  

    label    = sys.argv[1]
    field    = sys.argv[2]
    LIMIT    = int(sys.argv[3])
    infolder = 'Data/' + label + '-' + field + '-simple-careers'

    preproc_data(infolder, LIMIT, field, label)


