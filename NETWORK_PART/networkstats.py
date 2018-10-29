import os


ctype   = 'director'
folders = [f for f in os.listdir('networks/' + ctype) if ctype + 'ALL' in f]
data    = []


for f in folders:

    year  = int(f.split('_')[1])
    files = os.listdir('networks/' + ctype + '/' + f)
    nodes = 0
    edges = 0


    for fn in files:

        if 'node_list' in fn:
            nodes = len([line for line in open('networks/' + ctype + '/' + f + '/' + fn)])
        elif 'edges_list' in fn and 'gephi'  in fn:
            edges = len([line for line in open('networks/' + ctype + '/' + f + '/' + fn)])


    data.append((year, nodes, edges))
        
            
data.sort(key=lambda tup: tup[0])


for (y, n, e) in data:
    print y, '\t', n, '\t', e
