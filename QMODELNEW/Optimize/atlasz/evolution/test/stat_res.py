import os




folders   = ['Results/' + fn for fn in os.listdir('Results')]




for folder in folders:


    runs  = [ folder + '/' + f for f in os.listdir(folder)]
    field = folder.split('_',1)[1]    


    fields_results = []


    if len(field) < 8:
        field = field + '   '
    if len(field) < len('art_director-20'):
        field  = field + '      '


    print field, '\t', len(runs)

