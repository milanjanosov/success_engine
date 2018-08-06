import os




folders   = ['Results/' + fn for fn in os.listdir('Results')]




for folder in folders:


    runs  = [ folder + '/' + f for f in os.listdir(folder)]
    field = folder.split('_',1)[1]    


    fields_results = []

    print field, len(runs)
