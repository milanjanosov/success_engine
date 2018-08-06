import os




folders   = sorted(['Results/' + fn for fn in os.listdir('Results')])
folderout = 'Qparamfit'

if not os.path.exists(folderout):
    os.makedirs(folderout)




for folder in folders:


    runs  = [ folder + '/' + f for f in os.listdir(folder)]
    field = folder.split('_',1)[1]    


    fields_results = []


    for run in runs:



        generations = [int(f.split('_')[1].replace('.dat','')) for f in  os.listdir(run)]



        if len(generations) > 0:

            maxgenerations = max(generations)

            best = ''
            maxf = ''   

            for line in open(run + '/Generation_' + str(maxgenerations) + '.dat'):

         

                if 'max' in line:
                    maxf = [float(line.strip().split('\t')[1])]
                if 'best' in line:
                    best = [float(ttt) for ttt in line.strip().split('\t', 1)[1].split('\t')]


                    mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN = best
            
                    if (mu_N > 0 and mu_N < 5) and  (mu_p > 0 and mu_p < 5) and  (mu_Q > 0 and mu_Q < 5):

                        if (sigma_N > 0.001 and sigma_N < 5) and  (sigma_p > 0.001 and sigma_p < 5) and (sigma_p > 0.001 and sigma_p < 5):

                            if sigma_pQ != 0 and  sigma_pN != 0 and sigma_QN != 0:

                                if abs(sigma_pQ) < 0.1 and abs(sigma_pN) < 0.1 and abs(sigma_QN) < 0.1:

                                    fields_results.append( tuple(maxf + best) )



    #fields_results = fields_results.sort(key=lambda tup: tup[0])
    fields_results_s = sorted(fields_results, key=lambda tup: tup[0], reverse = True)

    if len(field) < 8:
        field = field + '   '
    if len(field) < len('art_director-20'):
        field  = field + '      '


    print field, '\t', len(fields_results_s)

    field = field.replace(' ', '')

    fout = open(folderout + '/' + field + '_qmodel_params.dat', 'w')


    for f in fields_results_s[0:5]:
        fout.write( '\t'.join([str(ff) for ff in f]) + '\n')
    fout.close()
        





