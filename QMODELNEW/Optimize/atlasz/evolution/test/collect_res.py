import os



folders = [fff for fff in os.listdir('./') if 'mlesuccess_' in fff]



outfolder = 'MLESUCCESS_RES'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

for folder in folders:



    outfile = open(outfolder + '/Genetic_results_' + folder.split('_')[-1] + '.dat', 'w')

    outfile.write('\t'.join(['maxfitness', 'mu_N', 'mu_p', 'mu_Q', 'sigma_N', 'sigma_Q', 'sigma_p', 'sigma_pQ', 'sigma_pN', 'sigma_QN']) + '\n')







    runs = [folder + '/' + run for run in os.listdir(folder) if '.dat' not in run]


    print folder, len(runs)

    if len(runs) > 0:

        for run in runs:

            if len(os.listdir(run)) > 0:

                lastgen = max([ int(fff.split('_')[1].replace('.dat', '')) for fff in  os.listdir(run) ])
                resfile = run + '/Generation_' + str(lastgen) + '.dat'


                best = ''
                maxf = ''   

                for line in open(resfile):
                    if 'max' in line:
                        maxf = line.strip().split('\t')[1]
                    if 'best' in line:
                        best = line.strip().split('\t', 1)[1]


                outfile.write( maxf + '\t' + best + '\n')

    outfile



        
        
