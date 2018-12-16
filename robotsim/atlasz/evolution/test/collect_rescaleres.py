import os
import sys
import math
import numpy as np

def update():


    folders = [fff for fff in os.listdir('Results_linrescaled')]



    if os.path.exists('opt_status_report_linrescaled.dat'):

        PREVSTAT          = {}
        PREVSTAT['RAW']   = {}
        PREVSTAT['CLEAN'] = {}

        for line in open('opt_status_report_linrescaled.dat'):
            fields = line.strip().split('\t')
        
            fieldname = fields[1], fields[1].split('_')[-1]

            PREVSTAT[fields[0]][fieldname] = int(fields[2].replace('.0', ''))


      
    STSOUT = open('opt_status_report_linrescaled.dat', 'w')



    outfolder = 'MLESUCCESS_RES_linrescaled'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    harminc = len('Results_linrescaled/sci_theoretical_computer_science')

    for folder in folders:



        outfile = open(outfolder + '/Genetic_results_' + folder.split('_', 1)[-1] + '.dat', 'w')


        outfile.write('\t'.join(['maxfitness', 'mu_N', 'mu_p', 'mu_Q', 'sigma_N', 'sigma_Q', 'sigma_p', 'sigma_pQ', 'sigma_pN', 'sigma_QN']) + '\n')

        folder = 'Results_linrescaled/' + folder
        runs = [folder + '/' + run for run in os.listdir(folder) if '.dat' not in run]

        fieldname = folder.replace('sci_', '')

    
        if len(runs) == 1000 or len(runs) == 500:
            print ('DONE:   \t',  folder, '    ', ''.join((harminc - len(folder))*[' ']),len(runs), '\t')

        else:
            print ('Running:\t',  folder, '    ', ''.join((harminc - len(folder))*[' ']),len(runs), '\t')
        


            
            STSOUT.write('RAW\t' + folder.replace('sci_', '') + '\t' + str(len(runs)) + '\n')

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

            outfile.close()
            


    print ('\n\n')
           

    STSOUT.close()

   

def collect():


    folders = [fff for fff in os.listdir('Results_linrescaled')]



    if os.path.exists('opt_status_report_linrescaled.dat'):

        PREVSTAT          = {}
        PREVSTAT['RAW']   = {}
        PREVSTAT['CLEAN'] = {}

        for line in open('opt_status_report_linrescaled.dat'):
            fields = line.strip().split('\t')
        
            fieldname = fields[1], fields[1].split('_')[-1]

            PREVSTAT[fields[0]][fieldname] = int(fields[2].replace('.0', ''))


      
    STSOUT = open('opt_status_report_linrescaled.dat', 'w')



    outfolder = 'MLESUCCESS_RES_linrescaled'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    harminc = len('Results_linrescaled/sci_theoretical_computer_science')

    for folder in folders:

        outfile = open(outfolder + '/Genetic_results_' + folder.split('_', 1)[-1] + '.dat', 'w')


        outfile.write('\t'.join(['maxfitness', 'mu_N', 'mu_p', 'mu_Q', 'sigma_N', 'sigma_Q', 'sigma_p', 'sigma_pQ', 'sigma_pN', 'sigma_QN']) + '\n')

        folder = 'Results_linrescaled/' + folder
        runs = [folder + '/' + run for run in os.listdir(folder) if '.dat' not in run]

        fieldname = folder.replace('sci_', '')

    
        if len(runs) == 1000 or len(runs) == 500:
            print ('DONE:   \t',  folder, '    ', ''.join((harminc - len(folder))*[' ']),len(runs), '\t')

        else:
            print ('Running:\t',  folder, '    ', ''.join((harminc - len(folder))*[' ']),len(runs), '\t')
        
            
        STSOUT.write('RAW\t' + folder.replace('sci_', '') + '\t' + str(len(runs)) + '\n')

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

        outfile.close()
           
    print ('\n\n')
         
    STSOUT.close()

  




def export():


    outfolder = 'MLESUCCESS_RES_linrescaled_FINAL'

    if not os.path.exists('../../../../QMODELNEW/Qparamfit_linrescaled/'): os.makedirs('../../../../QMODELNEW/Qparamfit_linrescaled/')

    final_files = os.listdir(outfolder)
    final_data  = {}

    for ffn in final_files:

        final_data[ffn] = []

        for line in open(outfolder + '/' + ffn):

            fields = tuple([float(fff) for fff in line.strip().split('\t')])
            final_data[ffn].append(fields)


    for field, data in final_data.items():

        data.sort(key=lambda tup: tup[0], reverse = True)   
        fieldname = field.split('_')[-1].replace('.dat', '')#, d[0])

        for d in data:
           
            fout = open('../../../../QMODELNEW/Qparamfit_linrescaled/' + fieldname + '-qmodel_params.dat', 'w')

            for d in data:
                fout.write('\t'.join(str(aaa) for aaa in d) + '\n')
            fout.close()





def results():

    measures_N = {}
    for line in open('MLESUCCESS_RES_linrescaled_GROUNDTRUTH.dat'):
        fieldd, avg, std = line.strip().split(' ')
        measures_N[fieldd.replace('-simple-careers', '').split('-')[-1]] = (avg, std)

    ### get things started
    root     = ''
    folderin = 'MLESUCCESS_RES_linrescaled/'
    fields   = [(f.replace('.dat', '').split('_', 2)[-1]) for f in os.listdir(root + folderin)]

            
    ### collect results and drop wrong optimizations
    results = {}
    for field in fields:
        
        
        sss = []
        results[field] = []

        for line in open('MLESUCCESS_RES_linrescaled/Genetic_results_' + field + '.dat'):
        
            if 'maxfit' not in line:   
                maxfitness, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN = [float(aaa) for aaa in line.strip().split('\t')]
                
                sigma_N = math.sqrt(sigma_N)
                sigma_Q = math.sqrt(sigma_Q)
                sigma_p = math.sqrt(sigma_p)
                
                records = [maxfitness, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN]
                
                minsigma  = min(sigma_N, sigma_Q, sigma_p)
                prodsigma = sigma_pQ * sigma_pN * sigma_QN
                maxsigma  = max([abs(sigma_pQ), abs(sigma_pN), abs(sigma_QN)])
                maxabs    = max([abs(a) for a in records[1:]])   

                
                if min([abs(r) for r in records]) > 0 and mu_N > 1 and abs(prodsigma) > 0.0 and maxsigma < 0.25 and minsigma > 0.001 and maxabs < 10.0:
                #if min([abs(r) for r in records]) > 0:    
                                
                    cov1 = abs(sigma_Q * sigma_p / sigma_pQ)
                    cov2 = abs(sigma_N * sigma_p / sigma_pN)
                    cov3 = abs(sigma_Q * sigma_N / sigma_QN)
                    
                    mu_N_meas = float(measures_N[field][0])
                    
                    
                    if min (cov1, cov2, cov3) > 3 and mu_N < 10 and (mu_N / mu_N_meas < 1.5):
                        records += [cov1, cov2, cov3]
                        results[field].append(records)      
                        
                        #print field, sigma_Q / (sigma_Q + sigma_p)

    
        
     
    names = ['maxfitness', 'mu_N', 'mu_p', 'mu_Q', 'sigma_N', 'sigma_Q', 'sigma_p', 'sigma_pQ', 'sigma_pN', 'sigma_QN']
    names = [n + ''.join(int(14-len(n))*[' '])  for n in names]

    maxlen = max([len(fgh.split('_')[0]) for fgh in fields ])

    folderout = root + 'MLESUCCESS_RES_linrescaled_ALL/'
    if not os.path.exists(folderout):
        os.makedirs(folderout)    
        
    for field, data in results.items():
        data.sort(key=lambda tup: tup[0], reverse = True)    
        
        
    Fsssq = {}
    Fsssp = {}
    for field, data in results.items():
        
        ffout = open(folderout + 'Genetic_results_' + field + '.dat', 'w')
        ffout.write('\t'.join(names) + '\n')
        
        sss = []
        sssq = []
        sssp = []

        
        print field.split('_')[0], ''.join((maxlen - len(field.split('_')[0]))*[' ']), len(data)
        
        for dind, d in enumerate(data):
            
            ffout.write('\t'.join([str(dd) for dd in d]) + '\n')
            if 'maxf' not in str(d[0]):
                sigma_Q = d[5]
                sigma_p = d[6] 
                sss.append(sigma_Q / (sigma_Q + sigma_p)) 
                sssq.append(sigma_Q)
                sssp.append(sigma_p)
                
            Fsssq[field] = np.mean(sssq)
            Fsssp[field] = np.mean(sssp)    


            
            if dind == 5: break
                
        #print field.split('_')[0], len(data), '\t',  round(np.mean(sss), 2)
        
        ffout.close()







if   sys.argv[1] == 'collect':  collect()
elif sys.argv[1] == 'update':   update()
elif sys.argv[1] == 'export':   export()
elif sys.argv[1] == 'results':  results()











