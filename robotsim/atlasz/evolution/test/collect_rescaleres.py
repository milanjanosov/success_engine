import os
import sys



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

    
        if len(runs) < 999:
            print ('Running:\t',  folder, '    ', ''.join((harminc - len(folder))*[' ']),len(runs), '\t')
        else
            print ('DONE:   \t',  folder, '    ', ''.join((harminc - len(folder))*[' ']),len(runs), '\t')

            
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
           



    '''files = [f for f in os.listdir('MLESUCCESS_RES_linrescaled') ]




    outfolder = 'MLESUCCESS_RES_linrescaled_FINAL'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

   
    for fn in files:

        counter = 0.0
        fout    = open(outfolder + '/' + fn.replace('sci_', 'final_sci_'), 'w')

        for line in open('MLESUCCESS_RES_linrescaled/' + fn):

            if 'max' not in line:
                fields = [float(fff) for fff in line.strip().split('\t')]


                maxfitness, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN = fields

                minmu     = min(mu_N, mu_p, mu_Q)
                minsigma  = min(sigma_N, sigma_Q, sigma_p)
                prodsigma = sigma_pQ * sigma_pN * sigma_QN
                maxsigma  = max([abs(sigma_pQ), abs(sigma_pN), abs(sigma_QN)])


                if minmu > 0 and minsigma > 0.005 and abs(prodsigma) > 0.0 and maxsigma < 0.3 and maxabs < 10.0:

                    fout.write(line)
                    counter += 1 
        

        fieldname = fn.split('_')[-1].replace('.dat', '')
        fout.close()



        if fieldname in PREVSTAT['RAW']:
            diff = counter - PREVSTAT['CLEAN'][fieldname]
            if counter < 4:
                print ('CLEAN runs:\t',  fieldname, '    ', ''.join((32 - len(fieldname))*[' ']), int(counter), '\t',  '+' + str(int(diff)))
            else:
                print ('FAIR ENOUGH\t', fieldname, '    ', ''.join((32 - len(fieldname))*[' ']), int(counter))
        else:
            print ('CLEAN runs:\t',  fieldname, '    ', ''.join((32 - len(fieldname))*[' ']), int(counter))


        STSOUT.write('CLEAN\t' + fieldname + '\t' + str(counter) + '\n')
    

    STSOUT.close()
    '''  



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

#        print field.split('_')[-1].replace('.dat'), '\n\n'
        fieldname = field.split('_')[-1].replace('.dat', '')#, d[0])


        for d in data:
           

            fout = open('../../../../QMODELNEW/Qparamfit_linrescaled/' + fieldname + '-qmodel_params.dat', 'w')

            for d in data:
                fout.write('\t'.join(str(aaa) for aaa in d) + '\n')
            fout.close()






if   sys.argv[1] == 'collect': collect()
elif sys.argv[1] == 'export':  export()












