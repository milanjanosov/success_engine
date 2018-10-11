import os



folders = [fff for fff in os.listdir('./') if 'sci_' in fff and 'all' not in fff]



if os.path.exists('opt_status_report.dat'):

    PREVSTAT          = {}
    PREVSTAT['RAW']   = {}
    PREVSTAT['CLEAN'] = {}

    for line in open('opt_status_report.dat'):
        fields = line.strip().split('\t')
        PREVSTAT[fields[0]][fields[1]] = int(fields[2].replace('.0', ''))


  
STSOUT = open('opt_status_report.dat', 'w')



outfolder = 'MLESUCCESS_RES'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

for folder in folders:


    outfile = open(outfolder + '/Genetic_results_sci_' + folder.split('_', 1)[-1] + '.dat', 'w')


    outfile.write('\t'.join(['maxfitness', 'mu_N', 'mu_p', 'mu_Q', 'sigma_N', 'sigma_Q', 'sigma_p', 'sigma_pQ', 'sigma_pN', 'sigma_QN']) + '\n')


    runs = [folder + '/' + run for run in os.listdir(folder) if '.dat' not in run]

    if PREVSTAT:
    #if os.path.exists('opt_status_report.dat'):
        diff = len(runs) - PREVSTAT['RAW'][folder.replace('sci_', '')]
        print 'RAW runs:\t',  folder, '    ', ''.join((32 - len(folder))*[' ']),len(runs), '\t',  '+' + str(diff)
    else:
        print 'RAW runs:\t',  folder, '    ', ''.join((32 - len(folder))*[' ']),len(runs), '\t', 

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



print '\n\n'
    



files = [f for f in os.listdir('MLESUCCESS_RES') if 'sci_' in f]




outfolder = 'MLESUCCESS_RES_FINAL'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)


for fn in files:

    counter = 0.0
    fout    = open(outfolder + '/' + fn.replace('sci_', 'final_sci_'), 'w')

    for line in open('MLESUCCESS_RES/' + fn):

        if 'max' not in line:
            fields = [float(fff) for fff in line.strip().split('\t')]

            maxfitness, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN = fields

            minmu     = min(mu_N, mu_p, mu_Q)
            minsigma  = min(sigma_N, sigma_Q, sigma_p)
            prodsigma = sigma_pQ * sigma_pN * sigma_QN
            maxsigma  = max([abs(sigma_pQ), abs(sigma_pN), abs(sigma_QN)])


            if minmu > 0 and minsigma > 0.001 and abs(prodsigma) > 0.0 and maxsigma < 0.15:
                fout.write(line)
                counter += 1 
    

    fieldname = fn.split('_sci_')[1].replace('.dat', '')
    fout.close()



    if PREVSTAT:
        diff = counter - PREVSTAT['CLEAN'][fieldname]
        print 'CLEAN runs:\t',  fieldname, '    ', ''.join((32 - len(fieldname))*[' ']), int(counter), '\t',  '+' + str(int(diff))
    else:
        print 'CLEAN runs:\t',  fieldname, '    ', ''.join((32 - len(fieldname))*[' ']), int(counter)


    STSOUT.write('CLEAN\t' + fieldname + '\t' + str(counter) + '\n')


STSOUT.close()












