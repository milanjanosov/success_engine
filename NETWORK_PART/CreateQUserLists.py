import os
import gzip


types     = ['producer-10', 'director-10', 'composer-10', 'art-director-20', 'writer-10']
allQnames = []
field     = 'film'
hout      = open('users_types/Q_' + field + '_EVERYONE_namelist.dat'        , 'w')
outf      = 'results/'
files     = os.listdir('users_types')



for ctype in types:
   
    if not os.path.exists('users_types'):
        os.makedirs('users_types')

    fout   = open('users_types/Q_'   + field + '_' + ctype.rsplit('-', 1)[0] + '_namelist.dat' , 'w')
    gout   = open('users_types/ALL_' + field + '_' + ctype.rsplit('-', 1)[0] + '_namelist.dat' , 'w')
    Qnames = [line.strip().split('\t')[0] for line in open('../QMODELNEW/pQData/Q_distribution_' + ctype.replace('art-', 'art_') + '_0.dat')]

    fout.write('\n'.join(Qnames))
    fout.close()


    for ind, fn in enumerate(os.listdir('simple-careers/' + field + '-' + ctype.rsplit('-', 1)[0] + '-simple-careers')):
        gout.write( fn.split('_')[0] + '\n')

    gout.close()    
    
    allQnames += Qnames


hout.write('\n'.join(allQnames))
hout.close()




if not os.path.exists(outf): os.makedirs(outf)

fout = open(outf + 'user_numbers_stats.dat', 'w')
fout.write('========================\nNumber of users with Q params\n\n')

for fn in files:
    if 'ALL' not in fn:
        with open('users_types/' + fn) as mf:
            fout.write( 'Q_' + fn.split('_')[2] + '\t' + str(len(mf.read().strip().split('\n'))) + '\n')

fout.write('\n\n========================\nAll the users\n\n')
for fn in files:
    if 'ALL' in fn:
        with open('users_types/' + fn) as mf:
            fout.write( 'ALL_' + fn.split('_')[2] + '\t' + str(len(mf.read().strip().split('\n'))) + '\n')

fout.close()
