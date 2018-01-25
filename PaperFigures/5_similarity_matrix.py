execfile("0_imports.py")



def get_sim_matrix(var, mode):


    fields = [ 'director', 'producer', 'writer', 'composer', 'art-director', 'electro', 'pop', 'rock', 'funk', 'folk', 'classical', 'jazz', 'hiphop', 'authors']
    fout   = open('ResultData/5_pQ_fit/SIM_matrix_' + var + '_' + mode + '.dat', 'w')

    for label1 in fields:

        xp1, yp1 = zip(*[tuple(float(ijk) for ijk in line.strip().split('\t'))  for line in open('ResultData/5_pQ_fit/' + label1 + '_log_' + var + '_' + mode + '_centered_fit.dat') ])             

        for label2 in fields:
        
            xp2, yp2 = zip(*[tuple(float(ijk) for ijk in line.strip().split('\t'))  for line in open('ResultData/5_pQ_fit/' + label2 + '_log_' + var + '_' + mode + '_centered_fit.dat') ])             
       
            fout.write(label1 + '\t' + label2 + '\t' +  str(stats.ks_2samp(yp1, yp2)[0]) + '\n')        

            print var, mode, '\t', label1 + '\t' + label2 + '\t' +  str(stats.ks_2samp(yp1, yp2)[0])




variables = ['p', 'Q']
modes     = [ 'peak', 'mean']   
Pros      = []
    
for var in variables:
    for mode in modes:
        p = Process(target = get_sim_matrix, args=(var, mode, ))
        Pros.append(p)
        p.start()
  
for t in Pros:
    t.join()


