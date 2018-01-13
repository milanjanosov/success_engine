execfile("0_imports.py")




def divideUnequal(list1, list2):

    counter=0
    step=0
    divided=[]
    for count in list1:
        count = int(count)
        step= counter+ count
        sublist= list2[counter: step]
        counter= step
        divided.append(sublist)
        
    return divided



def get_averaged_vectors(career_len, data, N, randomized):

    career_len_gen = []
    career_max_gen = []
     
    for i in range(N):
 
        if randomized == True: 
            random.shuffle(data)
       
        career_len_gen += career_len   
        synt_careers = divideUnequal(career_len, data)  

        for synt_career in synt_careers:
            career_max_gen.append(max(synt_career))    

    return career_len_gen, career_max_gen



def get_r_model_curves(data_file, max_data_file, ax, label, num_of_bins, title, xlabel, ylabel, log, rndmzd, N):


    ax.set_title(title,   fontsize = 19)
    ax.set_ylabel(ylabel, fontsize = 17)
    ax.set_xlabel(xlabel, fontsize = 17)

    data = [float(line.strip()) for line in open(data_file)]# if 'nan' not in line]
    (data_max, career_len) = zip(*[[float(num) for num in line.strip().split('\t')] for line in open(max_data_file)])#  if 'nan' not in line])
    ax.plot(career_len, data_max, marker = 'o', color = 'lightgrey', alpha = 0.15,linewidth = 0)
    
  
    career_len_gen,  career_max_gen  = get_averaged_vectors(career_len, data[:], N, randomized = rndmzd[0])#, True)
    career_len_data, career_max_data = get_averaged_vectors(career_len, data[:], N, randomized = rndmzd[1])#, False)
 
    xb_data, pb_data, pberr_data = binning.getPercentileBinnedDistribution(np.asarray(career_len_data),  np.asarray(career_max_data), num_of_bins)     
    xb_gen, pb_gen, pberr_gen    = binning.getPercentileBinnedDistribution(np.asarray(career_len_gen),  np.asarray(career_max_gen), num_of_bins)

    ax.errorbar(xb_data, pb_data, yerr = pberr_data, fmt = 'o-', color = 'grey', label = 'data', alpha = 0.9)
    ax.plot(xb_gen,  pb_gen, color = 'r', label = 'R-model', alpha = 0.9)                

    ax.set_ylim([min([min(pb_gen),  min(pb_data)]),  max(max(pb_data) + max(pberr_data), max(xb_gen) + max(pberr_gen))])
    ax.set_xlim([min([min(xb_gen),  min(xb_data)])-1,max(xb_data)+1])    

    


    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')


    outdir = 'ResultData/3_r_model_data'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fdata = open(outdir + '/' + label + '_r_model_bin_data.dat', 'w')
    for i in range(len(xb_data)):
        fdata.write(str(xb_data[i]) + '\t' + str(pb_data[i]) + '\t' + str(pberr_data[i]) + '\n')
    fdata.close


    fdata = open(outdir + '/' + label + '_r_model_raw_data.dat', 'w')
    for i in range(len(career_len)):
        fdata.write(str(career_len[i]) + '\t' + str(data_max[i]) + '\n')
    fdata.close

    
    fdata = open(outdir + '/' + label + '_r_model_bin_gen.dat', 'w')
    for i in range(len(xb_gen)):
        fdata.write(str(xb_gen[i]) + '\t' + str(pb_gen[i]) + '\n')
    fdata.close





def do_the_r_model():


 
    num_of_bins = 12
    
    folder = '../ProcessedData/ProcessedData_0_Normalized' 
    f, ax  = plt.subplots(1, 3, figsize=(23, 5))


    fields = [    ('director',     'film_rating_count_dist_',     'film_career_length_max_rating_count_'),]

    N = 100

    for (label, fn1, fn2) in fields:


        file_cnt    = folder + '/1_impact_distributions/'     + fn1 + label + '.dat'
        max_rat_cnt = folder + '/7_career_length_max_impact/' + fn2 + label + '.dat'  
        get_r_model_curves(file_cnt,  max_rat_cnt,    ax[0], label, num_of_bins, 'True, False, N = ' + str(N)  , 'Career length', 'Rating count' , True, [True, False], N)
        get_r_model_curves(file_cnt,  max_rat_cnt,    ax[1], label, num_of_bins, 'True, True,  N = '  + str(N)  , 'Career length', 'Rating count' , True, [True, True], N)
        get_r_model_curves(file_cnt,  max_rat_cnt,    ax[2], label, num_of_bins, 'False, False, N = ' + str(N)   , 'Career length', 'Rating count' , True, [False, False], N)




    plt.savefig('X_r_model_'+str(N)+'.png')
    plt.show()

   
    
      


do_the_r_model()


        
