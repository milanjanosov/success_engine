import matplotlib.pyplot as plt
import seaborn
import matplotlib
import numpy as np


 
 
 
def plot_career(input_file, output_file, title, impact_id = 0, cutoff = 0, cutoff_id = 0):

    x_label    = 'Time'
    y_label    = 'Impact'
    title_font = 40 
    label_font = 35
    tick_font  = 25
    
    times    = []
    impacts  = [] 
    products = []


    with open(input_file) as myfile:
        next(myfile)
        
        for line in myfile:
        
            fields = line.replace('None', '0').replace(',', '').strip().split('\t')        
        
            product = fields[0]
            time    = float(fields[1])
            impact  = float(fields[2 + impact_id])
            
            if float(fields[2 + cutoff_id]) > cutoff:
                times.append(time)
                impacts.append(impact) 
                products.append(product) 
                        

    max_index     = impacts.index(max(impacts))
    max_impact    = impacts.pop(max_index)
    max_time      = times.pop(max_index)
    best_product  = products.pop(max_index)


    
    matplotlib.rcParams.update({'font.size': 25, 'font.family': 'sans-serif', 'font.weight': 'light'})
 

    seaborn.set_style('white')   
    ff, ax = plt.subplots(1, 1, figsize=(20, 10))
    st = ff.suptitle( title, fontsize = title_font)


    ax.plot(times,     impacts,    'o', color = 'grey',   markersize = 19, alpha = 0.6,  markeredgecolor='black', markeredgewidth = 1)#, linewidth='0'
    ax.plot(max_time,  max_impact, 'o', color = 'tomato', markersize = 21, alpha = 0.85, markeredgecolor='black', markeredgewidth = 1)#, linewidth='0'
    ax.set_ylim([-0.05*max_impact,1.05*max_impact])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
       
    ticklines  = ax.get_xticklines()  + ax.get_yticklines()
    gridlines  = ax.get_xgridlines()  + ax.get_ygridlines()
    ticklabels = ax.get_xticklabels() + ax.get_yticklabels()
    ax.tick_params(labelsize = tick_font, pad = 10)   
    
    [line.set_linewidth(1)    for line in ticklines]
    [line.set_linestyle('-.') for line in gridlines]

    ax.set_xlabel(x_label, fontsize = label_font)
    ax.set_ylabel(y_label, fontsize = label_font)  
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10   

    [ax.axvline(x = times[ij], ymin=0, ymax=impacts[ij]/(1.05*max_impact),  linewidth=1, color = 'k')  for ij in range(len(impacts))]    
    #ax.set_yticklabels(['0', '0','200k', '400k', '600k', '800k', '1M'])#, rotation='vertical')
       
    plt.savefig(output_file)
    plt.show()
       



if __name__ == '__main__': 

    input_file  = 'CareerFiles/1_simple_career.dat'
    output_file = 'CareerPlots/blaabla.png'
    title       = 'Orwell\'s career'
    impact_id   = 0
    cutoff      = 100
    cutoff_id   = 1

    plot_career(input_file, output_file, title, impact_id, cutoff, cutoff_id)
   
        
        
        
                
        
        
        
        
        
        
        
        

