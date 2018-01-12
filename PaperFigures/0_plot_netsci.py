import matplotlib.pyplot as plt
import seaborn
import numpy as np


def align_plot(ax):

    font_tick = 15

    ax.legend(loc = 'left', fontsize = font_tick)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ticklines  = ax.get_xticklines()  + ax.get_yticklines()
    gridlines  = ax.get_xgridlines()  + ax.get_ygridlines()
    ticklabels = ax.get_xticklabels() + ax.get_yticklabels()
    
    for line in ticklines:
        line.set_linewidth(1)

    for line in gridlines:
        line.set_linestyle('-.')

    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15   
    ax.tick_params(labelsize = font_tick)    



def plot_red_lines(ax):

    x = np.arange(0,1, 1.0/20)
    colors = ['skyblue', 'salmon', 'springgreen']

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    yyy = [1 - y0 for y0 in x]
    ax.plot(x, yyy, '-', linewidth=8, alpha = 0.2, color = colors[0], label = '$U(0,1)$') 
    ax.set_xlabel('$N^{*}/N$', fontsize=21)
    ax.set_ylabel( r'$P( \geq  N^{*}/N)$' , fontsize=21)




def plot_NN(ax):


    alpha_hist = 0.2
    title_font = 27 
    labelsize = 27

    seaborn.set_style('white')   
    #f, ax = plt.subplots(1, 2, figsize=(18, 8))
    #st = f.suptitle( 'CCDF of the relative rank of the best product', fontsize = title_font+3)

    (xmovie, pmovie, perrmovie) =  zip(*[ [float(a) for a in line.strip().split('\t')] for line in open('2_r_rule_data/directorNNstar_data.dat')])  
    xmovie    = np.asarray(list(xmovie)) 
    pmovie    = np.asarray(list(pmovie))
    perrmovie = np.asarray(list(perrmovie)) 
    ax.fill_between(xmovie, pmovie-perrmovie, pmovie+perrmovie, alpha = 0.2, color = 'steelblue')            
    ax.errorbar(xmovie, pmovie, yerr=perrmovie, fmt= '-', linewidth = 4, color =  'steelblue', markersize = 0, marker = 'o', alpha = 0.95, label = 'Data, 746 careers') 
    ax.set_title('IMDb, Metascores', fontsize = title_font)


    plot_red_lines(ax)
    align_plot(ax) 
    
    
def test_r_model(ax):

    title_font = 27 
    labelsize = 27
    seaborn.set_style('white')   
    #f, ax = plt.subplots(1, 2, figsize=(17, 8))
    #st = f.suptitle( 'Predictions of the random impact rule', fontsize = title_font+3)
    
    ms = 10


    (xdata, ydata) =  zip(*[ [float(a) for a in line.strip().split('\t')] for line in open('3_r_model_data/director_r_model_raw_data.dat')])      
    ax.plot(xdata, ydata, marker = 'o', markersize = ms, color = 'lightskyblue', alpha = 0.15,linewidth = 0)


    (xbdata, pbdaza) =  zip(*[ [float(a) for a in line.strip().split('\t')] for line in open('3_r_model_data/director_r_model_bin_gen.dat')])  
    ax.plot(xbdata, pbdaza, linewidth = 4, color =  'k', markersize = 0, marker = 'o', alpha = 0.95, label = 'R-model' ) 
    
  


    (xbgen, pbgen, pberrgen) =  zip(*[ [float(a) for a in line.strip().split('\t')] for line in open('3_r_model_data/director_r_model_bin_data.dat')])  
    ax.errorbar(xbgen, pbgen, yerr=pberrgen, fmt= '-', linewidth = 4, color =  'steelblue', markersize = 0, marker = 'o', alpha = 0.95, label = 'Data (binned)') 
    ax.set_xlabel('Career length', fontsize = title_font)
    ax.set_ylabel('Success of the best product', fontsize = title_font)    
    ax.set_title('IMDb, Metascores', fontsize = title_font)    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([0, 100000])    


    align_plot(ax) 

       
   

def plot_career2(ax):

    
    years = []
    avgr  = []
    maxa  = []
    maxy  = [] 
    
    data = []
    
    
    for line in open('nm0000184_writer_simple_career'):
        
        if 'year' not in line:
    
            line = line.replace('None', '0')        
            


            fields  = line.strip().split('\t')
           
            try:
                movie   = fields[0]
                year    = int(fields[1])
                ratings = float(fields[2])
                count   = float(fields[3])
                
                
                if count > 1000 and count < 900000:
                    years.append(year)
                    avgr.append(count)   
                    data.append((year, count))
                    
                if count > 900000:
                    maxa.append(count)
                    maxy.append(year)  
                    data.append((year, count) )

            except:
                pass    
            
            
    print len(avgr) 

    years, avgr = zip(*sorted(data, key=lambda tup: tup[0]))
    years = range(1, len(years)+1)
    avgr  = list(avgr)

    print avgr
    

    #matplotlib.rcParams.update({'font.size': 25, 'font.family': 'sans-serif', 'font.weight': 'light'})
    title_font = 35 
    font_tick  = 31
    seaborn.set_style('white')   
    #ff, ax = plt.subplots(1, 1, figsize=(20, 10))
    #st = ff.suptitle( 'Career trajectory of $George$ $Lucas$', fontsize = title_font+5)


    ax.plot(years, avgr, 'o', color = 'grey', markersize = 19, alpha = 0.6, markeredgecolor='black', markeredgewidth = 1)#, linewidth='0'
    ax.plot(4,  maxa, 'o', color = 'tomato', markersize = 21, alpha = 0.85, markeredgecolor='black', markeredgewidth = 1)#, linewidth='0'
    ax.set_ylim([-20000,1050000])




    #years.append(maxy[0])
    #avgr.append(maxa[0])   

    ax.legend(loc = 'left', fontsize = font_tick)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ticklines  = ax.get_xticklines()  + ax.get_yticklines()
    gridlines  = ax.get_xgridlines()  + ax.get_ygridlines()
    ticklabels = ax.get_xticklabels() + ax.get_yticklabels()
    ax.tick_params(labelsize = font_tick, pad = 9)   
  
    ax.set_yticklabels(['0', '0','200k', '400k', '600k', '800k', '1M'])#, rotation='vertical')
    ax.set_xlabel('Rank of movies', fontsize = title_font)
    ax.set_ylabel('Number of ratings', fontsize = title_font)  
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10   
    [ax.axvline(x = years[ij], ymin=0, ymax=avgr[ij]/1050000,  linewidth=1, color = 'k')  for ij in range(len(years))] 
    [line.set_linewidth(1) for line in ticklines]
    [line.set_linestyle('-.') for line in gridlines]
   









  
#plot_NN()
#test_r_model()




fig =plt.figure(figsize=(24,16))
fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)
X = [ (2,3,(1,3)), (2,2,3), (2,2,4) ]
seaborn.set_style('white')  


for index, (nrows, ncols, plot_number) in enumerate(X):

    sub = fig.add_subplot(nrows, ncols, plot_number)

    if index == 0: 
        plot_career2(sub)
    if index == 1: 
        plot_NN(sub)
    elif index == 2: 
        test_r_model(sub)
    
#    plt.tight_layout(pad=8, w_pad=8, h_pad=8)              s   
plt.show()



