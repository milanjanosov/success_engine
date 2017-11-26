def align_ax(ax, font_tick):

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


def align_plot(ax):

    font_tick = 21   

    if len(ax.shape)> 1:
        for i in range(len(ax)):
            for j in range(len(ax[0])):
                align_ax(ax[i,j], font_tick)  
 
    else:
        for i in range(len(ax)):
            align_ax(ax[i], font_tick)
