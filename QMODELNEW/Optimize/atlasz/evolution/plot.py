"""This module contains scripts to plot evolution results."""

#external imports
noplotting = None
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
except ImportError:
    noplotting = "ERROR: could not import matplotlib. Not plotting."
import numpy
import glob
import os

#internal imports
import evolution


def collect_fitnesses_and_pvalues_from_files(inputdir, subdirglobname, filename="fitnesses.txt"):
    """Get e.g. 'fitnesses.txt' files from (originally atlasz) directories,
    collect results from them and plot them."""
    files = glob.glob(os.path.join(inputdir, subdirglobname, filename))
    allfitnesses = [{} for i in xrange(len(files))]
    allpvalues = [{} for i in xrange(len(files))]
    g_j = None
    p_j = None
    fitness_jlist = None
    pvalues_j = None
    last_j = None
    for fname in files:
        print ("Parsing", fname)
        for line in open(fname, 'r'):
            line = line.strip()
            if not line: continue
            linesplit = line.split("\t")
            # parse header line
            if line.startswith('#'):
                header = linesplit
                if "#g" in header:
                    g_j = header.index("#g")
                else:
                    continue
                if "p" in header:
                    p_j = header.index("p")
                if "pvalues" in header:
                    pvalues_j = header.index("pvalues")
                    last_j = len(header)
                else:
                    pvalues_j = len(header)
                    last_j = len(header)
                if "fitness" in header:
                    fitness_j = header.index("fitness")
                continue
            # parse data line
            allfitnesses[int(linesplit[g_j])][int(linesplit[p_j])] = \
                    dict((header[i], float(linesplit[i])) for i in xrange(fitness_j+1, pvalues_j))
            allpvalues[int(linesplit[g_j])][int(linesplit[p_j])] = \
                    [float(linesplit[i]) for i in xrange(pvalues_j+1, last_j)]
    return (allfitnesses, allpvalues)


def plot_allfitnesses(allfitnesses, outputdir=None):
    """Create neat plots of fitness evolution (both final and partial fitnesses are plotted)."""
    if noplotting:
        print noplotting
        return

    # create figure
    fig0 = plt.figure(0, figsize=(7,5))
    for f in ["fitnesses"] + allfitnesses[0][0].keys():
        if f == "fitnesses":
            x = [evolution.get_single_fitnesses(allfitnesses[g]) for g in xrange(len(allfitnesses))]
        else:
            x = [dict((p, allfitnesses[g][p][f]) for p in allfitnesses[g]) for g in xrange(len(allfitnesses))]

        # calcuate min/max/avg fitnesses
        fitmin = [min(x[g].values()) for g in xrange(len(x))]
        fitmax = [max(x[g].values()) for g in xrange(len(x))]
        fitavg = [numpy.mean(x[g].values()) for g in xrange(len(x))]
        fitstd = [numpy.std(x[g].values()) for g in xrange(len(x))]

        plt.clf()
        plt.xlabel("generations")
        plt.ylabel(f)
        plt.plot(range(len(fitmax)), fitmax, color='r', label="maximum")
        plt.plot(range(len(fitavg)), fitavg, color='b', label="avg/std")
        plt.fill_between(range(len(fitavg)),
                [fitavg[i] - fitstd[i] for i in xrange(len(fitavg))],
                [fitavg[i] + fitstd[i] for i in xrange(len(fitavg))],
                edgecolor='b', facecolor='b', alpha=0.2)
        plt.plot(range(len(fitmin)), fitmin, color='g', label="minimum")
        plt.legend(loc='lower right')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.show()
        if outputdir is not None:
            plt.savefig(os.path.join(outputdir, f + ".png"))


def plot_bestfitnesses(allfitnesses, outputdir=None):
    """Create neat plots of fitness evolution of the phenotype with best fitness
    in every generation."""
    if noplotting:
        print (noplotting)
        return

    # create figure
    fig0 = plt.figure(0, figsize=(7,5))
    plt.clf()
    # get phenotypes with max fitness
    x = [evolution.get_single_fitnesses(allfitnesses[g]) for g in xrange(len(allfitnesses))]
    v = [list(x[g].values()) for g in xrange(len(x))]
    k = [list(x[g].keys()) for g in xrange(len(x))]
    m = [k[g][v[g].index(max(v[g]))] for g in xrange(len(x))]
    plt.xlabel("generations")
    plt.ylabel("fitnesses of best phenotype")
    widthofline = 1
    for f in ["fitnesses"] + allfitnesses[0][0].keys():
        if f != "fitnesses":
            x = [dict((p, allfitnesses[g][p][f]) for p in allfitnesses[g]) for g in xrange(len(allfitnesses))]
            widthofline = 1
        else:
            widthofline = 2
        fitmax = [x[g][m[g]] for g in xrange(len(x))]
        plt.plot(range(len(fitmax)), fitmax, label=f, linewidth=widthofline)
    leg = plt.legend(loc='lower right', bbox_to_anchor = (1.05, 1.0))
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show()
    if outputdir is not None:
        plt.savefig(os.path.join(outputdir, "bestfitnesses.png"), bbox_extra_artists=(leg,), bbox_inches='tight')


def plot_fitness_as_function_of_pvalue(eparams, allfitnesses, allpvalues, outputdir=None, stability_range=None):
    """Create a figure showing pvalues and fitness evolution.

    :param eparams:          the evolutionparams python module name
    :param allfitnesses:     allfitnesses[g][p] = dict of multi-objective fitness values for generation g, phenotype p
    :param allpvalues:       allpvalues[g][p][i] = param value of generation g, phenotype p and param index i
    :param outputdir:        directory where the results will be saved
    :param stability_range:  the calculated stability range of the parameters

    """
    if noplotting:
        print (noplotting)
        return
    # get params list
    params = evolution.get_params_to_evolve(eparams)
    # get single fitnesses
    allsfitnesses = [evolution.get_single_fitnesses(allfitnesses[g]) for g in xrange(len(allfitnesses))]
    # create figure
    fig0 = plt.figure(0, figsize=(7,5))
    for i in xrange(len(params)):
        plt.clf()
        plt.xlabel(params[i].name)
        plt.ylabel("fitness")
        for g in xrange(len(allsfitnesses)):
    	    x = []
    	    y = []
            for p in xrange(len(allpvalues[g])):
                x.append(allpvalues[g][p][i])
                y.append(allsfitnesses[g][p])
            plt.plot(x, y, 'o', color=plt.cm.jet(float(g)/max(1,(len(allsfitnesses)-1))))
        # plot stability range if given
        if stability_range is not None:
            for j in range(2):
                plt.plot([stability_range[i][j], stability_range[i][j]], plt.ylim(), 'k--')
        # create colorbar for generations
        if g:
            ax = fig0.add_axes([0.9, 0.1, 0.03, 0.8])
            cmap = plt.cm.jet
            norm = matplotlib.colors.Normalize(vmin=0, vmax=g)
            cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', format='%i')
            cb.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)) # TODO: this line does not work somehow, why?
            cb.update_ticks()
            cb.set_label('generations')
        # show/save image
        plt.show()
        if outputdir is not None:
            plt.savefig(os.path.join(outputdir, "param__%s.png" % params[i].name))


def plot_fitness_pvalues_animation(eparams, allfitnesses, allpvalues,
        model=None, x=0, y=1, stability_range=None):
    """Create an animation showing pvalues and fitness evolution.

    :param eparams:          the evolutionparams python module name
    :param allfitnesses:     allfitnesses[g][p] = dict of multi-objective fitness values for generation g, phenotype p
    :param allpvalues:       allpvalues[g][p][i] = param value of generation g, phenotype p and param index i
    :param model:            the name of the model to get fitness heatmap from (if available)
    :param x:                the first param index to show on the 2D plot
    :param y:                the second param index to show on the 2D plot

    """
    if noplotting:
        print (noplotting)
        return
    # get params list
    params = evolution.get_params_to_evolve(eparams)
    # get single fitnesses
    allsfitnesses = [evolution.get_single_fitnesses(allfitnesses[g]) for g in xrange(len(allfitnesses))]
    # calcuate min/max/avg fitnesses
    fitmin = [min(allsfitnesses[g].values()) for g in xrange(len(allsfitnesses))]
    fitmax = [max(allsfitnesses[g].values()) for g in xrange(len(allsfitnesses))]
    fitavg = [numpy.mean(allsfitnesses[g].values()) for g in xrange(len(allsfitnesses))]
    fitstd = [numpy.std(allsfitnesses[g].values()) for g in xrange(len(allsfitnesses))]
    # initialize animation left window with heatmap of fitness function
    fig0 = plt.figure(0, figsize=(12,5))
    plt.clf()
    fig0.canvas.set_window_title("Fitness evolution")
    animleft = fig0.add_subplot(1, 2, 1)
    animleft.autoscale(enable=True, axis=u'both', tight=True)
    # draw fitness heatmap if applicable
    if model is not None:
        d = 100
        X = numpy.linspace(params[x].minv, params[x].maxv, d)
        Y = numpy.linspace(params[y].maxv, params[y].minv, d) # NOTE: -1x axis direction needed and deliberate
        Z = [[evolution.get_single_fitness(evolution.get_fitnesses(eparams, model, [[X[i],Y[j]]])[0]) for i in xrange(d)] for j in xrange(d)]
        animleft.imshow(Z, extent=[params[x].minv, params[x].maxv, params[y].minv, params[y].maxv], interpolation=None)
        # plot stability range if given
        if stability_range is not None:
            animleft.add_patch(Rectangle(
                    xy=[stability_range[x][0], stability_range[y][0]],
                    width=stability_range[x][1] - stability_range[x][0],
                    height=stability_range[y][1] - stability_range[y][0],
                    facecolor='none', edgecolor='k'))
    # set x-y limits to a slightly larger area than param ranges
    lag = 0.02
    rangex = params[x].maxv - params[x].minv
    minx = params[x].minv - rangex * lag
    maxx = params[x].maxv + rangex * lag
    rangey = params[y].maxv - params[y].minv
    miny = params[y].minv - rangey * lag
    maxy = params[y].maxv + rangey * lag
    plt.xlim([minx, maxx])
    plt.ylim([miny, maxy])
    # set axis labels
    plt.xlabel(params[x].name)
    plt.ylabel(params[y].name)
    animleft_title = plt.title("generation ##")
    animleft_dots, = animleft.plot([], [], 'ro')
    animleft_dotbest, = animleft.plot([], [], 'wo')

    # initialize animation right window with fitness evolution
    animright = fig0.add_subplot(1, 2, 2)
#    animright.autoscale(enable=True, axis=u'both', tight=True)
    plt.xlabel("generations")
    plt.ylabel("fitness")
    animright.plot(range(len(fitmax)), fitmax, color='r', label="maximum")
    animright.plot(range(len(fitavg)), fitavg, color='b', label="avg/std")
    animright.fill_between(range(len(fitavg)),
            [fitavg[i] - fitstd[i] for i in xrange(len(fitavg))],
            [fitavg[i] + fitstd[i] for i in xrange(len(fitavg))],
            edgecolor='b', facecolor='b', alpha=0.2)
    animright.plot(range(len(fitmin)), fitmin, color='g', label="minimum")
    animright_fitnessline, = animright.plot([], [], 'k--')
    animright.legend(loc='lower right')

    # animate generations
    def animinit():
        animleft_title.set_text("")
        animleft_dots.set_data([], [])
        animleft_dotbest.set_data([], [])
        animright_fitnessline.set_data([], [])
        return animleft_dots, animleft_dotbest, animleft_title, animright_fitnessline
    def animate(i):
        animleft_title.set_text("generation #%d" % i)
        #plt.figure(0)
        plt.draw()
        animleft_dots.set_data([allpvalues[i][j][x] for j in xrange(len(allpvalues[i]))], [allpvalues[i][j][y] for j in xrange(len(allpvalues[i]))])
        best = sorted(allfitnesses[i], key=allfitnesses[i].get, reverse=True)[0]
        animleft_dotbest.set_data([allpvalues[i][best][x]], [allpvalues[i][best][y]])
        miny = min(plt.yticks()[0])
        maxy = max(plt.yticks()[0])
        animright_fitnessline.set_data([i,i], [miny, maxy])
        return animleft_dots, animleft_dotbest, animleft_title, animright_fitnessline
    anim = animation.FuncAnimation(fig0, animate, frames = len(allfitnesses), init_func=animinit, interval=200, blit=False)
    plt.ioff()
    plt.show()
