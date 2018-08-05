"""
Run flyingrobots/simulations/robotsim/robotflocksim_main scripts on atlasz
as sbatch scripts in many iterations, optimizing it with a genetic algorithm.

General usage:

    The script is recursive. It first checks whether it is the only job on
    atlasz and if so, tries to collect information from last generation
    and continues evolution from there by generating the next generation jobs.

    Termination of a generation is checked by squeue and might get messy if you
    have other jobs running. So try to run only one evolution at a time or
    at least use very different labels for them. Better use only one evolution.

    It is possible to continue evolution any time with the same settings, since
    script is optimized to get info from last executed generation and continue
    evolution from there.

General terminology:

    params     usually refers to the parameters of the model that can be changed.
               These are stored in three separate settings file:
               (f)lockingparams.dat, (u)nitparams.dat, (i)nitparams.dat
               The evolution tries to optimize params stored in these files.
               param_t stores "filename name min max step" of each parameter
    pvalues    is usually a list of values corresponding to a list of params
    eparams    is an imported python module that stores tuning parameters of
               the evolutionary algo itself
    fitnesses  the fitness values of the last generation, stored as a dict where
               keys are phenotype indices, values are fitness dicts for each
               phenotype, averaged over all environments
    sfitnesses single-valued fitnesses with same structure as above
    g          generation index
    p          phenotype index within one generation
    n          environment index for testing a phenotype in multiple envs

"""

# external imports
from __future__ import print_function
import os
import sys
import argparse
import datetime
import subprocess
import shutil
import getpass
import glob
import importlib
import time
from collections import namedtuple

#internal imports
import atlasz.util
import atlasz.evolution.evolution as evolution
import atlasz.evolution.plot as evolplot
import atlasz.evolution.cmainterface as cmaif
import atlasz_batch_postprocess

# define user variables here
USER = getpass.getuser() # automatic, change only if you need something explicit
CODE_DIR = '/project/flockwork/%s/code/robotsim' % USER
WORK_DIR = '/project/flockwork/%s/robotsim_workdir/evolution/' % USER
RESULT_DIR = '/project/flockwork/%s/robotsim_resultdir/evolution/' % USER
DATE = str(datetime.datetime.now().date())


def parse_evolution_postprocess_file(filename):
    """Parse an evolution-type atlasz_batch_postprocess file into a data array."""
    header = []
    data = []
    for line in open(filename, 'r'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        linesplit = line.split('\t')
        if linesplit[0] == 'g' and linesplit[1] == 'p':
            header = linesplit
            continue
        data.append([float(p) for p in linesplit])
    return (header, data)


def find_last_executed_generation(eparams, options):
    """Find last generation that has been evaluated already.

    :param eparams:   the evolutionparams python module name
    :param options:   the main arguments

    """
    g = -1
    for d in glob.iglob(os.path.join(RESULT_DIR, options.label, "%s__g_*" % options.label)):
        # check whether d is a directory
        if not os.path.isdir(d): continue
        # kind of check whether the whole population is present
        i = 0
        for subdir in glob.iglob(os.path.join(d, "robotsim_evolution__g_*__p_*__n_*")):
            if os.path.isdir(subdir):
                i += 1
        if i != eparams.phenotypes * eparams.environments:
            continue
        # if no errors, check if this is the last generation
        newg = int(d[-5:])
        if newg > g:
            g = newg
    return g


def get_fitnesses_from_a_generation(eparams, options, g):
    """run postprocess and get fitness values for all phenotypes of a generation.

    :param eparams:   the evolutionparams python module name
    :param options:   the main arguments
    :param g:         the index of the generation

    """
    fitnessparameter_t = namedtuple("fitnessparameter_t", "header data")

    # check if there is an evaluated generation at all
    if g == -1:
        return None

    # First run postprocess
    print ("Running atlasz_batch_postprocess...")
    inputdir = os.path.join(RESULT_DIR, options.label, "%s__g_%05d" % (options.label, g))
    ppfilename = atlasz_batch_postprocess.main([inputdir, options.model, options.label])
    if not ppfilename:
        print ("  Warning: could not run 'atlasz_batch_postprocess'")
        return None

    # parse postprocess file
    print ("Parse postprocess values from:", ppfilename)
    ppheader, ppdata = parse_evolution_postprocess_file(ppfilename)

    # Get fitnesses of different phenotypes
    print ("Evaluating fitnesses of all phenotypes...")
    fitnesses = evolution.get_fitnesses(eparams, options.model, fitnessparameter_t(ppheader, ppdata))

    # And finally, return them
    return fitnesses


def get_cma_from_a_generation(eparams, options, fitnesses, pvalues, g):
    """Init or reload cmaes from last generation.

    :param eparams:   the evolutionparams python module name
    :param options:   the main arguments
    :param fitnesses: fitness values from the same generation as a dict
    :param pvalues:   the values of params to evolve
    :param g:         the index of the generation

    :return the cmaes evolution strategy object

    """
    if options.strategy != "cma":
        return None
    if g == -1:
        cmaes = cmaif.init_cma(eparams, pvalues, fitnesses)
    else:
        # load previously saved objects
        inputdir = os.path.join(RESULT_DIR, options.label, "%s__g_%05d" % (options.label, g))
        cmasolutions = atlasz.util.load_object(os.path.join(inputdir, "cmasolutions__g_%d.zip" % g))
        cmaes = atlasz.util.load_object(os.path.join(inputdir, "cmaes__g_%d.zip" % g))
        # update it with current evolutionparams (if e.g. number of generations has changed)
        cmaif.update_cma(cmaes, eparams)
        # refresh evolution strategy with fitnesses of last generation
        cmaif.store_fitnesses(cmaes, fitnesses, cmasolutions)
    return cmaes


def save_and_plot_generation(cmaes, eparams, options, fitnesses, pvalues, g):
    """Save and plot fitnesses and pvalues for a generation.

    :param cmaes:     the latest cma evolution strategy object
    :param eparams:   the evolutionparams python module name
    :param options:   the main arguments
    :param fitnesses: fitness values from the same generation as a dict
    :param pvalues:   the values of params to evolve
    :param g:         the index of the generation

    """
    if fitnesses is None or pvalues is None or g < 0:
        return

    # get inputdir
    inputdir = os.path.join(RESULT_DIR, options.label, "%s__g_%05d" % (options.label, g))
    # Save fitnesses to inputdir
    print ("Saving fitnesses...")
    filename = os.path.join(inputdir, "fitnesses.txt")
    # Save all phenotypes of current generation
    evolution.save_fitnesses(filename, eparams, fitnesses, g, pvalues)
    # Add mean solution
    (pmean, pstd) = cmaif.get_mean_solution(cmaes, eparams)
    evolution.save_fitnesses_hack(filename, eparams, len(fitnesses[0]),
            "mean", 1, pmean, "-")
    # Add favorite solution
    (pfavorite, pstd) = cmaif.get_favorite_solution(cmaes, eparams)
    evolution.save_fitnesses_hack(filename, eparams, len(fitnesses[0]),
            "favorite", 1, pfavorite, "-")
    # Add best solution
    (pbest, fitness_of_best, evaluation_of_best) = cmaif.get_best_solution(cmaes, eparams)
    evolution.save_fitnesses_hack(filename, eparams, len(fitnesses[0]),
            "best", evaluation_of_best, pbest, fitness_of_best)

    # Save params for mean solution
    meandir = os.path.join(inputdir, "_mean_solution")
    if not os.path.isdir(meandir):
        os.makedirs(meandir)
    shutil.copyfile(options.initparams, os.path.join(meandir, os.path.split(options.initparams)[1]))
    shutil.copyfile(options.unitparams, os.path.join(meandir, os.path.split(options.unitparams)[1]))
    shutil.copyfile(options.flockingparams, os.path.join(meandir, os.path.split(options.flockingparams)[1]))
    fui = {"f":os.path.split(options.flockingparams)[1],
            "u":os.path.split(options.unitparams)[1],
            "i":os.path.split(options.initparams)[1]}
    # redefinition needed because of fui --> paramfile
    params = [atlasz.util.param_t(fui[pp.paramfile], pp.name, pp.minv, pp.maxv, pp.step) for pp in evolution.get_params_to_evolve(eparams)]
    atlasz.util.replace_params_in_paramfiles(meandir, params, pmean)

    # Save params for favorite solution
    favoritedir = os.path.join(inputdir, "_favorite_solution")
    if not os.path.isdir(favoritedir):
        os.makedirs(favoritedir)
    shutil.copyfile(options.initparams, os.path.join(favoritedir, os.path.split(options.initparams)[1]))
    shutil.copyfile(options.unitparams, os.path.join(favoritedir, os.path.split(options.unitparams)[1]))
    shutil.copyfile(options.flockingparams, os.path.join(favoritedir, os.path.split(options.flockingparams)[1]))
    fui = {"f":os.path.split(options.flockingparams)[1],
            "u":os.path.split(options.unitparams)[1],
            "i":os.path.split(options.initparams)[1]}
    # redefinition needed because of fui --> paramfile
    params = [atlasz.util.param_t(fui[pp.paramfile], pp.name, pp.minv, pp.maxv, pp.step) for pp in evolution.get_params_to_evolve(eparams)]
    atlasz.util.replace_params_in_paramfiles(favoritedir, params, pfavorite)

    # Save params for best solution
    bestdir = os.path.join(inputdir, "_best_solution")
    if not os.path.isdir(bestdir):
        os.makedirs(bestdir)
    shutil.copyfile(options.initparams, os.path.join(bestdir, os.path.split(options.initparams)[1]))
    shutil.copyfile(options.unitparams, os.path.join(bestdir, os.path.split(options.unitparams)[1]))
    shutil.copyfile(options.flockingparams, os.path.join(bestdir, os.path.split(options.flockingparams)[1]))
    atlasz.util.replace_params_in_paramfiles(bestdir, params, pbest)

    # Plot fitness evolution until this point
    allfitnesses, allpvalues = evolplot.collect_fitnesses_and_pvalues_from_files(
            os.path.join(RESULT_DIR, options.label), "%s__g_*" % options.label)
    evolplot.plot_allfitnesses(allfitnesses, inputdir)
    evolplot.plot_bestfitnesses(allfitnesses, inputdir)
    evolplot.plot_fitness_as_function_of_pvalue(eparams, allfitnesses, allpvalues, inputdir)


def get_pvalues_from_a_generation(eparams, options, fitnesses, g):
    """Parse back pvalues from a generation and
    convert them to same format as fitnesses.

    :param eparams:   the evolutionparams python module name
    :param options:   the main arguments
    :param fitnesses: fitness values from the same generation as a dict
    :param g:         the index of the generation

    """
    # if there are no fitnesses, we return initial p values from options
    # as the "mean" phenotype
    pvalues = {}
    params_to_evolve = evolution.get_params_to_evolve(eparams)
    if not fitnesses:
        p = "mean"
        pvalues[p] = []
        settings = atlasz.util.settings_t(
                atlasz.util.parse_paramfile_into_dict(options.flockingparams),
                atlasz.util.parse_paramfile_into_dict(options.unitparams),
                atlasz.util.parse_paramfile_into_dict(options.initparams))
        for param in params_to_evolve:
            if param.paramfile == 'f':
                pvalues[p].append(settings.flockingparams[param.name])
            elif param.paramfile == 'u':
                pvalues[p].append(settings.unitparams[param.name])
            elif param.paramfile == 'i':
                pvalues[p].append(settings.initparams[param.name])
        return pvalues

    # if there are fitnesses from last generation, we parse pvalues for all
    # fitness keys (phenotypes)
    for p in fitnesses.keys():
        pvalues[p] = []
        jobid = "__".join(["robotsim_evolution", "g_%d" % g, "p_%d" % p, "n_0", options.label])
        inputdir = os.path.join(RESULT_DIR, options.label, "%s__g_%05d" % (options.label, g), jobid)
        settings = atlasz.util.parse_paramfiles_from_jobfile(os.path.join(inputdir, "job.sh"))
        if settings is None or not settings.initparams or not settings.unitparams or not settings.flockingparams:
            print ("  Warning: could not parse ini files; skipping p =", p)
            pvalues[p].append(None)
            continue
        for param in params_to_evolve:
            if param.paramfile == 'f':
                pvalues[p].append(settings.flockingparams[param.name])
            elif param.paramfile == 'u':
                pvalues[p].append(settings.unitparams[param.name])
            elif param.paramfile == 'i':
                pvalues[p].append(settings.initparams[param.name])
    return pvalues


def make_job(options, params, pvalues, g, p, n):
    """Returns .sh script, jobid and workdir.

    :param options:  the main arguments
    :param params:   the param_t list of params to be changed
    :param pvalues:  the new values of params to be set
    :param g:        generation/iteration index
    :param p:        population/phenotype index
    :param n:        nth environment for a given phenotype

    :return (jobscript, jobid, workdir)

    """
    # create job id
    jobid = "__".join(["robotsim_evolution", "g_%d" % g, "p_%d" % p, "n_%d" % n, options.label])
    gendir = "%s__g_%05d" % (options.label, g)
    # check result dir
    resultdir = os.path.join(RESULT_DIR, options.label, gendir, jobid)
    if os.path.isdir(resultdir):
        print ('Result directory already exists, I will delete it\n', resultdir)
        subprocess.call (["rm", "-r", resultdir])
    # check workdir
    workdir = os.path.join(WORK_DIR, options.label, gendir, jobid)
    if os.path.isdir(workdir):
        print ('Working directory already exists, I will delete it\n', workdir)
        subprocess.call (["rm", "-r", workdir])
    # create dirs
    os.makedirs(workdir)
    os.makedirs(resultdir)
    # copy all default ini files to working dir
    shutil.copyfile(options.initparams, os.path.join(workdir, os.path.split(options.initparams)[1]))
    shutil.copyfile(options.unitparams, os.path.join(workdir, os.path.split(options.unitparams)[1]))
    shutil.copyfile(options.flockingparams, os.path.join(workdir, os.path.split(options.flockingparams)[1]))
    shutil.copyfile(options.obstacles, os.path.join(workdir, os.path.split(options.obstacles)[1]))
    shutil.copyfile(options.arenas, os.path.join(workdir, os.path.split(options.arenas)[1]))
    shutil.copyfile(options.waypoints, os.path.join(workdir, os.path.split(options.waypoints)[1]))
    shutil.copyfile(options.outputconf, os.path.join(workdir, os.path.split(options.outputconf)[1]))
    # change params in ini files according to current iterated values
    if not atlasz.util.replace_params_in_paramfiles(workdir, params, pvalues):
        return (None, None, None)
    # create .sh
    template_variables = {
        'codedir': CODE_DIR,
        'workdir': workdir,
        'resultdir': resultdir,
        'initparams': os.path.join(workdir, os.path.split(options.initparams)[1]),
        'unitparams': os.path.join(workdir, os.path.split(options.unitparams)[1]),
        'flockingparams': os.path.join(workdir, os.path.split(options.flockingparams)[1]),
        'obstparams': os.path.join(workdir, os.path.split(options.obstacles)[1]),
        'arenaparams': os.path.join(workdir, os.path.split(options.arenas)[1]),
        'wpparams': os.path.join(workdir, os.path.split(options.waypoints)[1]),
        'outputconf': os.path.join(workdir, os.path.split(options.outputconf)[1]),
    }
    return (atlasz.util.ROBOTFLOCKSIM_JOB_TEMPLATE % template_variables, jobid, workdir)


def make_master_job(argv, options, g):
    """Returns .sh script, jobid and workdir.

    :params argv:    argument list passed to main()
    :param options:  the main arguments
    :param g:        generation/iteration index

    :return (jobscript, jobid, workdir)

    """
    # create job id
    jobid = "__".join(["robotsim_evolution", "g_%d" % g, options.label, "master"])
    gendir = "%s__g_%05d" % (options.label, g)
    # check result dir
    resultdir = os.path.join(RESULT_DIR, options.label, gendir, jobid)
    if os.path.isdir(resultdir):
        print ('Result directory already exists, I will delete it\n', resultdir)
        subprocess.call (["rm", "-r", resultdir])
    # check workdir
    workdir = os.path.join(WORK_DIR, options.label, gendir, jobid)
    if os.path.isdir(workdir):
        print ('Working directory already exists, I will delete it\n', workdir)
        subprocess.call (["rm", "-r", workdir])
    # create dirs
    os.makedirs(workdir)
    os.makedirs(resultdir)
    # create .sh
    template_variables = {
        'codedir': CODE_DIR,
        'workdir': workdir,
        'resultdir': resultdir,
        'argv': " ".join(argv),
    }
    return (atlasz.util.ROBOTFLOCKSIM_JOB_TEMPLATE_MASTER % template_variables, jobid, workdir)


def make_jobs_for_a_generation(argv, eparams, options, cmaes, lastfitnesses, lastpvalues, g):
    """Generate parameters for the next generation and run robotflocksim with each
    param set as atlasz jobs. Also generate a meta job that collects data if
    all generation jobs has terminated.

    :params argv:         argument list passed to main()
    :param eparams:       the evolutionparams python module name
    :param options:       the main arguments
    :param cmaes:         the latest cma evolution strategy object
    :param lastfitnesses: fitness values from the last generation
    :param lastpvalues:   the values of params to evolve from te last generation
    :param g:             the index of the next generation

    """
    cmasolutions = None
    popparams, poppvalues = evolution.generate_population(eparams, lastpvalues, lastfitnesses)
    if options.strategy == "cma":
        cmasolutions, poppvalues = cmaif.generate_population(cmaes, eparams)
    envparams, envpvalues = evolution.generate_random_environments(eparams)
    fui = {"f":os.path.split(options.flockingparams)[1],
            "u":os.path.split(options.unitparams)[1],
            "i":os.path.split(options.initparams)[1]}
    # get a list of all parameters to change
    # redefinition needed because of fui --> paramfile
    params = [atlasz.util.param_t(fui[pp.paramfile], pp.name, pp.minv, pp.maxv, pp.step) for pp in envparams + popparams]
    for n in xrange(len(envpvalues)):
        for p in xrange(len(poppvalues)):
            # get a list of all new pvalues
            print ("  n", n, envpvalues[n])
            print ("  p", p, poppvalues[p])
            pvalues = envpvalues[n] + poppvalues[p]
            # make and run job
            (jobscript, jobid, workdir) = make_job(options, params, pvalues, g, p, n)
            if jobscript is not None:
                atlasz.util.run_job(jobscript, jobid, workdir)
            else:
                print ("  WARNING: could not create jobscript for g=%d, p=%d, n=%d" % (g, p, n))
    # save cmasolutions and cmaes for later use
    resultdir = os.path.join(RESULT_DIR, options.label, "%s__g_%05d" % (options.label, g))
    atlasz.util.save_object(cmasolutions, os.path.join(resultdir, "cmasolutions__g_%d.zip" % g))
    atlasz.util.save_object(cmaes, os.path.join(resultdir, "cmaes__g_%d.zip" % g))
    # generate master job to monitor generation jobs
    (jobscript, jobid, workdir) = make_master_job(argv, options, g)
    if jobscript is not None:
        atlasz.util.run_job(jobscript, jobid, workdir)
    else:
        print ("  WARNING: could not create master jobscript for g=%d" % g)


def main(argv=[]):
    """Main entry point."""
#    print "This is", __file__, "SVN revision:", atlasz.util.get_svn_info(__file__)['revision']
    # parse command line arguments
    argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    argparser.add_argument("-i", "--initparams", metavar="file",
            dest="initparams",
            default=os.path.join(CODE_DIR, "parameters", "initparams.dat"),
            help="Define init params file")
    argparser.add_argument("-u", "--unitparams", metavar="file",
            dest="unitparams",
            default=os.path.join(CODE_DIR, "parameters", "unitparams.dat"),
            help="Define unit params file")
    argparser.add_argument("-f", "--flockingparams", metavar="file",
            dest="flockingparams",
            default=os.path.join(CODE_DIR, "parameters", "flockingparams.dat"),
            help="Define flocking params file")
    argparser.add_argument("-e", "--evolutionparams", metavar="module",
            dest="evolutionparams", default="evolution.evolutionparams",
            help="Define evolution params python module")
    argparser.add_argument("-obs", "--obstacles", metavar="file",
            dest="obstacles",
            default=os.path.join(CODE_DIR, "parameters", "obstacles.default"),
            help="Define obstacle file")
    argparser.add_argument("-arena", "--arenas", metavar="file", dest="arenas",
            default=os.path.join(CODE_DIR, "parameters", "arenas.default"),
            help="Define arena file")
    argparser.add_argument("-wp", "--waypoints", metavar="file",
            dest="waypoints",
            default=os.path.join(CODE_DIR, "parameters", "waypoints.default"),
            help="Define waypoint file")
    argparser.add_argument("-outputconf", "--outputconf", metavar="file",
            dest="outputconf",
            default=os.path.join(CODE_DIR, "config", "output_config.ini"),
            help="Define output config file")
    argparser.add_argument("-l", "--label", required=True, dest="label",
            help="Specific keyword related to the actual evolution")
    argparser.add_argument("-m", "--model", required=True, dest="model",
            help="The name of the simulation model that is executed")
    argparser.add_argument("-s", "--strategy", dest="strategy",
            default="default", choices=["default", "cma"],
            help="What kind of evolution strategy should we use?")
    argparser.add_argument("--stat", dest="statistics", default=False,
            action="store_true",
            help="With this option no optimization is run but given parameters "
            "will be used for statistics in 1 generation.")
    argparser.add_argument("-b", "--bounds", metavar="[lb, ub]", dest="bounds",
            default=None, help="Redefine inner bounds of the (cma) optimizer "
            "as a python string. 'lb' and 'ub' are either single values or "
            "None or n-dim vectors. Note that bounds of evolutionparams will be"
            " transformed to [0, 1], new bounds defined here should be relative"
            " to this. TODO: not working yet as "
            "BoxConstraintsLinQuadTransformation() do not get updated in cma.")

    # if arguments are passed to main(argv), parse them
    if argv:
        options = argparser.parse_args(argv)
    # else if called from command line or no arguments are passed to main, parse default argument list
    else:
        options = argparser.parse_args()

    # wait until last generation has ended already
    sleep_iteration = 0
    update_interval = 10
    while 1:
        queue = atlasz.util.queued_jobs(options.label)
        print ("  %ds: %d jobs left in queue" % (sleep_iteration*update_interval, len(queue)))
        sys.stdout.flush()
        if not queue:
            break
        time.sleep(update_interval)
        sleep_iteration += 1
    # get fitness values from the last generation
    print ("All jobs finished, evaluating last generation...")

    # parse evolution parameters to be used
    eparams = importlib.import_module(options.evolutionparams)

    # if we are in stat mode, we restrict limits to optimal single value and
    # restrict generations to 1 to get many instances of the same phenotypes
    if options.statistics:
        print("Running in statistics mode: no optimization, only multiple phenotypes around given setup")
        # load all params
        fui = {"f":atlasz.util.parse_paramfile_into_dict(options.flockingparams),
               "u":atlasz.util.parse_paramfile_into_dict(options.unitparams),
               "i":atlasz.util.parse_paramfile_into_dict(options.initparams)
        }
        # change param values to single one stored in file
        params = [atlasz.util.param_t(
                pp.paramfile, pp.name,
                fui[pp.paramfile][pp.name], fui[pp.paramfile][pp.name], pp.step)
                for pp in evolution.get_params_to_evolve(eparams)]
        # overwrite eparams
        eparams.params_to_evolve = [" ".join([str(x) for x in pp]) for pp in params]
        eparams.generations = 1

    # find last executed generation
    g = find_last_executed_generation(eparams, options)
    print ("Last generation:", g)

    # get multi-objective fitness values from the last generation
    lastfitnesses = get_fitnesses_from_a_generation(eparams, options, g)
    # check for errors
    if g != -1 and not lastfitnesses:
        print ("ERROR: No fitness values could be parsed")
        return

    # get p values from last generation (or init them from fui files)
    lastpvalues = get_pvalues_from_a_generation(eparams, options, lastfitnesses, g)

    # get cma from last generation
    cmaes = get_cma_from_a_generation(eparams, options, lastfitnesses, lastpvalues, g)

    # change bounds if defined
    if options.bounds is not None:
        bounds = eval(options.bounds)
        print("Bounds changed to", bounds)
        cmaes.boundary_handler.bounds = bounds

    # plot last generation
    save_and_plot_generation(cmaes, eparams, options, lastfitnesses, lastpvalues, g)

    # check termination
    if ((options.strategy == "cma" and cmaif.is_termination(cmaes, eparams, lastfitnesses, g)) or \
            (options.strategy == "default" and evolution.is_termination(eparams, lastfitnesses, g))):
        print ("Terminating at generation", g)
        return

    # step to next generation, create and run jobs for it
    g += 1
    print ("\nCreating generation", g)
    make_jobs_for_a_generation(argv, eparams, options, cmaes, lastfitnesses, lastpvalues, g)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:] if sys.argv[0] == __file__ else sys.argv))
    except Exception as ex:
        print (ex, file=sys.stderr)
        import traceback
        traceback.print_exc(ex)
        sys.exit(1)
