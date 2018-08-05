"""This is a quick test on the cma-es evolutionary algo,
which supposed to be a state-of-the-art solution, implemented by
https://www.lri.fr/~hansen/cmaesintro.html

Test functions for single-objective optimization problems:

    Possible 2-D models:

        - ackley     ( -100 <= x,y <= 100 )      max: f(0,0) = 0
        - eggholder  ( -100 <= x,y <= 100 )      max: f(100, 78.95154) = 959.6407
        - schaffer2  ( -100 <= x,y <= 100 )      max: f(0,0) = 0
        - cross      ( -100 <= x,y <= 100 )      max: f(+-13.4941, +-13.4941) = 2.06261

    Possible N-D models:

        - flat       ( -100 <= x_i <= 100 )   no max, f(x_i)   = 0
        - linear     ( -100 <= x_i <= 100 )      max: f(x_i=0) = 1
        - rosenbrock ( -100 <= x_i <= 100 )      max: f(x_i=1) = 0
        - sphere     ( -100 <= x_i <= 100 )      max: f(x_i=0) = 0

Test functions for multi-objective optimization problems:

    TODO

"""

# external imports
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

#internal imports
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), "../../..")))
import atlasz.util
import evolution.evolution
import evolution.plot
import evolution.cmainterface
import evolution.stability


def main(argv=[]):
    """Main entry point."""
    print "This is", __file__, "SVN revision:", atlasz.util.get_svn_info(__file__)['revision']
    # parse command line arguments
    argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    argparser.add_argument("-m", "--model", required=True, dest ="model", help="The name of the test model that is executed",
            choices=["ackley", "cross", "eggholder", "linear", "rosenbrock", "schaffer2", "sphere", "flat", "test", "mle", "mlesuccess"])

    # if arguments are passed to main(argv), parse them
    if argv:
        options = argparser.parse_args(argv)
    # else if called from command line or no arguments are passed to main, parse default argument list
    else:
        options = argparser.parse_args()

    # parse evolution parameters to be used
    eparams = importlib.import_module("evolutionparams")

    # initialize evolution
    g = 0
    fitnesses = None
    pvalues = None
    cmaes = evolution.cmainterface.init_cma(eparams, pvalues, fitnesses)

    # run evolution generation by generation
    allpvalues = []
    allfitnesses = []
    while not evolution.cmainterface.is_termination(cmaes, eparams, fitnesses, g):
        # create new generation
        solutions, pvalues = evolution.cmainterface.generate_population(cmaes, eparams)
        # evaluate new generation
        fitnesses = evolution.evolution.get_fitnesses(eparams, options.model, pvalues)
        # forward the fitnesses to the cma algo
        evolution.cmainterface.store_fitnesses(cmaes, fitnesses, solutions)
        # optional display info about the cma-es
        cmaes.disp()
        # save fitnesses for later use
        # evolution.evolution.save_fitnesses("fitnesses.txt", eparams, fitnesses, g, pvalues)
        # add pvalues and fitnesses for animation
        allpvalues.append(pvalues)
        allfitnesses.append(fitnesses)
        # iterate to next generation
        g += 1
    sfitnesses = evolution.evolution.get_single_fitnesses(fitnesses)
    print "  best solution from last generation:", pvalues[evolution.selection.elite(sfitnesses, 1)[0]]
    # result(self) method of cma.CMAEvolutionStrategy instance
    #     return ``(xbest, f(xbest), evaluations_xbest, evaluations, iterations, pheno(xfavorite), effective_stds)``
    result = cmaes.result
    print "  result (xbest, f(xbest), evaluations_xbest, evaluations, iterations, pheno(xfavorite), effective_stds):\n" \
          "     ", result
    mean_solution = evolution.cmainterface.get_mean_solution(cmaes, eparams)
    print "  mean solution:", mean_solution
    favorite_solution = evolution.cmainterface.get_favorite_solution(cmaes, eparams)
    print "  favorite solution:", favorite_solution
    best_solution = evolution.cmainterface.get_best_solution(cmaes, eparams)
    print "  best solution:", best_solution

    fitness_threshold = 350 # -m test
    print "\nStability range for solution", best_solution, "with fitness threshold", fitness_threshold, ":"
    stab_range = evolution.stability.get_stability_range(eparams, allfitnesses,
            allpvalues, best_solution[0], fitness_threshold)
    params = evolution.evolution.get_params_to_evolve(eparams)
    for i,param in enumerate(params):
        print param.name, "max-min:", stab_range[i][1] - stab_range[i][0], \
                "(%1.1f%%)" % (100*(stab_range[i][1] - stab_range[i][0])/(params[i].maxv - params[i].minv)), \
                "min:", stab_range[i][0], \
                "max:", stab_range[i][1]

    print "\nCreating animation, please wait...",
    outputdir = os.path.split(__file__)[0]
    evolution.plot.plot_bestfitnesses(allfitnesses, outputdir=outputdir)
    evolution.plot.plot_fitness_as_function_of_pvalue(eparams, allfitnesses,
            allpvalues, stability_range=stab_range, outputdir=outputdir)
    evolution.plot.plot_fitness_pvalues_animation(eparams, allfitnesses,
            allpvalues, model=options.model, x=0, y=1, stability_range=stab_range)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:])) # pass only real params to main
    except Exception as ex:
        print >>sys.stderr, ex
        import traceback
        traceback.print_exc(ex)
        sys.exit(1)
