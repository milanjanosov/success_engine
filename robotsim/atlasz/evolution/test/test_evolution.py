"""This is a quick test on the evolutionary algo itself.

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
        - mle        ( -100 <= x_i <= 100 )      max: f(x_i=0) = 0



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
import evolution.stability



argv = sys.argv[1:]


"""Main entry point."""
#    print "This is", __file__, "SVN revision:", atlasz.util.get_svn_info(__file__)['revision']
# parse command line arguments
argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
argparser.add_argument("-m", "--model", required=True, dest ="model", help="The name of the test model that is executed",
        choices=["mlesuccess_dir", "mlesuccess_art", "mlesuccess_prod", "mlesuccess_wr", "mlesuccess_comp", 
                 "mlesuccess_rock", "mlesuccess_pop", "mlesuccess_electro", "mlesuccess_folk", "mlesuccess_funk", "mlesuccess_jazz", "mlesuccess_class", "mlesuccess_hiphop", 
                 "mlesuccess_books",
                 "sci_mathematics",       "sci_theoretical_computer_science",    "sci_applied_physics", "sci_health_science", 
                 "sci_psychology",        "sci_space_science_or_astronomy",      "sci_geology",         "sci_biology", 
                 "sci_political_science", "sci_environmental_science",           "sci_engineering",     "sci_zoology", 
                 "sci_agronomy",          "sci_chemistry", "sci_physics"
                ])

argparser.add_argument("-min",  type=int, required=True, dest ="min")
argparser.add_argument("-max",  type=int, required=True, dest ="max")

# if arguments are passed to main(argv), parse them
if argv:
    options = argparser.parse_args(argv)
# else if called from command line or no arguments are passed to main, parse default argument list
else:
    options = argparser.parse_args()



runmin = options.min
runmax = options.max


for run in range(runmin, runmax):

    # parse evolution parameters to be used
    eparams = importlib.import_module("evolutionparams")

    # initialize evolution
    g = 0
    fitnesses = None
    pvalues = None
    maxs = -float("Inf")
    mins =  float("Inf")



    if not os.path.exists( 'Results_linrescaled/' + options.model ):
        os.makedirs( 'Results_linrescaled/' + options.model ) 









    outfolder = 'Results_linrescaled/' + options.model + '/run_' + str(run) 

    if not os.path.exists( outfolder ):
        os.makedirs( outfolder ) 



    print 'RUN   ', run

    # run evolution generation by generation

    allpvalues = []
    allfitnesses = []

    while not evolution.evolution.is_termination(eparams, fitnesses, g):





        # create next generation
        params, pvalues = evolution.evolution.generate_population(eparams, pvalues, fitnesses)
    #        envparams, envpvalues = evolution.evolution.generate_random_environments(eparams)
        # evaluate new generation
        fitnesses = evolution.evolution.get_fitnesses(eparams, options.model, pvalues)
        sfitnesses = evolution.evolution.get_single_fitnesses(fitnesses)
        if maxs < max(sfitnesses.values()): maxs = max(sfitnesses.values())
        if mins > min(sfitnesses.values()): mins = min(sfitnesses.values())

        print "  min fitness:", min(sfitnesses.values())
        print "  max fitness:", max(sfitnesses.values())
        print "  avg fitness:", sum(sfitnesses.values())/len(sfitnesses)
        print "  best solution:", pvalues[evolution.selection.elite(sfitnesses, 1)[0]]


        fout = open(outfolder + '/Generation_' + str(g) + '.dat'    , 'w' )
        fout.write( "  min fitness:"   + '\t' + str(min(sfitnesses.values())) + '\n')
        fout.write( "  max fitness:"   + '\t' + str(max(sfitnesses.values())) + '\n')
        fout.write( "  avg fitness:"   + '\t' + str(sum(sfitnesses.values())/len(sfitnesses)) + '\n')
        fout.write( "  best solution:" + '\t' + '\t'.join([str(fff) for fff in pvalues[evolution.selection.elite(sfitnesses, 1)[0]]]) + '\n')
        fout.close()



        # evolution.evolution.save_fitnesses("fitnesses.txt", eparams, fitnesses, g, pvalues)
        # add pvalues and fitnesses for animation
        allpvalues.append(pvalues)
        allfitnesses.append(fitnesses)
        # iterate to next generation
        g += 1
        



    fitness_threshold = mins + 0.8 * (maxs - mins)
    best_solution = pvalues[evolution.selection.elite(sfitnesses, 1)[0]]



## source /opt/virtualenv-python2.7/bin/activate




# screen -S evol15_mathematics 
# source /opt/virtualenv-python2.7/bin/activate  
# python2.7 test_evolution.py -m sci_mathematics -min 280 -max 300




# screen -S evol11_psychology 
# source /opt/virtualenv-python2.7/bin/activate
# python2.7 test_evolution.py -m sci_psychology -min 200 -max 220




# screen -S evol10_director
# source /opt/virtualenv-python2.7/bin/activate
# python2.7 test_evolution.py -m mlesuccess_dir  -min 180 -max 200



# screen -S  evol10_jazz
# source /opt/virtualenv-python2.7/bin/activate
# python2.7 test_evolution.py -m mlesuccess_jazz  -min 180 -max 200






## screen -S evol3_physics
## screen -S evol2_theoretical_computer_science
## screen -S evol4_applied_physics
## screen -S evol13_health_science
## screen -S evol_psychology
## screen -S evol2_space_science_or_astronomy
## screen -S evol2_geology
## screen -S evol2_political_science
## screen -S evol2_biology
# screen -S evol2_environmental_science
# screen -S evol12_engineering
# screen -S evol_zoology
# screen -S evol_agronomy
# screen -S evol12_chemistry



# source /opt/virtualenv-python2.7/bin/activate



# python2.7 test_evolution.py -m sci_mathematics
# python2.7 test_evolution.py -m sci_physics -min 100 -max 150
# python2.7 test_evolution.py -m sci_theoretical_computer_science -min 100 -max 200
# python2.7 test_evolution.py -m sci_applied_physics -min 230 -max 260



# python2.7 test_evolution.py -m sci_health_science -min 250 -max 260
# python2.7 test_evolution.py -m sci_psychology
# python2.7 test_evolution.py -m sci_space_science_or_astronomy -min 100 -max 200
# python2.7 test_evolution.py -m sci_geology -min 100 -max 200
# python2.7 test_evolution.py -m sci_biology -min 100 -max 130
# python2.7 test_evolution.py -m sci_political_science -min 100 -max 200
# python2.7 test_evolution.py -m sci_environmental_science -min 100 -max 150
# python2.7 test_evolu+tion.py -m sci_engineering -min 190 -max 210

# python2.7 test_evolution.py -m sci_zoology
# python2.7 test_evolution.py -m sci_agronomy
# python2.7 test_evolution.py -m sci_chemistry -min 270 -max 280






