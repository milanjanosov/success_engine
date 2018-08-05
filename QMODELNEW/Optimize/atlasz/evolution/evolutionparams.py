"""This file should contain all settings for the genetic algo.

param format in general (see util.param_t):  f|u|i name min max step

where f-->flockingparams.dat, u-->unitparams.dat, i-->initparams.dat

"""

import importlib


################################################################################
################################# PARAMETERS ###################################
################################################################################


# the list of parameters/properties/genotypes that will be mutated and evolved
# Gusz: Start with a "wish list", define as much parameters as possible
params_to_evolve_test = [
    "f D 0 1 0.1",
    "f R_0 0 2000 10",
    "f C_Frict 0 200000 1000",
]

#
# See parameters/flockingparams_traffic.dat for further details!
#
params_to_evolve_traffic_circle = [
    # repulsion
    "f V_Rep 10 800 10",
    "f Gamma_Rep 10.0 4000 10",
    "f R_0 10.0 8000 10",
    "f V_Max_Rep 0 1200 10",
    # friction
    "f C_Frict 0 2 0.05",
    "f Gamma_Frict 10.0 3000 10",
    "f Alpha_Frict -5.0 5.0 0.05",
    "f R_0_OffSet_Frict -3000 6000 10",
    "f V_Max_Frict 0 1200 10",
    # circle
    "f CircleRadiusCoeff 0.0 1.0 0.01",
    "f TangentialVelocity 0.0 800.0 10.0",
    "f ToRingVelocity 0.0 800.0 10.0",
    "f VisionRange 0.0 8000.0 10.0"
]

#
# See parameters/flockingparams_slowdown.dat for more details!
#
params_to_evolve_traffic_slowdown = [
    # repulsion
#    "f V_Rep 0 3000 10",
    "f R_0 0 6000 10",
    "f Slope_Rep 0 5 0.05",
    # friction
#    "f C_Frict 0 5 0.01",
    "f V_Frict 0 500 10",
    "f R_0_Offset_Frict -3000 6000 10",
#    "f Slope_Frict 0 10 0.05",
#    "f Acc_Frict 0 1000.0 0.05",
    # jam and obstacle
    "f Pliancy -6 6 0.1",
    "f Anisotropy 0.0 1.0 0.01",
    "f QueueingInteractionRange 0.01 8000.0 100.0",
    # target tracking / slowdown
    "f R_0_Offset_Safety -3000 3000 10",
    "f Tangentiality 0. 1. 0.01",
    # hierarchy
    "f HierarchyLevel 0. 20. 0.01",
    "f HierarchyType 0 18 1",
    "f R_Danger 0 6000 10",
]

#
# See parameters/flockingparams_random_targets.dat for more details!
#
params_to_evolve_random_targets = [
    # repulsion
    "f V_Rep 10 5000 10",
    "f R_0 400 900 10",
    "f Slope_Rep 0 1000 0.05",
    # friction
    "f V_Frict 10 5000 10",
    "f R_0_Offset_Frict -400 2000 10",
    "f Slope_Frict 0 1000 0.05",
    "f Alpha_Frict -5.0 5.0 0.05",
    # general
    "f V_0 10 600 10",
    "f V_Max 10 800 10",
#    # target tracking
#    "f Acc_TargetTracking 0 600 1",
#    "f Slope_TargetTracking 0 1000 0.05",
]


#
# See parameters/flockingparams_evol_spp.dat for further details!
#

params_to_evolve_spp = [
    # repulsion
    "f V_Rep 10 3000 10",
    "f R_0 10 8000 10",
    "f Slope_Rep 0 1000 0.05",
    # friction
    "f V_Frict 10 3000 10",
    "f R_0_Offset_Frict -3000 6000 10",
    "f Slope_Frict 0 1000 0.05",
    # shill
    "f C_Shill 0 5.0 0.5",
    "f Alpha_Shill -1.0 5.0 0.05",
    "f V_Shill 0 1200 10",
    "f Gamma 10.0 6000 10",
]

#
# See parameters/flockingparams_evol_spp.dat for further details!
#

params_to_evolve_spp_cutoff = [
    # repulsion
    "f V_Rep 10 3000 10",
    "f R_0 10 8000 10",
    "f Slope_Rep 0 1000 0.05",
    "f V_Max_Rep 0 1200 10",
    # friction
    "f V_Frict 10 3000 10",
    "f R_0_Offset_Frict -3000 6000 10",
    "f Slope_Frict 0 1000 0.05",
    "f Alpha -5.0 5.0 0.05",
    "f V_Max_Frict 0 1200 10",
    # shill
    "f C_Shill 0 100 0.5",
    "f Alpha_Shill -5.0 5.0 0.05",
    "f V_Shill 0 1200 10",
    "f Gamma 10.0 6000 10",
]

params_to_evolve_spp_linsqrt_4mps = [
    # repulsion
#    "f V_Rep 10 5000 10",
    "f R_0 10 6000 10",
    "f Slope_Rep 0 5 0.05",
    # friction
    "f V_Frict 0 500 10",
    "f C_Frict 0 1 0.1",
    "f R_0_Offset_Frict 0 6000 10",
    "f Slope_Frict 0 10 0.05",
    "f Acc_Frict 0 1000 0.05",
    # shill
    "f V_Shill 400 2000 0.5", # note: upper limit raised from 1500 on 2018.01.19.
    "f Acc_Shill 0 1000 0.05",
    "f Slope_Shill 0 10 0.05",
    "f R_0_Shill 0 2000 10",
]

params_to_evolve_spp_linsqrt_6mps = [
    # repulsion
#    "f V_Rep 10 5000 10",
    "f R_0 10 6000 10",
    "f Slope_Rep 0 5 0.05",
    # friction
    "f V_Frict 0 500 10",
    "f C_Frict 0 1 0.1",
    "f R_0_Offset_Frict 0 6000 10",
    "f Slope_Frict 0 10 0.05",
    "f Acc_Frict 0 1000 0.05",
    # shill
    "f V_Shill 600 2500 0.5", # note: upper limit raised from 2000 on 2018.01.19.
    "f Acc_Shill 0 1000 0.05",
    "f Slope_Shill 0 10 0.05",
    "f R_0_Shill 0 2000 10",
]

params_to_evolve_spp_linsqrt_8mps = [
    # repulsion
#    "f V_Rep 10 5000 10",
    "f R_0 2000 8000 10", # note: upper limit raised from 6000 on 2018.01.24. and lower from 1000 on 2018.01.25.
    "f Slope_Rep 0 5 0.05",
    # friction
    "f V_Frict 0 500 10",
    "f C_Frict 0 1 0.1",
    "f R_0_Offset_Frict 0 6000 10",
    "f Slope_Frict 0 10 0.05",
    "f Acc_Frict 0 1000 0.05",
    # shill
    "f V_Shill 1500 3500 0.5", # note: upper limit raised from 2500 on 2018.01.19. and from 3000 on 2018.01.24., lower from 800 on 2018.01.25.
    "f Acc_Shill 0 1000 0.05",
    "f Slope_Shill 0 10 0.05",
    "f R_0_Shill 0 2000 10",
]

params_to_evolve_spp_linsqrt_16mps = [
    # repulsion
#    "f V_Rep 10 5000 10",
    "f R_0 2000 12000 10",
    "f Slope_Rep 0 5 0.05",
    # friction
    "f V_Frict 0 500 10",
    "f C_Frict 0 1 0.1",
    "f R_0_Offset_Frict 0 8000 10",
    "f Slope_Frict 0 10 0.05",
    "f Acc_Frict 0 1000 0.05",
    # shill
    "f V_Shill 1000 7000 0.5",
    "f Acc_Shill 0 1000 0.05",
    "f Slope_Shill 0 10 0.05",
    "f R_0_Shill 0 5000 10",
]

params_to_evolve_spp_linsqrt_32mps = [
    # repulsion
#    "f V_Rep 10 5000 10",
    "f R_0 2000 20000 10",
    "f Slope_Rep 0 5 0.05",
    # friction
    "f V_Frict 0 500 10",
    "f C_Frict 0 1 0.1",
    "f R_0_Offset_Frict 0 10000 10",
    "f Slope_Frict 0 10 0.05",
    "f Acc_Frict 0 1000 0.05",
    # shill
    "f V_Shill 5000 12000 0.5",
    "f Acc_Shill 0 1000 0.05",
    "f Slope_Shill 0 10 0.05",
    "f R_0_Shill 0 5000 10",
]

params_to_evolve_spp_int_hier = [
    # repulsion
    "f R_0 500 8000 10",
    "f Slope_Rep 0 5 0.05",
    # friction
    "f V_Frict 0 500 10",
    # "f C_Frict 0 1 0.1",
    "f R_0_Offset_Frict 0 6000 10",
    "f Slope_Frict 0 10 0.05",
    "f Acc_Frict 0 1000 0.05",
    # shill
    "f V_Shill 1500 3500 0.5", # note: upper limit raised from 2500 on 2018.01.19. and from 3000 on 2018.01.24., lower from 800 on 2018.01.25.
    "f Acc_Shill 0 1000 0.05",
    "f Slope_Shill 0 10 0.05",
    "f R_0_Shill 0 2000 10",
    # interaction hierarchy
    "f W_R -1 1 0.05",
    "f W_A -1 1 0.05",
    "f R_A -1 1 0.05",
]


params_to_evolve_chasing_chaser = [
#    "f CFrict 0 10 0.1", # ???
    "f ChasersInteractionStrength 0 1 0.1",
    "f MaxPredictionTime 0 20 1",
    "f ChasersInteractionDistance 0 30000 1000",
#    "f SightRangeChaser, 0, 150000 1000",
    "i NumberOfAgents 2 20 1",
]

params_to_evolve_chasing_escaper = [
#    "f CFrict 0 10 0.1", # ???
    "f ZigZagFactor 0 10000 100",
    "f PanicTreshold 0 1 0.1",
    "f EscapersSensitivityRange 0 30000 1000",
#    "f SightRangeEscaper 0 150000 1000",
]

#params_to_evolve = params_to_evolve_spp_10agents
#params_to_evolve = params_to_evolve_spp_linsqrt_4mps
params_to_evolve = params_to_evolve_spp_linsqrt_6mps
params_to_evolve = params_to_evolve_spp_linsqrt_8mps
#params_to_evolve = params_to_evolve_spp_linsqrt_16mps
#params_to_evolve = params_to_evolve_spp_linsqrt_32mps
#params_to_evolve = params_to_evolve_chasing_chaser
#params_to_evolve = params_to_evolve_chasing_escaper
#params_to_evolve = params_to_evolve_random_targets
#params_to_evolve = params_to_evolve_traffic_slowdown

# the list of parameters that will serve as the random changing environment
params_as_environment = [
#    "f V_Flock 2 8 0.1",
#    "f ArenaSize 10000 100000 1000",
]


################################################################################
############################# FITNESS EVALUATIONs ##############################
################################################################################

# Gusz: won't work under 1000 fitness evaluations (generations * phenotypes)

# number of iterations/generations in the evolution
# typical: 50-100
generations = 100

# number of competing phenotypes/candidate solutions, i.e. population size
# typical: 100
phenotypes = 100

# number of random environments in one generation used for all phenotypes
# typical: ?????
environments = 1


################################################################################
####################i######### FITNESS FUNCTIONS ################################
################################################################################

# Gusz: multiobjective/multicriteria optimization:
#     a) independent fitnesses, arithmetic operation, 1 final fitness function
#        that is e.g. a linear combination of the partial fitnesses
#     b) keep independent parts separate, make evolutionary algo that treats this:
#        MOEA multi-objective-evolutionary-algorithm, e.g.:
#        NSGA2-3: dominancy measure: phenotype is winner in how many fitness parts?
#        pareto-front: phenotypes that are not dominated by anyone.
#        manual selection from pareto front who is the best

# the module that contains the fitness functions to be used
fitness = importlib.import_module('evolution.fitnessfunctions')


################################################################################
########################### REPRODUCTION FRACTIONS #############################
################################################################################

# Fractions of the operators for the next generation
# Always keep the sum of the three lines below at exactly 1!
elite_fraction         = 0.05
crossover_fraction     = 0.95
pure_mutation_fraction = 0.00


################################################################################
################################## SELECTION ###################################
################################################################################


# Parent selection = tournament selection
# http://en.wikipedia.org/wiki/Tournament_selection
# Set selection pressure with tournament size. Larger value is more pressure.
# typical: 2-3
tournament_size = 3


################################################################################
################################## CROSSOVER ###################################
################################################################################

# choose from uniform/average
crossover_operator = "average"
#crossover_operator = "uniform"


################################################################################
################################## MUTATION ####################################
################################################################################

# Probability of a mutation to happen in a gene
# Gusz: in best algos all genes mutate and values are taken from gauss noise
# Gabor: a limited but significant ratio of probability enhances random search,
#        reduces average but not best fitnesses
mutation_probability = 0.4

# mutations are treated as zero-mean Gaussian / uniform noise over the old parameter value.
# The standard deviation of the normal distribution / half-width of white noise is given in
# percentage of the actual parameter range
# Gusz: this is a very sensitive parameter, needs fine tuning.
# Gusz: Usually small is good
# Gabor/Anna: it is better to have larger value and limited mutation probability
mutation_sigma = 0.3

# Gusz: Have noise = 1/sqrt(len(params_to_evolve)) * NormalDistribution(0,1)
# If we need this, set this param to True. Note that it will override sigma setting
mutation_sqrtN = False

# Useful stuff:
# cma-es
# http://en.wikipedia.org/wiki/CMA-ES
# https://www.lri.fr/~hansen/cmaes_inmatlab.html#python
# http://www.mathworks.com/help/gads/how-the-genetic-algorithm-works.html
