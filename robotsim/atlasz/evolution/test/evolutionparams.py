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
params_to_evolve = [
    "f mu_N 0 5 0.1",
    "f mu_p 0 5 0.1",
    "f mu_Q 0 5 0.1",
    "f sigma_N 0.5 5 0.1",
    "f sigma_Q 0.5 5 0.1",
    "f sigma_p 0.5 5 0.1",
    "f sigma_pQ -0.5 0 0.1",
    "f sigma_pN -0.5 0 0.1",
    "f sigma_QN -0.5 0 0.1",
]





# the list of parameters that will serve as the random changing environment
params_as_environment = [
]


################################################################################
############################# FITNESS EVALUATIONS ##############################
################################################################################

# Gusz: won't work under 1000 fitness evaluations (generations * phenotypes)

# number of iterations/generations in the evolution
# typical: 50-100
generations = 50

# number of competing phenotypes/candidate solutions, i.e. population size
# typical: 100
phenotypes = 100

# number of random environments in one generation used for all phenotypes
# typical: ?????
environments = 50



#   best solution: [2.9469845405087103, 0.6661531950544164, 3.061895199470486, 3.457772970495238, 0.13187254140376486, 2.3897748255917066, -0.2618717294586237, -0.2401849413639054, -0.2887740188805981]
#   best solution: [2.9238488019954314, 0.6934528134509037, 1.3762426884368857, 2.467957598307776, 2.4027781754355777, 0.16120143963700614, -0.31326494121365295, -0.22302038864578722, -0.18301525882745617]



################################################################################
############################# FITNESS FUNCTIONS ################################
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
fitness = importlib.import_module('fitnessfunctions')


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
