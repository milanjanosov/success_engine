"""This file contains all functions needed for the evolution.

http://en.wikipedia.org/wiki/Genetic_algorithm

    - random initialization
    - run first generation
    - evaluate with fitness function, measure average/max fitness
    - select best candidates for next generation
    - mutation + crossover/recombination + regrouping + colonization-extinction + migration
    - run second generation
    - ...

"""

# external imports
import random
import sys
import os

# internal imports
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), "..")))
import util
import selection
import mutation
import crossover



def is_termination(eparams, fitnesses, g):
    """Return true if there should not be another generation to come.

    :param eparams:          the evolutionparams python module name
    :param fitnesses:        multi-objective fitness values of the last generation
    :param g:                the next generation number

    TODO: add other criteria based on:
        - A solution is found that satisfies minimum criteria
        - Allocated budget (computation time/money) reached
        - The highest ranking solution's fitness is reaching or has reached a plateau such that successive iterations no longer produce better results
        - Manual inspection
        - Combinations of the above

    """
    return g > eparams.generations


def generate_random_environments(eparams):
    """Generate a number of random environments.

    :param eparams:          the evolutionparams python module name

    """
    random.seed()
    params = [util.convert_string_to_param_t(env) for env in eparams.params_as_environment]
    pvalues = []
    for n in xrange(eparams.environments):
        pvalues.append([])
        for param in params:
            value = random.choice([f for f in util.frange(param.minv, param.maxv, param.step)])
            pvalues[-1].append(value)
    return (params, pvalues)


def generate_random_population(eparams):
    """Generate a number of random populations.

    :param eparams:          the evolutionparams python module name

    Return a list of params to evolve and a 2D-list of param values
    for each phenotype and each param.

    """
    random.seed()
    # get params to evolve
    params = get_params_to_evolve(eparams)
    # assign a param value for each param, for each phenotype
    pvalues = []
    for p in xrange(eparams.phenotypes):
        pvalues.append([])
        for param in params:
            value = random.random() * (param.maxv - param.minv) + param.minv
            pvalues[-1].append(value)
    return (params, pvalues)


def generate_population_history_0(eparams, lastpvalues, fitnesses):
    """Generate a new population based on the last one.

    :param eparams:      the evolutionparams python module name
    :param lastpvalues:  dict of evolvable pvalues for all phenotypes in the last generation.
    :param fitnesses:    dict of multi-objective fitnesses for all phenotypes in the last generation.

    Return a list of params to evolve and a 2D-list of param values
    for each phenotype and each param.

    """
    # if this is only an initialization, return a random population
    if not fitnesses or not lastpvalues:
        print ("  Note: no info from past, random population is generated.")
        return generate_random_population(eparams)

    # create single-valued fitnesses now
    # TODO: extend to multi-objective optimization
    sfitnesses = get_single_fitnesses(fitnesses)

    # if this is a next generation, create new population with selection + mutation + crossover
    # but first create the params list
    params = get_params_to_evolve(eparams)

    # elite selection
    elite_parents_p = selection.elite(sfitnesses, int(eparams.phenotypes * eparams.elite_fraction))
    elite_children_pvalues = [list(lastpvalues[p]) for p in elite_parents_p]

    # tournament selection + average crossover
    crossover_parents_p = selection.tournament(sfitnesses, eparams.tournament_size,
            int(eparams.phenotypes * eparams.crossover_fraction))
    crossover_children_pvalues = crossover.average(lastpvalues, crossover_parents_p,
            int(eparams.phenotypes * eparams.crossover_fraction))

    # gauss-noise mutation of parents
    mutation_parents_p = selection.fullrandom(sfitnesses, int(eparams.phenotypes * eparams.pure_mutation_fraction))
    mutation_children_pvalues = [list(lastpvalues[p]) for p in mutation_parents_p]
    mutation.gauss(params, mutation_children_pvalues, eparams.mutation_sigma, eparams.mutation_probability, eparams.mutation_sqrtN)
    # TODO TODO how many mutations are needed? One solution, if fitness is normalized in [0-1]:
    # AktualisMutEh = MaximalisMutEh * (1-GenereacioAtlagosFitnesse)

    # summarize all new generation
    pvalues = elite_children_pvalues + crossover_children_pvalues + mutation_children_pvalues

    # a quick error check on population size
    diff = len(pvalues) - eparams.phenotypes
    if diff < 0:
        print ("Warning: population size has been reduced! Adding %d new random phenotypes." % (-diff))
        a, b = generate_random_population(eparams)
        pvalues += b[0:-diff]
    elif diff > 0:
        print ("Warning: population size has grown! Removing %s random phenotypes." % diff)
        indices = random.sample(range(len(pvalues)), diff)
        pvalues = [i for j, i in enumerate(pvalues) if j not in indices]

    return (params, pvalues)


def generate_population(eparams, lastpvalues, fitnesses):
    """Generate a new population based on the last one.

    :param eparams:      the evolutionparams python module name
    :param lastpvalues:  dict of evolvable pvalues for all phenotypes in the last generation.
    :param fitnesses:    dict of multi-objective fitnesses for all phenotypes in the last generation.

    Return a list of params to evolve and a 2D-list of param values
    for each phenotype and each param.

    """
    # if this is only an initialization, return a random population
    if not fitnesses or not lastpvalues:
        print ("  Note: no info from past, random population is generated.")
        return generate_random_population(eparams)

    # create single-valued fitnesses now
    # TODO: extend to multi-objective optimization
    sfitnesses = get_single_fitnesses(fitnesses)

    # if this is a next generation, create new population with selection + crossover + mutation
    # but first create the params list
    params = get_params_to_evolve(eparams)

    # elite selection
    elite_parents_p = selection.elite(sfitnesses, int(eparams.phenotypes * eparams.elite_fraction))
    pvalues = [list(lastpvalues[p]) for p in elite_parents_p]

    # tournament selection of parents + crossover + mutation
    for i in xrange(int(eparams.phenotypes * eparams.crossover_fraction)):
        # get some parents
        crossover_parents_p = selection.tournament(sfitnesses, eparams.tournament_size, len(sfitnesses))
        # generate one child
        if eparams.crossover_operator == "uniform":
            crossover_children_pvalues = crossover.uniform(lastpvalues, crossover_parents_p, 1)
        elif eparams.crossover_operator == "average":
            crossover_children_pvalues = crossover.average(lastpvalues, crossover_parents_p, 1)
        else:
            raise ValueError("unknown crossover operator: %s" % eparams.crossover_operator)
        # mutate some genes of the new child
        mutation.gauss(params, crossover_children_pvalues, eparams.mutation_sigma, eparams.mutation_probability, eparams.mutation_sqrtN)
        # TODO TODO how many mutations are needed? One solution, if fitness is normalized in [0-1]:
        # AktualisMutEh = MaximalisMutEh * (1-GenereacioAtlagosFitnesse)
        # add new child to next generation
        pvalues += crossover_children_pvalues

    # pure mutation fraction (if needed, not necessary)
    mutation_parents_p = selection.fullrandom(sfitnesses, int(eparams.phenotypes * eparams.pure_mutation_fraction))
    mutation_children_pvalues = [list(lastpvalues[p]) for p in mutation_parents_p]
    mutation.gauss(params, mutation_children_pvalues, eparams.mutation_sigma, eparams.mutation_probability, eparams.mutation_sqrtN)
    pvalues += mutation_children_pvalues

    # a quick error check on population size
    diff = len(pvalues) - eparams.phenotypes
    if diff < 0:
        print ("Warning: population size has been reduced! Adding %d new random phenotypes." % (-diff))
        a, b = generate_random_population(eparams)
        pvalues += b[0:-diff]
    elif diff > 0:
        print ("Warning: population size has grown! Removing %s random phenotypes." % diff)
        indices = random.sample(range(len(pvalues)), diff)
        pvalues = [i for j, i in enumerate(pvalues) if j not in indices]

    return (params, pvalues)


def get_fitnesses(eparams, model, fitnessfunctionparam):
    """Return dictionary with keys as phenotypes and values as multi-objective fitnesses.

    :param eparams:               the evolutionparams python module name
    :param model:                 the name of the robotsim model that is used
    :param fitnessfunctionparam:  a single user parameter that is passed to the fitness functions

    """
    try:
        fitnesses = eval("eparams.fitness.fitness_%s" % model)(fitnessfunctionparam)
    except NameError as e:
        print ("Model type '%s' is not implemented yet (%s)." % (model, e))
        print eparams.fitness.fitness_template.__doc__
        return None
    return fitnesses


def get_single_fitnesses(fitnesses):
    """Return single-valued fitness list from multi-objective fitnesses."""
    return dict((p, get_single_fitness(fitnesses[p])) for p in fitnesses)


def get_single_fitness(fitnessdict):
    """Return a single fitness value from a multi-objective fitness dict.

    :param fitnessdict:  the dict of fitness values for a phenotype
    """
    retval = 1.0
    for x in fitnessdict:
        retval *= fitnessdict[x]
    return retval


def save_fitnesses(filename, eparams, fitnesses, g, pvalues=None):
    """Save multi-objective fitnesses to a file.

    :param filename:  save fitnesses to this file
    :param fitnesses: multi-objective fitness values for all phenotypes as a dict of dicts
    :param g:         the index of the generation
    :param pvalues:   pvalues of the given generation (None if not needed in output)

    """
    sfitnesses = get_single_fitnesses(fitnesses)
    f = open(filename, 'w')
    # write header
    f.write("\t".join( ["#g", "p", "fitness"] + ["fitness_%s" % x for x in sorted(fitnesses[0])] ))
    if pvalues is not None:
        params = get_params_to_evolve(eparams)
        f.write("\tpvalues\t" + "\t".join([p.name for p in params]))
    f.write("\n")
    # write data for each phenotype
    for p in sorted(sfitnesses, key=sfitnesses.get, reverse=True):
        f.write("\t".join( ["%d" % g, "%d" % int(float(p)), "%g" % sfitnesses[p]] + \
                ["%g" % fitnesses[p][x] for x in sorted(fitnesses[p])] ))
        if pvalues is not None:
            f.write("\t%d\t" % len(params) + "\t".join(str(pvalues[p][i]) for i in xrange(len(params))))
        f.write("\n")
    f.close()


def save_fitnesses_hack(filename, eparams, len_fitnesses, name, evaluation, pvalues, sfitness):
    """This is a hack to be able to add favorite and best solutions to the
    standard fitnesses file. Use in synchrony with save_fitnesses().
    """
    f = open(filename, 'a')
    f.write("\t".join( ["#%s_%d" % (name, (evaluation-1)/eparams.generations),
            str((evaluation-1) % eparams.generations), str(sfitness)] + ["-"]*len_fitnesses ))
    params = get_params_to_evolve(eparams)
    f.write("\t%d\t" % len(params) + "\t".join(str(pvalues[i]) for i in xrange(len(params))))
    f.write("\n")
    f.close()


def get_params_to_evolve(eparams):
    """Return the parameters that are to be evolved.

    :param eparams:  the evolutionparams python module name

    """
    return [util.convert_string_to_param_t(p) for p in eparams.params_to_evolve]


