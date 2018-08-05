"""This module contains crossover operators that can be used for parent gene
recombination."""

import random
import numpy


def fitness_proportionate(sfitnesses, n):
    """Get selected phenotypes through fitness proportionate selection.

    :param sfitnesses:    dict of single fitnesses for all phenotypes
    :param n:             select n phenotypes

    """
    selected = []
    sumfitness = sum(sfitnesses.values())
    cumfitness = [sfitnesses[0]]
    for i in xrange(1, len(sfitnesses)):
        cumfitness.append(cumfitness[-1] + sfitnesses[i])
    for i in xrange(n):
        f = random.random()*sumfitness
        j = 0
        while sfitnesses[j] < f:
            j += 1
        selected.append(j)
    return selected


def tournament(sfitnesses, tournament_size, n):
    """Get selected phenotypes through tournament selection.

    :param sfitnesses:       dict of single fitnesses for all phenotypes
    :param tournament_size:  size of tournaments
    :param n:                select n phenotypes

    """
    selected = []
    rangep = range(len(sfitnesses))
    for i in xrange(n):
        tournament = random.sample(rangep, tournament_size)
        argmax = numpy.argmax([sfitnesses[t] for t in tournament])
        selected.append(tournament[argmax])
    return selected


def stochastic_acceptance(sfitnesses, n):
    """Get selected phenotypes through stochastic acceptance selection.

    :param sfitnesses:   dict of single fitnesses for all phenotypes
    :param n:            select n phenotypes

    """
    selected = []
    fmax = max(sfitnesses.values())
    while len(selected) < n:
        i = random.randint(0, len(sfitnesses)-1)
        if random.random() <= sfitnesses[i]/fmax:
            selected.append(i)
    return selected


def elite(sfitnesses, n):
    """Get selected phenotypes through elite selection.

    :param sfitnesses:   dict of single fitnesses for all phenotypes
    :param n:            select n phenotypes

    """
    indices = sorted(sfitnesses, key=sfitnesses.get, reverse=True)
    # print [sfitnesses[indices[i]] for i in xrange(n)]
    return [indices[i] for i in xrange(n)]


def fullrandom(sfitnesses, n):
    """Get selected phenotypes through a fully random process.

    :param sfitnesses:   dict of single fitnesses for all phenotypes
    :param n:            select n phenotypes

    """
    return random.sample(range(len(sfitnesses)), n)
