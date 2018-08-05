"""This module contains crossover operators that can be used for parent gene
recombination."""

import random


def average(pvalues, parents, n):
    """Return children that average each gene from two random parents.

    :param pvalues:  the dict of list of param values for the whole population
    :param parents:  indices of the parents in the population
    :param n:        the number of children generated

    """
    children_pvalues = []
    for i in xrange(n):
        children_pvalues.append([])
        for j in xrange(len(pvalues[0])):
            a, b  = random.sample(parents, 2)
            children_pvalues[-1].append((pvalues[a][j] + pvalues[b][j])/2.0)
    return children_pvalues


def uniform(pvalues, parents, n):
    """Return children that inherit each gene from a random parent.

    :param pvalues:  the dict of list of param values for the whole population
    :param parents:  indices of the parents in the population
    :param n:        the number of children generated

    """
    children_pvalues = []
    for i in xrange(n):
        children_pvalues.append([])
        for j in xrange(len(pvalues[0])):
            a = random.sample(parents, 1)[0]
            children_pvalues[-1].append((pvalues[a][j]))
    return children_pvalues

