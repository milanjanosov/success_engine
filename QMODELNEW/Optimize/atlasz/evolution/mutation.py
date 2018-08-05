"""This module contains mutation operators that can be used for increasing
gene variation."""

import random
import math


def gauss(params, pvalues, sigma, probability, sqrtN=False):
    """Add zero-mean gaussian noise to all of the genes in the population.

    :param params:       param_t descriptors for all params that are evolved
    :param pvalues:      the dict of list of param values for a population
    :param sigma:        the standard deviation of the added gaussian noise,
                         as a percentage of full param ranges
    :param probability:  mutation probability for each phenotype
    :param sqrtN:        should we have heuristic setting of noise level?

    Function does not return a value but changes pvalues list of list.

    """
    ranges = [p.maxv - p.minv for p in params]
    if sqrtN:
        C = 1/math.sqrt(len(params))
        sigma = 1
    else:
        C = 1
    for j in xrange(len(pvalues)):
        if random.random() > probability:
            continue
        for k in xrange(len(params)):
            pvalues[j][k] += C * random.gauss(0, sigma * ranges[k])
            if pvalues[j][k] > params[k].maxv:
                pvalues[j][k] = params[k].maxv
            if pvalues[j][k] < params[k].minv:
                pvalues[j][k] = params[k].minv


def uniform(params, pvalues, sigma, probability):
    """Add uniform white noise to all of the genes in the population.

    :param params:       param_t descriptors for all params that are evolved
    :param pvalues:      the dict of list of param values for a population
    :param sigma:        the (half)range of the white noise,
                         as a percentage of full param ranges (e.g 0.1 = +=0.1)
    :param probability:  mutation probability for each phenotype

    Function does not return a value but changes pvalues list of list.

    """
    ranges = [p.maxv - p.minv for p in params]
    for j in xrange(len(pvalues)):
        if random.random() > probability:
            continue
        for k in xrange(len(params)):
            pvalues[j][k] += (2*random.random()-1) * sigma * ranges[k]
            if pvalues[j][k] > params[k].maxv:
                pvalues[j][k] = params[k].maxv
            if pvalues[j][k] < params[k].minv:
                pvalues[j][k] = params[k].minv
