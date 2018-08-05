"""This module contains scripts to measure stability of our solution."""

#external imports
import math
import numpy
import glob
import os
import itertools
import copy
import time

#internal imports
import evolution


def get_largest_empty_cube_around_center(center, points):
    """Define largest cube around center that does not contain any point
    from points.

    :param center:  N-dimensional vector of a center point
    :param points:  list of N-dimensional vectors to avoid

    :return half-width of cube
    """
    mindp = float("Inf")
    for p in points:
        dp = [abs(center[i] - p[i]) for i in range(len(center))]
        x = max(dp)
        if x < mindp:
            mindp = x
    return mindp


def get_some_kind_of_rectangle_around_center(center, points):
    """Second algo, does not work well in some cases, depreciated..."

    :param center:  N-dimensional vector of a center point
    :param points:  list of N-dimensional vectors to avoid

    :return [min, max] ranges in all dimensions
    """
    minmax = [[-float("Inf"), float("Inf")] for i in range(len(center))]
    for p in points:
        dp = [abs(center[i] - p[i]) for i in range(len(center))]
        x = max(dp)
        i = dp.index(x)
        # increase min
        if p[i] < center[i] and p[i] > minmax[i][0]:
            minmax[i][0] = p[i]
        # decrease max
        elif p[i] > center[i] and p[i] < minmax[i][1]:
            minmax[i][1] = p[i]
    return minmax


def _binsearch(points, axis, ordering, x):
    lo, hi = 0, len(points)-1
    while lo <= hi:
        mid = (lo + hi) >> 1
        midval = points[ordering[mid]][axis]
        if midval < x:
            lo = mid+1
        elif midval > x:
            hi = mid-1
        else:
            return mid
    return lo


def copy_nested_lists(list_of_lists):
    """Copies a list of lists up to two levels deep - faster than
    ``copy.deepcopy()``.
    """
    return [list(item) for item in list_of_lists]


def find_points_in_range(points, axis, ordering, lo, hi):
    """Finds the indexes of the points in the given points array where a
    given coordinate of the point falls between 'lo' (exclusive) and
    'hi' (exclusive).

    :param points: the array of points
    :param axis: the axis to consider
    :param ordering: index into the points array such that the array is
        ordered along the given axis if we follow this ordering
    :param lo: the lower bound (exclusive)
    :param hi: the upper bound (exclusive)

    :return: the indexes of the points that fall within the given range
        along the given axis
    """
    start = _binsearch(points, axis, ordering, lo)
    while start < len(points) and points[ordering[start]][axis] == lo:
        start += 1
    end = _binsearch(points, axis, ordering, hi)
    if points[ordering[end]][axis] == hi:
        while end > 0 and points[ordering[end]][axis] == hi:
            end -= 1
        if end >= 0 and points[ordering[end]][axis] != hi:
            end += 1
    return ordering[start:end]


def sorted_list_intersection(*args):
    """Intersection of multiple sorted lists with unique elements in each
    list. Returns a set with all elements that are contained in all of the
    lists.
    """
    smallest_list = set(min(args, key=len))
    for arg in args:
        smallest_list.intersection_update(arg)
    return smallest_list


def get_largest_empty_volume_around_center(center, points, r):
    """Permutate all solutions and get one with max volume.
    TODO: brute force method is very slow in large dimensions.

    :param center:  N-dimensional vector of a center point
    :param points:  list of N-dimensional vectors to avoid
    :param r:       minimal r to start with, e.g. largest r of a cube
                    around center, calculated by
                    get_largest_empty_cube_around_center()

    :return [min, max] ranges in all dimensions

    """
    display_interval = 1 # [s]
    starttime = time.time()
    num_dims = len(center)
    inf, neg_inf = float("Inf"), -float("Inf")
    lastdisplaytime = starttime
    maxvolume = 0
    bestmins, bestmaxs = [], []
    n_iters = math.factorial(num_dims)
    mins_orig = [center[i]-r for i in xrange(num_dims)]
    maxs_orig = [center[i]+r for i in xrange(num_dims)]
    previous_perm = None
    points_ordered_by_axes = [
        sorted(range(len(points)), key=lambda i: points[i][axis])
        for axis in xrange(num_dims)
    ]
    points_blocking_axis_at_start = [
        find_points_in_range(points, j, points_ordered_by_axes[j], mins_orig[j], maxs_orig[j])
        for j in xrange(num_dims)
    ]
    previous_perm = None
    print ("brute force comparison of", n_iters, "solutions in", num_dims, "dimensions...")
    for p_index, perm in enumerate(itertools.permutations(range(num_dims))):
        now = time.time()
        if now - lastdisplaytime > display_interval:
            print ("%1.1fs (@%1.2f%%, %1.2fs left, %1.2fs total): %s" % (
                    now-starttime,
                    100.0*p_index/n_iters,
                    (now-starttime)/p_index*(n_iters-p_index),
                    (now-starttime)/p_index*n_iters,
                    str(perm)))
            lastdisplaytime = now

        if previous_perm is not None:
            identical_prefix_length = num_dims
            for i in xrange(num_dims):
                if perm[i] != previous_perm[i]:
                    identical_prefix_length = i
                    break
        else:
            identical_prefix_length = 0

        axes_to_process = perm[identical_prefix_length:]
        if identical_prefix_length > 0:
            identical_axes = perm[:identical_prefix_length]
            volume = 1.0
            for i in identical_axes:
                volume *= maxs[i] - mins[i]
            for i in axes_to_process:
                mins[i] = mins_orig[i]
                maxs[i] = maxs_orig[i]
                points_blocking_axis[i] = list(points_blocking_axis_at_start[i])
        else:
            mins = mins_orig[:]
            maxs = maxs_orig[:]
            points_blocking_axis = copy_nested_lists(points_blocking_axis_at_start)
            volume = 1.0

        for i in axes_to_process:
            tempmin, tempmax = neg_inf, inf
            points_to_consider = sorted_list_intersection(
                *[points_blocking_axis[j] for j in xrange(num_dims) if j != i]
            )
            changed = False
            center_coord = center[i]
            for p_idx in points_to_consider:
                coord = points[p_idx][i]
                if coord < center_coord and coord > tempmin:
                    tempmin = coord
                    changed = True
                elif coord > center_coord and coord < tempmax:
                    tempmax = coord
                    changed = True
            if tempmin > neg_inf:
                mins[i] = tempmin
            if tempmax < inf:
                maxs[i] = tempmax
            volume *= maxs[i] - mins[i]
            if changed:
                points_blocking_axis[i] = \
                    find_points_in_range(points, i, points_ordered_by_axes[i], mins[i], maxs[i])
        if volume > maxvolume:
            maxvolume = volume
            bestmins[:] = mins
            bestmaxs[:] = maxs

        previous_perm = perm

    return zip(bestmins, bestmaxs)


def get_stability_range(eparams, allfitnesses, allpvalues, solution, threshold):
    """Return main axes of an N dimensional cube in the parameter space
    inside which all solutions have fitness higher than specified threshold,
    but not necessarily on the boundary. In other words, find maximal
    rectangular area of a clearing inside a forest.

    :param eparams:       the evolutionparams python module name
    :param allfitnesses:  allfitnesses[g][p] = dict of multi-objective fitness values for generation g, phenotype p
    :param allpvalues:    allpvalues[g][p][i] = param value of generation g, phenotype p and param index i
    :param solution:      the solution around which we analyse stability
    :param threshold:     fitness threshold above which we treat the system as stable

    TODO: this version is implemented only for overall/single fitness values
    TODO: so far only brute force method is used which becomes slow over 5-6
          parameter-space dimensions...

    """

    params = evolution.get_params_to_evolve(eparams)
    allsfitnesses = [evolution.get_single_fitnesses(allfitnesses[g]) for g in xrange(len(allfitnesses))]
    stability_range = [[params[i].minv, params[i].maxv] for i in xrange(len(params))] # [min, max]
    center = [(solution[i] - params[i].minv)/(params[i].maxv - params[i].minv) for i in xrange(len(params))]
    points = []
    for g in xrange(len(allfitnesses)):
        for p in allfitnesses[g]:
            # skip high fitness points
            if allsfitnesses[g][p] >= threshold:
                continue
            # add normalized point with low fitness to forest list
            points.append([(allpvalues[g][p][i] - params[i].minv)/(params[i].maxv - params[i].minv) for i in xrange(len(params))])
    # get stability range
    r = get_largest_empty_cube_around_center(center, points)
    stability_range = get_largest_empty_volume_around_center(center, points, r)
    # push it back from [0,1] to real parameters space
    for i in xrange(len(stability_range)):
        for j in xrange(2):
            stability_range[i][j] = params[i].minv + stability_range[i][j] * (params[i].maxv - params[i].minv)

    return stability_range
