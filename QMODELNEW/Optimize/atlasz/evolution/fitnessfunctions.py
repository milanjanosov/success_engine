"""This file should contain all model specific fitness functions that are
using the output of atlasz_batch_postprocess.py"""


# external import
import math

# internal import (see math_utils.py)
import math_utils


################################################################################
######################## MODEL-SPECIFIC FITNESS FUNCTIONS ######################
################################################################################


def fitness_template(pp):
    """All model fitness functions should look like this:

        def fitness_mymodel(pp):

    Arguments:
    ----------

    pp - the model-specific output of atlasz_batch_postprocess.py,
    parsed and stored in a namedtuple that is defined in
    atlasz_evolve_robotsim.py, in get_fitnesses_from_a_generation()

    the fitnessparameter_t type pp structure currently has two elements:

    .header - postprocess headers
    .data   - postprocess data

    Usage:
    ------

    Use any code to define a fitness list for each phenotype denoted by
    column 'p' in the batch_postprocess output.

    Return a dictionary that contains all phenotype (p) values as
    integer keys and dict of multi-objective fitnesses as float values,
    preferably normalized into [0,1]

    """
    pass


def fitness_traffic(pp):
    """Fitness function for traffic model data.

    Elements of the fitness function are the following:

        * ratio of collisions
        * effective velocity / flocking velocity
        * (a_max - acceleration) / a_max

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get interesting data elements
    j_p                 = pp.header.index("p")
    j_collisions_avg    = pp.header.index("ratio_of_collisions_avg")
    j_effective_vel_avg = pp.header.index("effective_vel_avg_(cm/s)_avg")
    j_v_flock_avg       = pp.header.index("v_flock_(cm/s)_avg")
    j_a_max             = pp.header.index("a_max_(cm/s/s)_avg")
    j_a_avg             = pp.header.index("acc_abs_(cm/s/s)_avg")
    # calculate fitness for each phenotype

    for i in xrange(len(pp.data)):

        # Below, we used sigma = 0.00003 for the characteristic maximal collision risk
        # TODO: Maybe sigma has to be defined as a function of the number of agents...
        # delta_coll = (math.exp ( - (math.pow (pp.data[i][j_collisions_avg], 2.0)) / 1.6e-09))
        a = 0.000002
        delta_coll = (a*a / (math.pow(pp.data[i][j_collisions_avg] + a, 2)))

        # calculate normalized effective velocity
        v_eff_norm = pp.data[i][j_effective_vel_avg] / pp.data[i][j_v_flock_avg]

        # calculate normalized acceleration
        acc_norm = (pp.data[i][j_a_max] - pp.data[i][j_a_avg]) / pp.data[i][j_a_max]

        fitnesses[int(pp.data[i][j_p])] = {
                "v_eff_norm": v_eff_norm,
                "acc_norm": acc_norm,
                "collision": delta_coll,
        }

    return fitnesses


def fitness_traffic_slowdown(pp):
    return fitness_traffic(pp)

def fitness_traffic_circle(pp):
    return fitness_traffic(pp)


def fitness_random_targets(pp):
    """Fitness function for "random_targets" model data.

    Elements of the fitness function are the following:

        * ratio of collisions
        * time of mission completion
        * average distance from targets

    """
    # initialize empty fitness dictionary
    fitnesses = {}
    # get interesting data elements
    j_p                 = pp.header.index("p")
    j_collisions_avg    = pp.header.index("ratio_of_collisions_avg")
    j_dist              = pp.header.index("dist_from_target_avg")
    j_static_dist       = pp.header.index("static_dist_from_target_avg")
    j_timeneeded_avg    = pp.header.index("time_needed_for_mission_completion_avg")
    j_reltime_avg       = pp.header.index("relative_time_of_mission_completion_avg")
    # calculate fitness for each phenotype

    for i in xrange(len(pp.data)):

        # Below, we used sigma = 0.00003 for the characteristic maximal collision risk
        # TODO: Maybe sigma has to be defined as a function of the number of agents...
        # delta_coll = (math.exp ( - (math.pow (pp.data[i][j_collisions_avg], 2.0)) / 1.6e-09))
        a = 0.000002
        delta_coll = (a*a / (math.pow(pp.data[i][j_collisions_avg] + a, 2)))

        # calculate time fitnesses as 0 if mission is not completed, 1 if
        # completed in minimum theoretical time, linear in between
        reltime = (1 - pp.data[i][j_reltime_avg]) / (1 - pp.data[i][j_timeneeded_avg])

        # calculate distance fitness as 0 if average static distance is over 10m,
        # 1 if average distance from target is 0m, linear between
        # note the [cm] -> [m] conversion
        dist = max(0, 1000 - pp.data[i][j_static_dist])/1000.0

        fitnesses[int(pp.data[i][j_p])] = {
                "reltime": reltime,
                "disttotrg": dist,
                "collision": delta_coll,
        }

    return fitnesses


def fitness_spp_evol(pp):
    """Fitness function for the spp evol algorithm.

    The exact form of fitness function:

    Elements of the fitness function are the following:

        * Average of velocity correlation
        * delta (Ratio of collisions)
        * A "smooth Heaviside-theta" of the average velocity magnitude (should be centered at V_Flock)
        * delta (Average of distance from arena)
        * delta (StDev of distance from arena)

    where "delta" is an approximation of Dirac-delta function
    (can be Gauss, or step-function, or anything).
    Note that this "delta" should not be normalized, its maximum value should be 1.0.
    fitness is in the interval [0, 1].

    Smooth version of the Heaviside function can be e.g. the Fermi-Dirac distribution f(E).

    """

    # initialize empty fitness dictionary
    fitnesses = {}
    # get interesting data elements
    j_p                     = pp.header.index("p")
    j_vel_corr_avg          = pp.header.index("cluster_vel_corr_avg")
    j_collisions_avg        = pp.header.index("ratio_of_collisions_avg")
    j_velocity_magn_avg     = pp.header.index("vel_abs_(cm/s)_avg")
    j_cluster_minsize       = pp.header.index("min_cluster_size_avg")
    j_agents_not_in_cluster = pp.header.index("agents_not_in_cluster_avg")

    j_distance_from_arena_avg = pp.header.index("distance_from_arena_(cm)_avg")
    #j_distance_between_neighbours = pp.header.index("distance_between_neighbours_(cm)_avg")
    j_V_Flock = pp.header.index("v_flock_(cm/s)_avg")
    j_R_0 = pp.header.index("R_0_(cm)_avg")

    for i in xrange(len(pp.data)):

        # Below, we used sigma = 0.00003 for the characteristic maximal collision risk
        # TODO: Maybe sigma has to be defined as a function of the number of agents...

        a = 0.00003
        delta_coll = math_utils.peak(pp.data[i][j_collisions_avg], a)
        #(a*a / (math.pow(pp.data[i][j_collisions_avg] + a, 2.0)))

        # sigmoid curve for calculating 'velocity magnitude' fitness
        V_flock = pp.data[i][j_V_Flock]
        vel_tolerance = 150.0/400.0 * V_flock
        vel_magn_coeff = 1.0 - math_utils.sigmoid(
                pp.data[i][j_velocity_magn_avg], vel_tolerance, V_flock, 1.0)
        #vel_magn_coeff = 1.0 - 1.0 / (math.exp ((pp.data[i][j_velocity_magn_avg] - (V_flock - 150.0)) / 50.0) + 1.0)

        # Below, we used sigma = 200.0 cm for the stdev of the Gauss-function.
        sigmasquare = 40000.0
        delta_distance_from_arena = math_utils.gauss(
                pp.data[i][j_distance_from_arena_avg], sigmasquare, 0.0, 1.0)
        #delta_distance_from_arena = (math.exp ( - (math.pow (pp.data[i][j_distance_from_arena_avg], 2.0)) / sigmasquare))

        # Velocity correlation should be saturated at 0
        correlation_coeff = 0
        if pp.data[i][j_vel_corr_avg] > 0:
            correlation_coeff = pp.data[i][j_vel_corr_avg]

        # Cluster size should be at least N / 5
        N = 10.0
        min_size = N / 5.0
        if pp.data[i][j_cluster_minsize] > min_size:
            min_cluster_size_fitness = 1.0
        else:
            min_cluster_size_fitness = pp.data[i][j_cluster_minsize] * 2.0 / N

        # Independent agents should not exist
        agents_not_in_cluster_fitness = math_utils.peak(
                pp.data[i][j_agents_not_in_cluster], min_size)
        #agents_not_in_cluster_fitness = ((min_size) * (min_size) / (math.pow(pp.data[i][j_agents_not_in_cluster] + min_size, 2)))

        # Distance between nearest neighbours should be around R_0
        # tolerance = 300.0
        # R_0 = float (pp.data[i][j_R_0])
        # temp_distance = math.fabs(pp.data[i][j_distance_between_neighbours] - R_0)
        # distance_between_neighbours_fitness = math_utils.gauss (temp_distance, tolerance, 0.0, 1.0)

        # Save everything
        fitnesses[int(pp.data[i][j_p])] = {
                "vel_corr":  correlation_coeff,
                "collision": delta_coll,
                "vel_magn":  vel_magn_coeff,
                "distance_from_arena_avg": delta_distance_from_arena,
                "min_cluster_size": min_cluster_size_fitness,
                "agents_not_in_cluster": agents_not_in_cluster_fitness,
                #"distance_between_neighbours": distance_between_neighbours_fitness
        }

    return fitnesses

def fitness_spp_int_hier(pp):
    return fitness_spp_evol(pp)

def fitness_chasing_chaser(pp):
    """Fitness function for the chasing algorithm from the chaser's viewpoint."""
    # initialize empty fitness dictionary
    fitnesses = {}
    # get interesting data elements
    j_p                     = pp.header.index("p")
    j_eff_chasers           = pp.header.index("effectiveness_chasers_avg")
    # store fitness value
    for i in xrange(len(pp.data)):
        fitnesses[int(pp.data[i][j_p])] = {
                "eff_chasers":  pp.data[i][j_eff_chasers],
        }
    return fitnesses


def fitness_chasing_escaper(pp):
    """Fitness function for the chasing algorithm from the escaper's viewpoint."""
    # initialize empty fitness dictionary
    fitnesses = {}
    # get interesting data elements
    j_p                     = pp.header.index("p")
    j_eff_escapers          = pp.header.index("effectiveness_escapers_avg")
    # store fitness value
    for i in xrange(len(pp.data)):
        fitnesses[int(pp.data[i][j_p])] = {
                "eff_escapers":  pp.data[i][j_eff_escapers],
        }
    return fitnesses


def fitness_chasing(pp):
#    return fitness_chasing_chaser(pp)
    return fitness_chasing_escaper(pp)
