"""Postprocess atlasz batch robotsim threads and collect output data.

Usage: __file__ inputdir model [label]

   where 'model' is the name of the simulation model,
   'inputdir' is a directory containing those subdirs
   that contain outputs of the data to be summarized
   (output of atlasz_batch_robotsim.py in standard format),
   and 'label' is the optional label attached to the selected simulations

For details on model-specific postprocessing implementation, check out
the process_*() functions.

"""

from __future__ import print_function
import os, sys, glob, collections, math

import atlasz.util


################################################################################
######################### COMMON OBJECT DEFINITIONS ############################
################################################################################

param_t = collections.namedtuple("param_t", ["param", "value"])

class data_t(object):
    """Holder object for averaged output values to be saved from summarized, averaged simulations.

    Class contains the following variables:
        .avg: list of average of output values over multiple simulations with same parameters (n=0,1,2,...)
        .stv: list of standard variation of output values over multiple simulations with same parameters (n=0,1,2,...)
        .n:   list of the number of data points in the running averages for all output values
        .headers: list of names of output values in the averaging
        .paramlist_t: namedtuple of the actual param list that was changed during the simulation

    """

    def __init__(self):
        """Initialize the class with empty variables.

        .avg, .stv and .n dictionaries should be addressed with the same .paramlist_t keys.
        .headers list should have the same size as .avg, .stv and .n

        """
        self.paramlist_t = None
        self.avg = {}
        self.stv = {}
        self.n = {}
        self.headers = []

    def define_paramlist_t(self, pvlist):
        """Define self.paramlist_t from param_t list"""
        self.paramlist_t = collections.namedtuple("paramlist_t", [p.param for p in pvlist])

    def is_compatible(self, pvlist):
        """Check whether param_t list is compatible with inner paramlist_t data."""
        a = [p.param for p in pvlist]
        b = list(self.paramlist_t._fields)
        if a == b:
            return True
        print ("  Warning:", a, "not compatible with", b)
        return False

    def add_value(self, plist, i, value):
        """Add new value to the i-th output value averaging of the plist paramlist_t params.

        Running average and standard deviation calculation source:
        http://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods

        Note: standard deviation at all times is sqrt(stv/n)

        """
        # check for new plist entry and initialize if needed
        if plist not in self.n.keys():
            self.avg[plist] = [0]
            self.stv[plist] = [0]
            self.n[plist] = [0]

        # check for dinamic new size and initialize if needed
        j = i - len(self.n[plist]) + 1
        if j > 0:
            self.avg[plist] += [0]*j
            self.stv[plist] += [0]*j
            self.n[plist] += [0]*j

        self.n[plist][i] += 1
        prevavg = self.avg[plist][i]
        self.avg[plist][i] += (value - prevavg)/self.n[plist][i]
        self.stv[plist][i] += (value - prevavg)*(value-self.avg[plist][i])


################################################################################
########################### COMMON PROCESSING FUNCTIONS ########################
################################################################################


def process_collisions(inputdir, plist, data, settings, column_offset=0):
    """ Processes collision data.
    Can be useful for all of the algorithms.

    :param column_offset: specify the first usable column in the output structure."""
    new_columns = ["ratio_of_collisions", "collisions"]
    # setting up header line
    if len(data.headers) < column_offset:
        print ("Warning, column_offset is too large. Skipping.")
        return len(new_columns)
    if len(data.headers) == column_offset:
        data.headers += new_columns
#    if len(data.headers) != column_offset + 2:
#        print "Warning, data header size mismatch. Skipping."
#        return

    # opening collision input file
    collisionratios=[]
    inputfile = os.path.join(inputdir, "collision_ratio.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return len(new_columns)
    for line in open (inputfile, 'r'):
        # Average collision ratio (2nd column)
        line = line.strip()
        if not line or line.startswith('t'):
            continue

        actualcollstats = line.split('\t')
        if len(actualcollstats) < 2:
            continue
        collisionratios.append(float(actualcollstats[1]))

    inputfile = os.path.join(inputdir, "collisions.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: collision output file (%s) does not exist!" % inputfile)
        collisions = None
    else:
        collfile = open (inputfile, 'r')
        lines = collfile.readlines()
        lastline = lines[len(lines) - 1].split('\t')
        collisions = float (lastline[1])

        print ("  collisions: %d" % int(collisions))

    # add new data
    if collisionratios:
        data.add_value(plist, column_offset, sum(collisionratios)/len(collisionratios))
    else:
        print ("  Warning: no data in file; skipping dir")
        return len(new_columns)

    if collisions is not None:
        data.add_value(plist, column_offset+1, collisions)
    return len(new_columns)


def process_vel_corr(inputdir, plist, data, settings, column_offset=0):
    new_columns = ["vel_corr"]
    # setting up header line
    if len(data.headers) < column_offset:
        print ("Warning, column_offset is too large. Skipping.")
        return len(new_columns)
    if len(data.headers) == column_offset:
        data.headers += new_columns

    # parse all velocity correlation files
    velcorrs = []
    inputfile = os.path.join(inputdir, "correlation.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return len(new_columns)
    for line in open (inputfile, 'r'):
        # Average correlation (2nd column) is a correct order parameter for 3D SPP algorithm
        line = line.strip()
        if not line or line.startswith('t'):
            continue
        actualcorrstats = line.split('\t')
        if len(actualcorrstats) < 2:
            continue
        velcorrs.append(float(actualcorrstats[1]))

    data.add_value(plist, column_offset, sum(velcorrs)/len(velcorrs))
    return len(new_columns)


def process_cluster_vel_corr(inputdir, plist, data, settings, column_offset=0):
    new_columns = ["cluster_vel_corr"]
    # setting up header line
    if len(data.headers) < column_offset:
        print ("Warning, column_offset is too large. Skipping.")
        return len(new_columns)
    if len(data.headers) == column_offset:
        data.headers += new_columns

    # parse all velocity correlation files
    velcorrs = []
    inputfile = os.path.join(inputdir, "cluster_dependent_correlation.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return len(new_columns)
    for line in open (inputfile, 'r'):
        # Average correlation (2nd column) is a correct order parameter for 3D SPP algorithm
        line = line.strip()
        if not line or line.startswith('t'):
            continue
        actualcorrstats = line.split('\t')
        if len(actualcorrstats) < 2:
            continue
        velcorrs.append(float(actualcorrstats[1]))

    data.add_value(plist, column_offset, sum(velcorrs)/len(velcorrs))
    return len(new_columns)


def process_cluster(inputdir, plist, data, settings, column_offset=0):
    new_columns = ["min_cluster_size", "agents_not_in_cluster"]
    # setting up header line
    if len(data.headers) < column_offset:
        print ("Warning, column_offset is too large. Skipping.")
        return len(new_columns)
    if len(data.headers) == column_offset:
        data.headers += new_columns

    # parse all velocity correlation files
    min_clusters = []
    independent_agents = []
    inputfile = os.path.join(inputdir, "cluster_parameters.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return len(new_columns)
    for line in open (inputfile, 'r'):
        # Average correlation (2nd column) is a correct order parameter for 3D SPP algorithm
        line = line.strip()
        if not line or line.startswith('t'):
            continue
        actualclusterstats = line.split('\t')
        if len(actualclusterstats) < 4:
            continue
        min_clusters.append(float(actualclusterstats[1]))
        independent_agents.append(float(actualclusterstats[3]))

    data.add_value (plist, column_offset, sum(min_clusters)/len(min_clusters))
    data.add_value (plist, column_offset + 1, sum(independent_agents)/len(independent_agents))
    return len(new_columns)


def process_velocity_magn(inputdir, plist, data, settings, column_offset=0):
    new_columns = ["vel_abs_(cm/s)"]
    # setting up header line
    if len(data.headers) < column_offset:
        print ("Warning, column_offset is too large. Skipping.")
        return len(new_columns)
    if len(data.headers) == column_offset:
        data.headers += new_columns

    # parse all velocity files
    vels = []
    inputfile = os.path.join(inputdir, "velocity.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return len(new_columns)
    for line in open (inputfile, 'r'):
        # Average velocity (2nd column) is a correct order parameter for 3D SPP algorithm
        line = line.strip()
        if not line or line.startswith('t'):
            continue
        actualvelstats = line.split('\t')
        if len(actualvelstats) < 2:
            continue
        vels.append(float(actualvelstats[1]))

    data.add_value(plist, column_offset, sum(vels)/len(vels))
    return len(new_columns)


def process_acceleration_magn(inputdir, plist, data, settings, column_offset=0):
    new_columns = ["acc_abs_(cm/s/s)"]
    # setting up header line
    if len(data.headers) < column_offset:
        print ("Warning, column_offset is too large. Skipping.")
        return len(new_columns)
    if len(data.headers) == column_offset:
        data.headers += new_columns

    # parse all acceleration files
    accs = []
    inputfile = os.path.join(inputdir, "acceleration.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return len(new_columns)
    for line in open (inputfile, 'r'):
        # Average acceleration (2nd column) is a correct order parameter for traffic  or other algorithms
        line = line.strip()
        if not line or line.startswith('t'):
            continue
        actualaccstats = line.split('\t')
        if len(actualaccstats) < 2:
            continue
        accs.append(float(actualaccstats[1]))

    data.add_value(plist, column_offset, sum(accs)/len(accs))
    return len(new_columns)


# TODO: Maybe an abstract stat process should be defined... :(
def process_distance_between_neighbours(inputdir, plist, data, settings, column_offset=0):
    new_columns = ["distance_between_neighbours_(cm)"]
    # setting up header line
    if len(data.headers) < column_offset:
        print ("Warning, column_offset is too large. Skipping.")
        return len(new_columns)
    if len(data.headers) == column_offset:
        data.headers += new_columns

    # parse all velocity files
    distances = []
    inputfile = os.path.join(inputdir, "dist_between_neighbours.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return len(new_columns)
    for line in open (inputfile, 'r'):
        # Average correlation (2nd column) is a correct order parameter for 3D SPP algorithm
        line = line.strip()
        if not line or line.startswith('t'):
            continue
        actualdistancestats = line.split('\t')
        if len(actualdistancestats) < 2:
            continue
        distances.append(float(actualdistancestats[1]))

    data.add_value(plist, column_offset, sum(distances)/len(distances))
    return len(new_columns)


################################################################################
####################### MODEL SPECIFIC PROCESSING FUNCTIONS ####################
################################################################################

def process_template(inputdir, plist, data, settings):
    """All model processing functions should look like this:

    def process_mymodel(inputdir, plist, data, settings):

        Arguments:
        ----------

        inputdir - the directory where the robotsim simulation outputs can be found,
                   together with the settings files and the job.sh script
        plist    - the namedtuple list of the values that should be used as the
                   main key for storing the actual data
        data     - the global averaging database
        settings - all settings parsed in one structure. Use like this, e.g.
                   settings.initparams["NumberOfAgents"]

        Usage:
        ------

        Add header strings for all output values as a list like this, e.g.:
        data.headers = ["helo", "belo"]

        Use any model-specific simulation files in inputdir and
        the settings structure to generate output values and
        use the function data.add_value(plist, i, value)
        to add a new value to the i-th output averaging.

        Everything else is fully automatic.

    """
    pass


def process_chasing(inputdir, plist, data, settings):
    """Process chasing model data.

    Output structure:
        * effectiveness_chasers
        * effectiveness_escapers
        * sumrealpath_chaser
        * sumrealpath_escaper
        * lifetime of 1st escaper caught
        * lifetime of 2nd escaper caught
        * ...
        * lifetime of last escaper caught

    Note that if an escaper is not caught, it does not appear in the lifetime list.
    Therefore, number of times when this happens can be calculated by subtracting
    averaging n from the total number of parallel simulations.

    """
    try:
        NumberOfAgents = settings.initparams['NumberOfAgents']
        NumberOfEscapers = settings.flockingparams['NumberOfEscapers']
        VMaxChaser = settings.flockingparams['VMaxChaser']
        VMaxEscaper = settings.flockingparams['VMaxEscaper']
        SimulationLength = settings.initparams['Length']
    except KeyError as e:
        print ("  Warning: error in settings (%s); skipping dir" % e)
        return
    NumberOfChasers = NumberOfAgents - NumberOfEscapers

    # parse SumRealPath
    sumrealpath_chaser = 0
    sumrealpath_escaper = 0
    inputfile = os.path.join(inputdir, "SumRealPathData.txt")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return
    for line in open(inputfile, 'r'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        linesplit = line.split()
        if linesplit[0] == "chaser":
            sumrealpath_chaser += float(linesplit[1])
        elif linesplit[0] == "escaper":
            sumrealpath_escaper += float(linesplit[1])
    avgrealpath_chaser = sumrealpath_chaser/NumberOfChasers
    maxrealpath_chaser = SimulationLength*VMaxChaser

    # parse all lifetimes
    lifetimes = []
    inputfile = os.path.join(inputdir, "EscapersLifeTimeData.txt")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return
    for line in open(inputfile, 'r'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        lifetimes.append(float(line.split()[0]))
    # Escaper effectiveness is ratio of average lifetime in max time allowed.
    # Those who are not caught are taken into account with Effectiveness of 1.
    if len(lifetimes):
        effectiveness_escapers = ( sum(lifetimes)/SimulationLength +
                NumberOfEscapers - len(lifetimes) ) / NumberOfEscapers
    else:
        effectiveness_escapers = 1

    # define output file header
    if not data.headers:
        data.headers += ["effectiveness_chasers", "effectiveness_escapers", "sumrealpath_chaser", "sumrealpath_escaper"]
    if len(lifetimes) > len(data.headers) - 4:
        data.headers += ["lifetime_%d" % (i+1) for i in xrange(len(data.headers) - 4, len(lifetimes))]

    # add new data (initialization, averaging and indexing is automatic)
    # 0: effectiveness_chasers
    data.add_value(plist, 0, 1/(effectiveness_escapers * (avgrealpath_chaser/maxrealpath_chaser) * NumberOfChasers))
    # 1: effectiveness_escaper
    data.add_value(plist, 1, effectiveness_escapers)
    # 2-3: SumRealPath
    data.add_value(plist, 2, sumrealpath_chaser)
    data.add_value(plist, 3, sumrealpath_escaper)
    # 4- : lifetimes
    for i,lifetime in enumerate(lifetimes):
        data.add_value(plist, i+4, lifetime)


def process_spp_3D(inputdir, plist, data, settings):
    """Process SPP_3D model data.

    Output structure:
        * vel_corr

    """

    # add common outputs
    column_offset = 0
    column_offset += process_vel_corr(inputdir, plist, data, settings, column_offset)


def process_spp_evol(inputdir, plist, data, settings):
    """ Process evolution-featured spp data.

    Output structure:
        * distance_from_arena_avg , vel_corr , vel_abs , ratio_of_collisions , collisions

    """

    """ Reading out some necessary flocking parameters """
    try:
        R_0 = settings.flockingparams['R_0']
        V_Flock = settings.flockingparams['V_Flock']
        Gamma_Rep = 0 #settings.flockingparams['Gamma_Rep']
    except (KeyError, ValueError) as e:
        print ("  Warning: error in settings (%s); skipping dir" % e)
        return

    # setting up header line
    data.headers = ["distance_from_arena_(cm)", "R_0_(cm)", "v_flock_(cm/s)", "gamma_rep_(cm)"]

    # parse all velocity correlation files
    distances = []

    inputfile = os.path.join(inputdir, "distance_from_arena.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return
    for line in open (inputfile, 'r'):
        line = line.strip()
        if not line or line.startswith('time') or line.startswith('#'):
            continue
        actualdiststats = line.split('\t')
        if len(actualdiststats) < 3:
            continue
        distances.append(float(actualdiststats[1]))

    data.add_value(plist, 0, sum(distances)/len(distances))
    data.add_value(plist, 1, R_0)
    data.add_value(plist, 2, V_Flock)
    data.add_value(plist, 3, Gamma_Rep)

    # add common outputs
    column_offset = 4
    column_offset += process_cluster(inputdir, plist, data, settings, column_offset)
    column_offset += process_cluster_vel_corr(inputdir, plist, data, settings, column_offset)
    column_offset += process_velocity_magn(inputdir, plist, data, settings, column_offset)
    column_offset += process_collisions(inputdir, plist, data, settings, column_offset)
    column_offset += process_distance_between_neighbours(inputdir, plist, data, settings, column_offset)


def process_traffic(inputdir, plist, data, settings):
    """Process traffic model data.
    Can be useful for analyzing traffic circle and slowing down models

    Output structure:
        * SumRealPath
        * SumEffectivePath
        * SumTheoreticalMaxPath
        * SumTargets
        * ...
        * // EffectiveFlux = v*rho =
          //     SumEffectivePath/TotalTime * NumberOfAgents/ArenaSize
        * flocking velocity
        * effective velocity
        * collisions
        * max theoretical acceleration
        * average acceleration
        * number of targets reached

    Note that all simulations output these values for all agents, so
    in the final averaging n = NumberOfAgents*NumberOfSimulations

    """
    # parse settings parameters for flux/effective velocity calculation
    try:
        VFlock = settings.flockingparams['V_Flock']
        Amax = settings.unitparams['a_max']
        """
        NumberOfAgents = settings.initparams['NumberOfAgents']
        SimulationLength = settings.initparams['Length']
        ArenaSize = settings.flockingparams['ArenaSize']
        if ArenaSize <= 0:
            raise ValueError("ArenaSize is %f" % ArenaSize)
        ArenaShape = settings.flockingparams['ArenaShape']
        # circle
        if ArenaShape == 0:
            ArenaArea = ArenaSize * ArenaSize / 4 * math.pi
        # square
        elif ArenaShape == 1:
            ArenaArea = ArenaSize * ArenaSize / 4
        # other, not handled
        else:
            raise ValueError("ArenaShape unknown")
        rho = NumberOfAgents / ArenaArea
        """
    except (KeyError, ValueError) as e:
        print ("  Warning: error in settings (%s); skipping dir" % e)
        return
    # parse userdata.txt
    """
    inputfile = os.path.join(inputdir, "userdata.txt")
    if not os.path.isfile(inputfile):
        print "  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile
        return
    for line in open (inputfile, 'r'):
        # check new line
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        linesplit = line.split('\t')
        # add header
        if linesplit[0] == "ID":
            data.headers = linesplit[1:] + ["EffectiveFlux"]
            continue
        # add new data line by line
        for i in xrange(1,len(linesplit)):
            data.add_value(plist, i-1, float(linesplit[i]))
        # add flux to the end, line by line
        data.add_value(plist, i, float(linesplit[data.headers.index("SumEffectivePath")]) / SimulationLength * rho)
    """

    data.headers = ["effective_vel_avg_(cm/s)", "effective_vel_std_(cm/s)", "time_between_targets_avg_(s/target)", "v_flock_(cm/s)", "a_max_(cm/s/s)"]

    # parse all effective velocity files
    effectivevels = []
    effectivevelsstd = []
    time_between_targets = []
    inputfile = os.path.join(inputdir, "traffic_stat.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return
    for line in open (inputfile, 'r'):
        line = line.strip()
        if not line or line.startswith('time') or line.startswith('#'):
            continue
        actualstats = line.split('\t')
        if len(actualstats) < 3:
            continue
        effectivevels.append(float(actualstats[1]))
        effectivevelsstd.append(float(actualstats[2]))
        time_between_targets.append(float(actualstats[0]) / float(actualstats[9]))

    if not effectivevels or not effectivevelsstd or not time_between_targets:
        print ("  Warning: simulation output file (%s) empty; skipping dir" % inputfile)
        return

    data.add_value(plist, 0, sum(effectivevels)/len(effectivevels))
    data.add_value(plist, 1, sum(effectivevelsstd)/len(effectivevelsstd))
    data.add_value(plist, 2, sum(time_between_targets)/len(time_between_targets))
    data.add_value(plist, 3, VFlock)
    data.add_value(plist, 4, Amax)

    # add common outputs
    column_offset = 5
    column_offset += process_acceleration_magn(inputdir, plist, data, settings, column_offset)
    column_offset += process_collisions(inputdir, plist, data, settings, column_offset)

# hack: aliases for process_traffic
process_traffic_slowdown = process_traffic
process_traffic_circle = process_traffic
process_spp_int_hier = process_spp_evol


def process_random_targets(inputdir, plist, data, settings):
    """Process "random_targets" model data.
    Can be useful for analyzing random_targets / waypointcloud models

    Output structure:
        * relative_time_of_mission_completion
        * dist_from_target
        * static_dist_from_target
        * collisions

    Note that all simulations output some of these values for all agents, so
    in the final averaging n = NumberOfAgents*NumberOfSimulations

    """
    # parse settings parameters for output calculation
    try:
        SimulationLength = settings.initparams['Length']
        TargetAutoUpdateTime = settings.flockingparams['TargetAutoUpdateTime']
    except (KeyError, ValueError) as e:
        print ("  Warning: error in settings (%s); skipping dir" % e)
        return

    data.headers = [
            "time_needed_for_mission_completion",
            "relative_time_of_mission_completion",
            "dist_from_target",
            "static_dist_from_target"]

    # parse all random_targets stat files
    # header: "time(s)\tdist_from_target_avg(cm/s)\tn_dist\tstatic_dist_from_target_avg(cm/s)\tn_static_dist\n"
    inputfile = os.path.join(inputdir, "random_targets_stat.dat")
    if not os.path.isfile(inputfile):
        print ("  Warning: simulation output file (%s) does not exist; skipping dir" % inputfile)
        return
    for line in open (inputfile, 'r'):
        line = line.strip()
        if not line or line.startswith('time') or line.startswith('#'):
            continue
        actualstats = line.split('\t')
        if len(actualstats) != 6:
            continue
        # we need first good line (assuming stat and not timeline output mode)
        time_of_mission_completion = float(actualstats[0])
        target_set_index, target_count = [int(x) for x in actualstats[1].split("/")]
        dist_from_target = float(actualstats[2])
        dist_from_target_num = int(actualstats[3])
        static_dist_from_target = float(actualstats[4])
        static_dist_from_target_num = int(actualstats[5])
        break
    else:
        print ("  Warning: simulation output file (%s) does not contain proper data; skipping dir" % inputfile)
        return

    # 0: time_needed_for_mission_completion
    data.add_value(plist, 0, (TargetAutoUpdateTime * target_count)/SimulationLength)
    # 1: relative_time_of_mission_completion
    data.add_value(plist, 1, time_of_mission_completion/SimulationLength)
    # 2: dist_from_target
    if dist_from_target_num > 0:
        #for i in range(dist_from_target_num):
        data.add_value(plist, 2, dist_from_target)
    # 3: static_dist_from_target
    if static_dist_from_target_num > 0:
        #for i in range(static_dist_from_target_num):
        data.add_value(plist, 3, static_dist_from_target)
    # 4: add common outputs
    column_offset = 4
    column_offset += process_collisions(inputdir, plist, data, settings, column_offset)


################################################################################
################################### MAIN CODE ##################################
################################################################################


def main(argv = []):
    # parse arguments
    if len(argv) not in [2,3]:
        print (__doc__)
        print (process_template.__doc__)
        return None
    model = argv[1]
    if len(argv) == 3:
        label = argv[2]
    else:
        label = ""
    inputdirs = glob.glob(os.path.join(argv[0], "*%s" % label))
    print (len(inputdirs), "input dirs found")
    if not inputdirs:
        print ("exiting")
        return None

    # parse data
    evolution = -1
    data = data_t()
    for inputdir in inputdirs:
        if not os.path.isdir(inputdir):
            continue

        # parse directory name and do basic check
        print ("parsing", inputdir)
        params = os.path.split(inputdir)[1].split('__')
	if params[0] == 'robotsim':
            if evolution == 1:
                print ("  Warning: conflicting dirs (evolution ON is used now); skipping data")
                continue
            evolution = 0
        elif params[0] == 'robotsim_evolution':
            if evolution == 0:
                print ("  Warning: conflicting dirs (evolution OFF is used now); skipping data")
                continue
            evolution = 1
        else:
            print ("  Warning: inputdir should start with 'robotsim' or 'robotsim_evolution'; skipping data")
            continue

        # parse param names and values from directory name, skipping "robotsim"
        pvlist = []
        is_there_n = False
        for param_and_value in params[1:]:
            p = param_t(*param_and_value.rsplit('_', 1))
            if p.param == "n":
                is_there_n = True
                break
            pvlist.append(p)
        if not is_there_n:
            print ("  Warning: inputdir shoud end with 'n_*'; skipping data")
            continue

        # initialize data's param namedtuple
        if not data.paramlist_t:
            data.define_paramlist_t(pvlist)

        # basic error check
        if not data.is_compatible(pvlist):
            print ("  Warning: incompatible param list; skipping data")
            continue

        # parse all local settings/params
        settings = atlasz.util.parse_paramfiles_from_jobfile(os.path.join(inputdir, 'job.sh'))
        if settings is None or not settings.initparams or not settings.unitparams or not settings.flockingparams:
            print ("  Warning: could not parse ini files; skipping dir")
            continue

        # convert plist into simpler form to use in data dictionaries
        plist = data.paramlist_t(*[p.value for p in pvlist])

        # collect and average data in a model-specific way
        try:
            eval("process_%s" % model)(inputdir, plist, data, settings)
        except NameError as e:
            if str(e) == "name 'process_%s' is not defined" % model:
                print ("Model type '%s' is not implemented yet (%s)." % (model, e))
                print (process_template.__doc__)
                return None
            raise

    # write header
    if label:
        x = "_%s.dat" % label
    else:
        x = ".dat"
    outputfilename = os.path.join(argv[0], os.path.splitext(os.path.split(__file__)[1])[0] + "_" + argv[1] + x)
    f = open(outputfilename, 'w')
    x = list(data.paramlist_t._fields)
    for h in data.headers:
        x += [h + "_avg", h + "_std", h + "_n"]
    x += "\n"
    f.write("\t".join(map(str, x)))

    # write data
    for plist in sorted(data.n, key=lambda x:[float(i) for i in x]):
        x = list(plist)
        for i in xrange(len(data.avg[plist])):
            if data.n[plist][i]:
                x += [data.avg[plist][i], math.sqrt(data.stv[plist][i])/data.n[plist][i], data.n[plist][i]]
            else:
                x += [float('nan'), float('nan'), 0]
        for ii in xrange(len(data.headers) - len(data.avg[plist])):
            x += [float('nan'), float('nan'), 0]
        x += "\n"
        f.write("\t".join(map(str, x)))
    f.close()

    # return name of the generated postprocess file
    return outputfilename


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as ex:
        print (ex, file=sys.stderr)
        import traceback
        traceback.print_exc(ex)
        sys.exit(1)
