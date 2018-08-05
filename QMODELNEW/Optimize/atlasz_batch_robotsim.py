"""
Run flyingrobots/simulations/robotsim/robotflocksim_main scripts on atlasz
as sbatch scripts, sweeping through parameter ranges.

"""

# external imports
from __future__ import print_function
import os
import sys
import argparse
import datetime
import subprocess
import collections
import shutil
import getpass
import itertools

#internal imports
import atlasz.util

# define user variables here
USER = getpass.getuser() # automatic, change only if you need something explicit
CODE_DIR = '/project/flockwork/%s/code/robotsim' % USER
WORK_DIR = '/project/flockwork/%s/robotsim_workdir/' % USER
RESULT_DIR = '/project/flockwork/%s/robotsim_resultdir/' % USER
DATE = str(datetime.datetime.now().date())

def make_job(options, params, pvalues, n):
    """Returns .sh script, jobid and workdir."""
    # create job id
    jobid = "__".join(['robotsim'] + ["%s_%g" % (params[i].name, pvalues[i]) for i in xrange(len(pvalues))] + \
            ["n_%d" % n])
    if options.label:
        jobid += "__%s" % options.label
    # check result dir
    resultdir = os.path.join(RESULT_DIR, DATE, jobid)
    if os.path.isdir(resultdir):
        print ('Result directory already exists, I will delete it\n', resultdir)
        subprocess.call (["rm", "-r", resultdir])
    # check workdir
    workdir = os.path.join(WORK_DIR, DATE, jobid)
    if os.path.isdir(workdir):
        print ('Working directory already exists, I will delete it\n', workdir)
        subprocess.call (["rm", "-r", workdir])
    # create dirs
    os.makedirs(workdir)
    os.makedirs(resultdir)
    # copy all default ini files to working dir
    shutil.copyfile(options.initparams, os.path.join(workdir, os.path.split(options.initparams)[1]))
    shutil.copyfile(options.unitparams, os.path.join(workdir, os.path.split(options.unitparams)[1]))
    shutil.copyfile(options.flockingparams, os.path.join(workdir, os.path.split(options.flockingparams)[1]))
    shutil.copyfile(options.obstacles, os.path.join(workdir, os.path.split(options.obstacles)[1]))
    shutil.copyfile(options.arenas, os.path.join(workdir, os.path.split(options.arenas)[1]))
    shutil.copyfile(options.waypoints, os.path.join(workdir, os.path.split(options.waypoints)[1]))
    shutil.copyfile(options.outputconf, os.path.join(workdir, os.path.split(options.outputconf)[1]))
    # change params in ini files according to current iterated values
    if not atlasz.util.replace_params_in_paramfiles(workdir, params, pvalues):
        return (None, None, None)
    # create .sh
    template_variables = {
        'user': USER,
        'codedir': CODE_DIR,
        'workdir': workdir,
        'resultdir': resultdir,
        'initparams': os.path.join(workdir, os.path.split(options.initparams)[1]),
        'unitparams': os.path.join(workdir, os.path.split(options.unitparams)[1]),
        'flockingparams': os.path.join(workdir, os.path.split(options.flockingparams)[1]),
        'obstparams': os.path.join(workdir, os.path.split(options.obstacles)[1]),
        'arenaparams': os.path.join(workdir, os.path.split(options.arenas)[1]),
        'wpparams': os.path.join(workdir, os.path.split(options.waypoints)[1]),
        'outputconf': os.path.join(workdir, os.path.split(options.outputconf)[1]),
    }
    return (atlasz.util.ROBOTFLOCKSIM_JOB_TEMPLATE % template_variables, jobid, workdir)


def make_jobs_recursively(options, params, pvalues, n, joblist, it=0):
    """Since we do not know how many params are defined, we need recursive
    iterations..."""
    # all iterations are set, finally lets do the job
    if it == len(params):
        # flatten param list here
        (jobscript, jobid, workdir) = make_job(options, list(itertools.chain(*params)), list(itertools.chain(*pvalues)), n)
        if jobscript is not None:
            joblist.append((jobscript, jobid, workdir))
        return
    # iterate, we have not iterated enough
    for plist in zip(*[atlasz.util.frange(params[it][i].minv, params[it][i].maxv, params[it][i].step) for i in xrange(len(params[it]))]):
        pvalues[it] = plist
        make_jobs_recursively(options, params, pvalues, n, joblist, it+1)


def run_jobs_in_clusters(joblist, j):
    """Run jobs with j-sized simulation clusters."""
    # this is the standard 1-simulation per job mode
    if j <= 1:
        for (jobscript, jobid, workdir) in joblist:
            atlasz.util.run_job(jobscript, jobid, workdir)
    # this is the clustered j-simulations per job mode
    else:
        # first create individual jobfiles for postprocessing
        for (jobscript, jobid, workdir) in joblist:
            atlasz.util.run_job(jobscript, jobid, workdir, only_create_jobfile=True)
        # but run the jobs only in clusters
        for i in xrange(0, len(joblist), j):
            cluster = joblist[i:i+j]
            metajobscript = "\n".join(["#!/bin/sh"] + [x[0].replace("#!/bin/sh", "") for x in cluster])
            metajobid = cluster[0][1]
            metaworkdir = cluster[0][2] + "__clusteredjob_%d" % i
            # check workdir
            if os.path.isdir(metaworkdir):
                print ('Working directory already exists, I will delete it\n', metaworkdir)
                subprocess.call (["rm", "-r", metaworkdir])
            os.makedirs(metaworkdir)
            # run clustered job
            atlasz.util.run_job(metajobscript, metajobid, metaworkdir)


def main(argv=[]):
    """Main entry point."""
    print ("This is", __file__, "SVN revision:", atlasz.util.get_svn_info(__file__)['revision'])
    # parse command line arguments
    argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    argparser.add_argument("-j", "--jobsize", dest="j", type=int, default=1, help="Define number of simulations for each atlasz job")
    argparser.add_argument("-n", "--number", dest="n", type=int, default=1, help="Define number of simulations to run with one parameter setup")
    argparser.add_argument("-p", "--params", dest="p", nargs='+', default = [], help="Define any number of parameters to iterate, each with the following subparams: f|u|i param minv maxv step [AND|NEXT ...]. If you have the 'AND' operator specified between two params, their value will change together (no matrix, only diagonal). In this case be careful to prepare equal number of iterations. If you have 'NEXT' between params, they will be independently iterated.")
    argparser.add_argument("-i", "--initparams", metavar="file", dest="initparams", default=os.path.join(CODE_DIR, "parameters", "initparams.dat"), help="Define init params file")
    argparser.add_argument("-u", "--unitparams", metavar="file", dest="unitparams", default=os.path.join(CODE_DIR, "parameters", "unitparams.dat"), help="Define unit params file")
    argparser.add_argument("-f", "--flockingparams", metavar="file", dest="flockingparams", default=os.path.join(CODE_DIR, "parameters", "flockingparams.dat"), help="Define flocking params file")
    argparser.add_argument("-obs", "--obstacles", metavar="file", dest="obstacles", default=os.path.join(CODE_DIR, "parameters", "obstacles.default"), help="Define obstacle file")
    argparser.add_argument("-arena", "--arenas", metavar="file", dest="arenas", default=os.path.join(CODE_DIR, "parameters", "arenas.default"), help="Define arena file")
    argparser.add_argument("-wp", "--waypoints", metavar="file", dest="waypoints", default=os.path.join(CODE_DIR, "parameters", "waypoints.default"), help="Define waypoint file")
    argparser.add_argument("-outputconf", "--outputconf", metavar="file", dest="outputconf", default=os.path.join(CODE_DIR, "config", "output_config.ini"), help="Define output config file")
    argparser.add_argument("-l", "--label", default="", dest = "label", help="Specific keyword related to the set of input parameters.")
    # if arguments are passed to main(argv), parse them
    if argv:
        options = argparser.parse_args(argv)
    # else if called from command line or no arguments are passed to main, parse default argument list
    else:
        options = argparser.parse_args()
    # check arguments
    if (len(options.p) % 6) != 5:
        print ("ERROR: -p should be followed by 6x-1 number of parameters (f|u|i param minv maxv step [AND|NEXT ...])!")
        return
    params = []
    for i in xrange(len(options.p)/6 + 1):
        if options.p[i*6+0] == 'f':
            f = os.path.split(options.flockingparams)[1]
        elif options.p[i*6+0] == 'u':
            f = os.path.split(options.unitparams)[1]
        elif options.p[i*6+0] == 'i':
            f = os.path.split(options.initparams)[1]
        else:
            print ("ERROR: option -p 6*i+0 param should be one of ['f', 'i', 'u']!")
            return
        for j in xrange(2,5):
            options.p[i*6+j] = float(options.p[i*6+j])
        p = atlasz.util.param_t(f,         # file
                options.p[i*6+1],          # name
                float(options.p[i*6+2]),   # minv
                float(options.p[i*6+3]),   # maxv
                float(options.p[i*6+4]))   # step
        if i > 0 and options.p[i*6-1] == "AND":
            params[-1].append(p)
        elif i == 0 or options.p[i*6-1] == "NEXT":
            params.append([p])
        else:
            print ("ERROR: option -p 6*i+5 param should be one of ['AND', 'NEXT']!")
            return
    # create batch jobs for all parameter values separately
    joblist = []
    for n in xrange(options.n):
        pvalues = [[]]*len(params)
        make_jobs_recursively(options, params, pvalues, n, joblist)
    # submit them to queue
    run_jobs_in_clusters(joblist, options.j)

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:])) # pass only real params to main
    except Exception as ex:
        print (ex, file=sys.stderr)
        import traceback
        traceback.print_exc(ex)
        sys.exit(1)
