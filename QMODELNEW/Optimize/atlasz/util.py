"""Common utilities for atlasz batch robotsim python scripts."""


from __future__ import print_function
import os
import sys
import subprocess
import shutil
import collections
import getpass
if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle
import gzip
import gc
import dill

# define user variables here
USER = getpass.getuser() # automatic, change only if you need something explicit
SBATCH_DEFAULT_PARAMS = ' -p flockingall --mem-per-cpu=2048 ' # (2 Gb memory usage per cpu)

# these are the parameters needed to iterate one model parameter
param_t = collections.namedtuple('param_t', 'paramfile name minv maxv step')

# these are the settings files used to store different robotsim params
settings_t = collections.namedtuple("settings_t", ["flockingparams", "unitparams", "initparams"])


# the main template to call robotflocksim_main on atlasz
ROBOTFLOCKSIM_JOB_TEMPLATE = """#!/bin/sh
    %(codedir)s/robotflocksim_main -novis -outputconf %(outputconf)s -o %(workdir)s -i %(initparams)s -u %(unitparams)s -f %(flockingparams)s -obs %(obstparams)s -arena %(arenaparams)s -wp %(wpparams)s
    mv %(workdir)s/* %(resultdir)s
"""

# the main template to call robotflocksim_main on atlasz
ROBOTFLOCKSIM_JOB_TEMPLATE_MASTER = """#!/bin/sh
    python %(codedir)s/atlasz_evolve_robotsim.py %(argv)s
    mv %(workdir)s/* %(resultdir)s
"""

def convert_string_to_param_t(string):
    """Convert a string (like eparams.params_as_environment members)
    to param_t type variable."""
    env = string.split()
    return param_t(env[0], # file
            env[1],          # name
            float(env[2]),   # minv
            float(env[3]),   # maxv
            float(env[4]))   # step


def parse_paramfile_into_dict(inputfile):
    """Parse an ini file (flocingparams, unitparams, initparams, etc.)"
    into a dict and return it."""
    retval = {}
    for line in open(inputfile, 'r'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        linesplit = line.split('=')
        if len(linesplit) != 2:
            print ("  Warning: input file format error in", inputfile, "line:", line)
        retval[linesplit[0]] = float(linesplit[1])
    return retval


def parse_paramfiles_from_jobfile(jobfile):
    """Parse a job.sh file to get f,u,i files and parse those, too."""
    if not os.path.isfile(jobfile):
        print ("  Warning: jobfile does not exist.")
        return None

    fparams = None
    uparams = None
    iparams = None
    for line in open(jobfile, 'r'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        linesplit = line.split()
        try:
            i = linesplit.index('-f')
            fparams = linesplit[i+1]
            i = linesplit.index('-u')
            uparams = linesplit[i+1]
            i = linesplit.index('-i')
            iparams = linesplit[i+1]
        except ValueError:
            pass

    if None in [fparams, uparams, iparams]:
        print ("  Warning: could not fully parse jobfile.")
        return None
    path = os.path.split(jobfile)[0]
    fparams = os.path.join(path, os.path.split(fparams)[1])
    uparams = os.path.join(path, os.path.split(uparams)[1])
    iparams = os.path.join(path, os.path.split(iparams)[1])
    if os.path.isfile(fparams) and os.path.isfile(uparams) and os.path.isfile(iparams):
        return settings_t(
            parse_paramfile_into_dict(fparams),
            parse_paramfile_into_dict(uparams),
            parse_paramfile_into_dict(iparams))

    print ("  Warning: settings files missing.""")
    return None


def replace_params_in_paramfiles(workdir, params, pvalues):
    """Replace a list of params in paramfiles with a new value.

    :param workdir: the working directory
    :param params:  the list of param_t params (must contain local .paramfile names)
    :param pvalues: the corresponding list of new param values

    :return true on success

    """
    if len(pvalues) != len(params):
        print ("ERROR: size of pvalues(%d) and params(%d) must match!" % (len(pvalues), len(params)))
        return False
    # get all paramfiles
    paramfiles = set()
    for i in xrange(len(pvalues)):
        paramfiles.add(os.path.join(workdir, params[i].paramfile))
    # replace params in paramfiles
    found = [0]*len(pvalues)
    for paramfile in paramfiles:
        tempfile = os.path.join(workdir, "temp.dat")
        f = open(tempfile, 'w')
        for line in open(paramfile, 'r'):
            for i in xrange(len(pvalues)):
                if line.startswith(params[i].name + '='):
                    f.write("%s=%g\n" % (params[i].name, pvalues[i]))
                    found[i] += 1
                    break
            else:
                f.write(line)
        f.close()
        shutil.move(tempfile, paramfile)
    # check results
    for i in xrange(len(found)):
        if found[i] != 1:
            print ("ERROR: param", params[i].name, "in", params[i].paramfile, "found", found[i], "times!")
            return False
    return True


def run_job(jobscript, jobid, workdir, unique_flags = '', only_create_jobfile=False):
    if only_create_jobfile:
        print ("Writing", jobid, "jobfile...")
    else:
        print ("Running", jobid, "job...")
    sys.stdout.flush()
    job_filename = os.path.join(workdir, 'job.sh')
    print (jobscript, file=open(job_filename, 'w'))
    if only_create_jobfile:
        return
    if (unique_flags == ''):
        sbatch_command = 'sbatch %(default_params)s --job-name=%(jobid)s --error=%(workdir)s/stderr --output=%(workdir)s/stdout %(job_filename)s' % {
            'default_params': SBATCH_DEFAULT_PARAMS,
            'job_filename': job_filename,
            'jobid': jobid,
            'workdir': workdir}
    else:
        sbatch_command = 'sbatch %(default_params)s %(unique_flags)s --job-name=%(jobid)s --error=%(workdir)s/stderr --output=%(workdir)s/stdout %(job_filename)s' % {
            'default_params': SBATCH_DEFAULT_PARAMS,
            'unique_flags': unique_flags,
            'job_filename': job_filename,
            'jobid': jobid,
            'workdir': workdir}
    os.system(sbatch_command)
    sys.stdout.flush()


def get_svn_info(filename=__file__):
    """Get svn revision string for a specific file."""
    realpath = os.path.split(os.path.realpath(__file__))[0]
    info = subprocess.Popen(['svn', 'info', realpath],
            stdout = subprocess.PIPE).communicate()[0].split('\n')
    infodict = {}
    for line in info:
        x = line.split(':',1)
        if len(x) == 2:
            infodict[x[0].lower()] = x[1].strip()
    return infodict


def frange(minv, maxv, step):
    """Return range between min and max, including max, in float steps."""
    assert step > 0.0
    total = minv
    compo = 0.0
    while total <= maxv:
        yield total
        y = step - compo
        temp = total + y
        compo = (temp - total) - y
        total = temp


def queued_jobs(endlabel="", list_all_users=False):
    """Parse output of squeue -- also includes running, and might include
    finished jobs.

    If param 'list_all_users' is True, output will be a list of all queued jobs
    from all users.

    If param 'endlabed' is defined, jobnames ending with it will be listed only

    """
    jobs = []
    # see http://docs.python.org/release/2.6/library/subprocess.html#replacing-bin-sh-shell-backquote
    paramlist = ["squeue", "--noheader", "-o", "%t %j %u"]
    if not list_all_users:
        paramlist += ["-u", USER]
    for line in subprocess.Popen(paramlist, stdout=subprocess.PIPE).communicate()[0].split('\n'):
        line = line.strip()
        if not line: continue
        state, jobname, user = line.split()
        if endlabel and not jobname.endswith(endlabel): continue
        jobs.append((state, jobname, user))
    return jobs


def load_object(filename):
    """Load a compressed object from disk.

    :param filename: name of the .zip file to load from

    """
    gc.disable()
    if filename.endswith('.zip'):
        f = gzip.GzipFile(filename, 'rb')
    else:
        f = open(filename, 'rb')
    try:
#        object = cPickle.load(f)
        object = dill.load(f)
    except EOFError:
        print ("ERROR loading", filename)
        object = None
    f.close()
    gc.enable()
    return object


def save_object(object, filename, protocol = cPickle.HIGHEST_PROTOCOL):
    """Save a (compressed) object to disk.

    :param filename: name of the .zip file to save to

    """
    gc.disable()
    if filename.endswith('.zip'):
        f = gzip.GzipFile(filename, 'wb')
    else:
        f = open(filename, 'wb')
#    cPickle.dump(object, f, protocol)
    dill.dump(object, f)
    f.close()
    gc.enable()
