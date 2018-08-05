"""Merge several atlasz_batch_postprocess.py output files into one.

Usage: __file__ files_to_merge output_file

"""

from __future__ import print_function
import os, sys, glob, math


def parse_batch_postprocess_file(filename):
    """Parse an atlasz_batch_postprocess file into a data array."""
    header = []
    data = []
    for line in open(filename, 'r'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        linesplit = line.split('\t')
        if not header:
            header = linesplit
            continue
        if len(linesplit) != len(header):
            print ("Error in data file '%s' line '%s': len(header)=%d, len(line)=%d" % \
                    (filename, str(linesplit), len(header), len(linesplit)))
            return (None, None)
        data.append([float(p) for p in linesplit])
    return (header, data)


def get_params_from_header(header):
    """Return the params part of the header."""
    params = []
    for h in header:
        if h.endswith("_avg"):
            break
        params.append(h)
    return params


def find_corresponding_row_index(data, newrow, n):
    """Find row of data that is the same as newrow in the first n places."""
    for i,row in enumerate(data):
        if row[:n] == newrow[:n]:
            return i
    return -1


def merge_rows(a, b, n):
    """Add num, average and standard deviation of independent populations.
    Source: https://en.wikipedia.org/wiki/Standard_deviation#Combining_standard_deviations
    """
    row = a[:n]
    for i in range(n, len(a), 3):
#        print a[i:i+3]
        avga, stda, na = a[i:i+3]
#        print b[i:i+3]
        avgb, stdb, nb = b[i:i+3]
        nab = na + nb
        # if there is data, we merge them
        if nab:
            # store added weighted averages
            row.append((avga*na + avgb*nb)/nab)
            # store added weighted standard deviations
            row.append(math.sqrt((na*stda*stda + nb*stdb*stdb)/nab + \
                    na*nb*(avga-avgb)*(avga-avgb)/(nab*nab)))
        # if a and b are both empty, it does not matter what we store, so we store a
        else:
            row.append(avga)
            row.append(stda)
        # store added n
        row.append(nab)
    return row


################################################################################
################################### MAIN CODE ##################################
################################################################################

def main(argv = []):
    # check arguments
    if len(argv) < 2:
        print (__doc__)
        return
    # parse all but last arguments as input file
    inputfiles = []
    for x in argv[:-1]:
        if sys.platform.startswith('win'):
            inputfiles += glob.glob(x)
        else:
            inputfiles += [x]
    # parse last argument as output file
    outputfile = argv[-1]
    if os.path.isfile(outputfile):
        print ("Output file '%s' exists, dunno what to do, exiting." % outputfile)
        return

    # parse first file
    print (inputfiles[0])	
    header, data = parse_batch_postprocess_file(inputfiles[0])
    if header is None:
        return
    nparams = len(get_params_from_header(header))

    # parse all other files and merge data
    for inputfile in inputfiles[1:]:
        print (inputfile)
        newheader, newdata = parse_batch_postprocess_file(inputfile)
        if newheader is None:
            return
        if header != newheader:
            print ("Headers of '%s' and '%' are not the same, exiting." % (inputfiles[0], inputfile))
            return
        for newrow in newdata:
            i = find_corresponding_row_index(data, newrow, nparams)
            # if it is a new line, we add it
            if i < 0:
                data.append(newrow)
                continue
            # if it is an old line, we average them
            data[i] = merge_rows(data[i], newrow, nparams)

    # write output
    f = open(outputfile, 'w')
    f.write("\t".join(header))
    f.write("\n")
    for x in data:
        f.write("\t".join(["%g" % xx for xx in x]))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as ex:
        print (ex, file=sys.stderr)
        import traceback
        traceback.print_exc(ex)
        sys.exit(1)
