import os
import sys
import matplotlib
#matplotlib.use('Agg')
#import seaborn
import numpy as np
import random
import math
import matplotlib.pyplot as plt

from scipy import stats
from matplotlib.colors import LogNorm
from multiprocessing import Process


sys.path.insert(0, '../')

import CareerAnalysisHelpers.binningFunctions as binning
import CareerAnalysisHelpers.fittingImpactDistributions as fit
from CareerAnalysisHelpers.alignPlots import align_plot
from CareerAnalysisHelpers.binningFunctions import getDistribution
from CareerAnalysisHelpers.writeRow import write_row
