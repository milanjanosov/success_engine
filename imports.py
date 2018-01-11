import os
import sys
import matplotlib
#matplotlib.use('Agg')
import seaborn
import numpy as np
import random
import matplotlib.pyplot as plt
import CareerAnalysisHelpers.binningFunctions as binning
import CareerAnalysisHelpers.fittingImpactDistributions as fit
from scipy import stats
from matplotlib.colors import LogNorm
from CareerAnalysisHelpers.alignPlots import align_plot
from CareerAnalysisHelpers.binningFunctions import getDistribution
from CareerAnalysisHelpers.writeRow import write_row
