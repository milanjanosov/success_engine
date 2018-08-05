""" This file contains useful mathematical tools for calculating 
    fitnesses """


# external imports
import math


# Gaussian function
# Can be used for approximating Dirac-delta around a specific point
def gauss(x, sigmasquare, mean, height):
    return height * (math.exp ( - (math.pow (x - mean, 2.0)) / sigmasquare))

# Sigmoid curve
# ______
#       \
#        \______
#         
# Can be used for constraining a quantity below a maximum value
# Below 'mean' with decreasing 'x', this function grows smoothly 
# to 'height' with a finite decay characterized by 'tolerance'
def sigmoid(x, tolerance, maximum, height):
    if maximum < x:
        return 0.0
    elif x > (maximum - tolerance): 
        return height * 0.5 * (math.sin ((math.pi * (1.0 / tolerance)) * (x - maximum) - math.pi * 0.5) + 1.0)
    else:
        return height

# This is a zero-centered monotonically decreasing function
# The derivative of the curve is inifinity at x = 0
# Useful for collision-related fitness calculations
def peak (x, a):
    return (a*a / math.pow(x + a, 2.0))

