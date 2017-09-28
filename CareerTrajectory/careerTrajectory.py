import os
import gzip
import random
import numpy as np



''' A SimpleCareerTrajectory has two fields: the id of the individual, and her career trajectory as a list of time events. '''
''' Each time event is a triple of a time stamp a rational number, e.g. success measure, and a unique id for that time event. ''' 



class SimpleCareerTrajectory:

    
    def __init__(self, name, inputfile, impactm):
        self.impactm = impactm
        self.name    = inputfile
        
        events = []       
        for line in gzip.open(inputfile):
            if 'year' not in line:
                fields  = line.strip().split('\t')
                product = fields[0]
                try:    
                    year    = float(fields[1])
                    impact  = float(fields[impactm + 2])
                    if impact > 0 and year > 1500 and year < 2018:
                        events.append((product, year, impact))
                except ValueError:
                    pass
        
        self.events = events        
        
        
        #self.events  = [tuple([line.strip().split('\t')[0],float(line.strip().split('\t')[1]),float(line.strip().split('\t')[impactm + 2])]) for line in gzip.open(inputfile) if 'year' not in line and 'None' not in line.strip().split('\t')[impactm+ 2] and not line.strip().split('\t')[0].isspace()]  
 

    ### this gices back the impact values of the individual as a list
    def getImpactValues(self):    
        return [e[2] for e in self.events]


    ### this gices back the dates of the time eventsof during the career
    def getTimeStamps(self):
        return [e[1] for e in self.events]   
        
    
    ### def products by year
    def getYearlyProducts(self):
    
        time_series = {}
        for e in self.events:
            try:
                year = float(e[1])
                if year not in time_series:
                    time_series[year] = [e[2]]
                else:
                    time_series[year].append(e[2])
            except:
                pass
    
        return time_series
    
    
    ### this gives back the best impact
    def getMaxImpact(self):
        
        try:
            return max([e[2] for e in self.events])           
        except ValueError:
            return 'nan'
         
        
         
    
    ### if we are looking for the max value and is is degenerated, then this  function gives back _ALL_
    ### the top value time events - here we get the rank!
    def getRankOfMaxImpact(self):
        
        N = len(self.events)
        
        try:
            maxValue      = max([e[2] for e in self.events])
            sorted_events = sorted(self.events, key=lambda tup: tup[1])  
            ranks_of_maxs = [sorted_events.index(e) for e in sorted_events if e[2] == maxValue]   
            return (ranks_of_maxs, random.choice(ranks_of_maxs), N)
        except ValueError:
            return ('nan', 'nan', 'nan')
            
            
            




def getDistribution(keys, normalized = True):
    
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    distr = np.bincount(bins) 

    if normalized == 1: distr = distr/float(np.sum(distr)) 

    return np.asarray(uniq_keys.tolist()), np.asarray(distr.tolist())




def getBinnedDistribution(x, y, nbins):

    n, _   = np.histogram(x, bins=nbins)
    sy, _  = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    return _, mean, std



#pista = SimpleCareerTrajectory('kiss_pista', 'kiss_pista.dat.gz', 0)
#print pista.getRankOfMaxImpactAll()
#print pista.getRankOfMaxImpactRand()



