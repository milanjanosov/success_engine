import os
import gzip
import random
import numpy as np




''' A MultipleImpactCareerTrajectory has .... fields: .................................................'''

class MultipleImpactCareerTrajectory:

    def __init__(self, name, inputfile, norm_factors, randomized = False):
        self.name    = inputfile
        events       = []

        for line in gzip.open(inputfile):
            if 'year' not in line:
                fields  = line.strip().split('\t')
                product = fields[0]
                impacts = [imp if 'None' not in imp else '0'  for imp in fields[2:] ]
                try:    
                    year    = float(fields[1])
                    if year > 1500 and year < 2018:
                        if len(norm_factors) > 0:
                            impacts = [float(impacts[i])/norm_factors[i][year] if year in norm_factors[i]  else 0  for i in range(len(norm_factors))  ]
                        events.append((product, year, [str(i) for i in impacts]))
                except ValueError:
                    pass
        
        
        if randomized:
            
            events_rand = []
            impacts_to_rand = []
            for impact in range(len(events[0][2])):
                impacts_to_rand.append([event[2][impact] for event in events])             
                random.shuffle(impacts_to_rand[impact])
         
            for i in range(len(events)):           
                events_rand.append((events[i][0], events[i][1], [impacts_to_rand[impact][i] for impact in range(len(events[0][2]))] ))
            
            events = events_rand
                                      
        self.events = events   


    def getImpactValues(self):
        
        if len(self.events) > 0:
            num_impacts    = len(self.events[0][2]) 
            return ['\t'.join(event[2]) for event in self.events]
        else:
            return []
       
        
        



''' A SimpleCareerTrajectory has .... fields: .................................................'''

class SimpleCareerTrajectory:

    
    def __init__(self, name, inputfile, impactm, norm_factors, randomized = False):
        self.impactm = impactm
        self.name    = inputfile
        events       = []             
           
        for line in gzip.open(inputfile):
            if 'year' not in line:
                fields  = line.strip().split('\t')
                product = fields[0]
                try:    
                    year    = float(fields[1])
                    impact  = float(fields[impactm + 2])
                                    
                    if impact > 0 and year > 1850 and year < 2018:
                        if len(norm_factors) > 0:
                            impact = impact/norm_factors[year]
                        events.append((product, year, impact))
                except ValueError:
                    pass
    

        
        if randomized:
            impacts_to_rand = [e[2] for e in events]
            random.shuffle(impacts_to_rand)
            events_rand = []
            for e in events:
                events_rand.append((e[0], e[1], impacts_to_rand[events.index(e)]))

            events = events_rand

        self.events = events                                  
                    
        


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
        
        
    def getCareerLength(self):
        return len(self.events)
    
    
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





#pista = SimpleCareerTrajectory('kiss_pista', 'kiss_pista.dat.gz', 0, {}, True)
#print pista.getImpactValues()

#print pista.getCareerLength()
#print pista.getRankOfMaxImpact()

#print pista.getYearlyProducts()

#gyurika = MultipleImpactCareerTrajectory('george_lucas', 'george_lucas.gz', [], False)
#gyurika.getImpactValues()
#print '\n'

#for imp in  gyurika.getImpactValues():
#    print imp

