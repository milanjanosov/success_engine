import os
import gzip
import random
import math
import numpy as np



''' A MultipleImpactCareerTrajectory has .... fields: .................................................'''

class MultipleImpactCareerTrajectory:

    def __init__(self, name, inputfile, norm_factors, randomized):
        self.name    = inputfile
        events       = []
      
        for line in gzip.open(inputfile):           
            
            if 'year' not in line:
                fields  = line.strip().split('\t')
                #print fields
                
                product = fields[0]
                impacts = [0 if imp is None else imp  for imp in fields[2:] ]    
                impacts = [0 if 'None' in imp else imp  for imp in fields[2:] ]                                        
                try:              
                    year    = float(fields[1]) 
                    if year > 1500 and year < 2018:
                        if len(norm_factors) > 0:
                            impacts = [float(impacts[i])/norm_factors[i][year] if year in norm_factors[i]  else 0  for i in range(len(norm_factors))  ]
                        events.append((product, year, [str(i) for i in impacts]))
                except:
                    pass
        
        if randomized and len(events) > 0:
            
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
            return [event[0] + '\t' + '\t'.join(event[2]) for event in self.events]
        else:
            return []
       
        
        



''' A SimpleCareerTrajectory has .... fields: .................................................'''

class SimpleCareerTrajectory:

    
    def __init__(self, name, inputfile, impactm, norm_factors, randomized, min_rating_count):
        self.impactm = impactm
        self.name    = inputfile
        events       = []   
                  
           
        for line in gzip.open(inputfile):
        
            line = line.replace(',','')

            if 'year' not in line:
                                               
                fields  = line.strip().split('\t')
                product = fields[0]
                
                cango = False
                
                if 'film' in self.name or 'book' in self.name:
                    try:
                        if float(fields[3]) > min_rating_count:
                            cango = True
                    except:
                        pass
                
                try:
                    if 'music' in self.name:
                        if float(fields[2]) > min_rating_count:
                            cango = True

                    year    = float(fields[1])
                    impact  = float(fields[impactm + 2])
                    if impact > 0 and year > 1850 and year < 2018 and cango:
                        if len(norm_factors) > 0:
                            impact = impact/norm_factors[year]

                        if impact > 0:
                            events.append((product, year, impact))
                except:  
                    pass

        
        if randomized and len(events) > 0:
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
        
        

    ### get the autocorrelation
    def getAutoCorrelCoeff(self):
    
        N = len(self.events)
        
        if N > 4:
 
            values   = [e[2] for e in self.events]
            maxValue = max([e[2] for e in self.events])
            maxIndex = sorted(values).index(maxValue)
            sigma2   = np.std(values)**2
            rho      = sum([( maxValue * values[tau % N ] - maxValue**2 )/(sigma2) for tau in range(N)]) / N
              
            return 0 #rho 

        else:   
              return 0
            
    
    ### this gives back the best impact
    def getMaxImpact(self):
        
        try:
            return max([e[2] for e in self.events])           
        except ValueError:
            return 'nan'
         
        
    def getTimeOfTheBest(self):
    
        if len(self.events) > 0:
    
            firstTime     = min([e[1] for e in self.events])
            maxValue      = max([e[2] for e in self.events])
            sorted_events = sorted(self.events, key=lambda tup: tup[1])          
            time_of_max   = min([e[1] for e in sorted_events if e[2] == maxValue] )

            return (time_of_max - firstTime)
        
        else:
            return -1
             
    
    ### if we are looking for the max validtue and is is degenerated, then this  function gives back _ALL_
    ### the top value time events - here we get the rank!
    def getRankOfMaxImpact(self):
        
        N = len(self.events)
        
        try:
            maxValue      = max([e[2] for e in self.events])
            sorted_events = sorted(self.events, key=lambda tup: tup[1])  
            ranks_of_maxs = [sorted_events.index(e)+1 for e in sorted_events if e[2] == maxValue] 
            #print ranks_of_maxs  
            return (ranks_of_maxs, random.choice(ranks_of_maxs), N)
        except ValueError:
            return ('nan', 'nan', 'nan')
            
            
    # log p_alpha = log_c_10ialpha - log Q_i
    # assume Q_i is constant over the career and P(p) is lognormal:
    # log Q_i = <log c_10ialpha> - mu_p
    def getLogPwithZeroAvg(self):
    
        log_impacts = [math.log(e[2]) for e in self.events]    
        log_impacts_avg = np.log(np.mean([e[2] for e in self.events]))
          
        return [i - log_impacts_avg for i in log_impacts]




def getDistribution(keys, normalized = True):
    
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    distr = np.bincount(bins) 

    if normalized == 1: distr = distr/float(np.sum(distr)) 

    return np.asarray(uniq_keys.tolist()), np.asarray(distr.tolist())




def getBinnedDistribution(x, y, nbins):

    n, bins   = np.histogram(x, bins=nbins)
    sy, _  = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)


    #print 'NA', len((_[1:] + _[:-1])/2), len(_)

    return _, mean, std


def getLogBinnedDistribution(x, y, nbins):

    bins   = 10 ** np.linspace(np.log10(min(x)), np.log10(max(x)), nbins)  
    values = [ np.mean([y[j]  for j in range(len(x)) if x[j] >= bins[i] and x[j] < bins[i+1]])  for i in range(nbins-1)]    
    error  = [ np.std( [y[j]  for j in range(len(x)) if x[j] >= bins[i] and x[j] < bins[i+1]])  for i in range(nbins-1)]
    bins   = (bins[1:] + bins[:-1])/2

    return bins, values, error








#pista = SimpleCareerTrajectory('kiss_pista', 'kiss_pista.dat.gz', 0, {}, False)

#print pista.events
#print pista.getTimeOfTheBest()
#
#print pista.getAutoCorrelCoeff()
#print pista.getLogPwithZeroAvg()

#print pista.getCareerLength()
#print pista.getRankOfMaxImpact()

#print pista.getYearlyProducts()

#gyurika = MultipleImpactCareerTrajectory('george_lucas', 'george_lucas.gz', [], True)
#gyurika.getImpactValues()
#print '\n'

#for imp in  gyurika.getImpactValues():
#    print imp

