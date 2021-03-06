import os
import gzip
import random
import math
import numpy as np
from scipy.stats import binned_statistic


def yearIsOK(year, date_of_birth, date_of_death):

    yearIsOK = False    

    lower_limit = max(1800, date_of_birth)
    upper_limit = min(2018, date_of_death)

    if year >= lower_limit and year <= upper_limit:
        yearIsOK =  True
   
    return yearIsOK
    


''' A MultipleImpactCareerTrajectory has .... fields: .................................................'''

class MultipleImpactCareerTrajectory:

    def __init__(self, name, inputfile, norm_factors, randomized, date_of_birth, date_of_death):
        self.name    = inputfile
        events       = []
      
        for line in gzip.open(inputfile):           
            
            if 'year' not in line:
                fields  = line.strip().split('\t')

               # print fields
                #print fields
                
                product = fields[0]
                impacts = [0 if imp is None else imp  for imp in fields[2:] ]    
                impacts = [0 if 'None' in imp else imp  for imp in fields[2:] ]    

                                    
                try:              
                    year    = float(fields[1]) 
                    if yearIsOK(year, date_of_birth, date_of_death):
                        if len(norm_factors) > 0:
                            impacts = [float(impacts[i])/norm_factors[i][year] if year in norm_factors[i]  else 0  for i in range(len(norm_factors))  ]
                        #print product , year, [str(i) for i in impacts]
                        events.append((product, year, [str(i) for i in impacts]))
                except:
                    pass
        
        '''if randomized and len(events) > 0:
            
            events_rand = []
            impacts_to_rand = []
            for impact in range(len(events[0][2])):
                impacts_to_rand.append([event[2][impact] for event in events])             
                random.shuffle(impacts_to_rand[impact])
         
            for i in range(len(events)):           
                events_rand.append((events[i][0], events[i][1], [impacts_to_rand[impact][i] for impact in range(len(events[0][2]))] ))
            
            events = events_rand
        '''
        #print events                                      

        self.events = events   


    def getImpactValues(self):
        
        if len(self.events) > 0:
            num_impacts    = len(self.events[0][2]) 

            #print self.events

            return [event[0] + '\t' + '\t'.join(event[2]) for event in self.events]
        else:
            return []
       
        
        



''' A SimpleCareerTrajectory has .... fields: .................................................'''

class SimpleCareerTrajectory:

    
    def __init__(self, name, inputfile, impactm, normalize, norm_factors, randomized, min_rating_count, date_of_birth, date_of_death, fitted_Qp):
        self.impactm    = impactm
        self.name       = inputfile
        self.fitted_Qp  = fitted_Qp
        events          = []   


        
           
        for line in gzip.open(inputfile):
        
            line = line.replace(',','')

            if 'year' not in line:
            

          

                                               
                fields  = line.strip().split('\t')
                product = fields[0]
                
                cango = False
              
                min_rating_count = 4.0

    
                if 'film' in self.name or 'book' in self.name:
                    


                    try:
                        if int(fields[3]) > min_rating_count:
                            cango = True
                    except:
                        pass
                
               
                if 'music' in self.name:
                    try:
                        if float(fields[2]) > min_rating_count:
                            cango = True
                    except:
                        pass
                

               
                try:
                #if 2 == 2:
                    year    = float(fields[1])
                    impact  = float(fields[impactm + 2])



                    if year > 1500:# and year < 2016:
                
                        

                        if impact > 0 and yearIsOK(year, date_of_birth, date_of_death) and cango:
                            if 'no' not in normalize:

                                impact = impact/norm_factors[year]


                                
                            if impact > 0:
                                events.append((product, year, impact))
                except:  
                    pass

        if len(events) >  0:
            min_year = min([e[1] for e in events])
            events = [event for event in events if (event[1] - 81 < min_year)]
                   
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
    '''def getRankOfMaxImpact(self):
        
        N = len(self.events)
        
        try:
            maxValue      = max([e[2] for e in self.events])
            sorted_events = sorted(self.events, key=lambda tup: tup[1])  
            ranks_of_maxs = [sorted_events.index(e)+1 for e in sorted_events if e[2] == maxValue] 
            #print ranks_of_maxs  
            return (ranks_of_maxs, random.choice(ranks_of_maxs), N)
        except ValueError:
            return ('nan', 'nan', 'nan')
       
    '''
    def getRankOfMaxImpact(self):
        
        N = len(self.events)
        
        try:
            maxValue      = max([e[2] for e in self.events])
            sorted_events = sorted(self.events, key=lambda tup: tup[1])         
            years         = sorted(list(set([s[1] for s in sorted_events])))  
            
            maxes = []
            for s in sorted_events:
                if s[2] == maxValue:
                    maxes.append(years.index(s[1]) + 1 )                

            return (maxes, random.choice(maxes), len(years))

        except ValueError:
            return ('nan', 'nan', 'nan')
 



    def getRankOfMaxImpact(self):
        
        N = len(self.events)
        
        try:
            maxValue      = max([e[2] for e in self.events])
            sorted_events = sorted(self.events, key=lambda tup: tup[1])         
            years         = sorted(list(set([s[1] for s in sorted_events])))  
            
            maxes = []
            for s in sorted_events:
                if s[2] == maxValue:
                    maxes.append(years.index(s[1]) + 1 )                

            return (maxes, random.choice(maxes), len(years))

        except ValueError:
            return ('nan', 'nan', 'nan')
            
 


           
    
    # log p_alpha = log_c_10ialpha - log Q_i
    # assume Q_i is constant over the career and P(p) is lognormal:
    # log Q_i = <log c_10ialpha> - mu_p
    def getLogPwithZeroAvg(self):
    
        log_impacts     = [math.log(e[2]) for e in self.events]    
        log_impacts_avg = np.log(np.mean([e[2] for e in self.events]))
          
        return [i - log_impacts_avg for i in log_impacts]



    def getexactp(self):
    
        logQ        = math.log(self.getExactQ())
        log_impacts = [math.log(e[2]) for e in self.events]    
       
        return [math.exp(i - logQ) for i in log_impacts]


        
    def getLogQ(self):

        return np.mean([ np.log(e[2]) for e in self.events] ) #- mu_p

        
    def getApproxQ(self):



        mu_p = self.fitted_Qp[1]
        return math.exp(np.mean([ np.log(e[2]) for e in self.events] ) - mu_p ) 





       
    def getExactQ(self):


        fitted_Qp = self.fitted_Qp


        mu_N     = fitted_Qp[0]   #    2.2
        mu_p     = fitted_Qp[1]   #    0.2
        mu_Q     = fitted_Qp[2]   #    1.1
        sigma_N  = fitted_Qp[3]   #    2.1
        sigma_Q  = fitted_Qp[4]   #    1.4
        sigma_p  = fitted_Qp[5]   #    1.4
        sigma_pQ = fitted_Qp[6]   #    0.01
        sigma_pN = fitted_Qp[7]   #    0.02
        sigma_QN = fitted_Qp[8]   #    0.12


        logN_i    = math.log(len(self.events))
        N_i       = len(self.events)
        K_QN      = sigma_Q**2 * sigma_N**2 - sigma_QN**2  
        avg_log_c = np.mean([ np.log(e[2]) for e in self.events])
      
  
        TERM1 = (  sigma_N**2 * sigma_p**2 * mu_Q + sigma_QN * sigma_p**2 * (logN_i - mu_N)  ) / (N_i * K_QN)
        TERM2 = sigma_N**2 * sigma_p**2 / ( N_i * K_QN)

        logQ  =  (avg_log_c - mu_p + TERM1)  /  (1.0 + TERM2)


        return math.exp(  logQ   ) 









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
    mean = sy/n
    
   
    std = np.sqrt(sy2/n - mean*mean) 

    #print 'NA', len((_[1:] + _[:-1])/2), len(_)

    return _, mean, std


def getLogBinnedDistribution(x, y, nbins):


    bins   = 10 ** np.linspace(np.log10(min(x)), np.log10(max(x)), nbins)  
    values = [ np.mean([y[j]  for j in range(len(x)) if x[j] >= bins[i] and x[j] < bins[i+1]])  for i in range(nbins-1)]    
    error  = [ np.std( [y[j]  for j in range(len(x)) if x[j] >= bins[i] and x[j] < bins[i+1]])  for i in range(nbins-1)]
    bins   = (bins[1:] + bins[:-1])/2

    return bins, values, error



def getPercentileBinnedDistribution(x, y, nbins):

 
    x, y = zip(*sorted(zip(x, y), key=lambda tup: tup[0]))
    elements_per_bin = len(x)

    xx  = [np.mean(x[i*elements_per_bin:(i+1)*elements_per_bin]) for i in range(nbins)]
    yy  = [np.mean(y[i*elements_per_bin:(i+1)*elements_per_bin]) for i in range(nbins)]
    std = [np.std(y[i*elements_per_bin:(i+1)*elements_per_bin])  for i in range(nbins)]
     
    return xx, yy, std








'''opt_sol = 0
for ind, line in enumerate(open('../QFitConstants/art-director.dat')):
    if ind == opt_sol:
        maxfitness, mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN = [float(fff) for fff in line.strip().split('\t')]
        print mu_p
        #mu_p = 15.0
        fitted_Qp = (mu_N, mu_p, mu_Q, sigma_N, sigma_Q, sigma_p, sigma_pQ, sigma_pN, sigma_QN)




#pista = SimpleCareerTrajectory('filmbook', 'music_kiss_pista.dat.gz', 0, 'no', {}, False, min_rating_count = 0, date_of_birth = 0, date_of_death = 9999, fitted_Qp = fitted_Qp)
pista = SimpleCareerTrajectory('film', 'film_george_lucas.gz', 1, 'no', {}, False, min_rating_count = 0, date_of_birth = 0, date_of_death = 9999, fitted_Qp = fitted_Qp)

print 'Qa\t ', pista.getApproxQ()
print 'Qe\t', pista.getExactQ()
print [round(p,3) for p in pista.getexactp()]

ps = [round(p,3) for p in pista.getexactp()]
Q  = pista.getExactQ()

for p in ps:
    print p*Q


print [t[2] for t in pista.events]
'''



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

