import os




''' A SimpleCareerTrajectory has two fields: the id of the individual, and her career trajectory as a list of time events. '''
''' Each time event is a triple of a time stamp a rational number, e.g. success measure, and a unique id for that time event. ''' 

class SimpleCareerTrajectory:

    
    def __init__(self, name, inputfile):
        self.name   = inputfile
        self.events = [tuple([line.strip().split('\t')[0], int(line.strip().split('\t')[1]), int(line.strip().split('\t')[2])]) for line in open(inputfile)]


    ### this gices back the impact values of the individual as a list
    def getImpactValues(self):    
        return [e[2] for e in self.events]


    ### this gices back the dates of the time eventsof during the career
    def getTimeStamps(self):
        return [e[1] for e in self.events]   
        
    
    ### this gives back the best impact
    def getMaxImpact(self):
        return max([e[2] for e in self.events])      
         
    
    ### if we are looking for the max value and is is degenerated, then this  function gives back _ALL_
    ### the top value time events - here we get the rank!
    def getRankOfMaxImpactAll(self):
        
        maxValue      = max([e[2] for e in self.events])
        sorted_events = sorted(self.events, key=lambda tup: tup[1])
       
        return [sorted_events.index(e) for e in sorted_events if e[2] == maxValue]
     
     
    ### here the same just but getting the entire event
    def getEventOfMaxImpactAll(self):
        
        maxValue      = max([e[2] for e in self.events])
        sorted_events = sorted(self.events, key=lambda tup: tup[1])
        
        return [e for e in sorted_events if e[2] == maxValue] 
        

    ### if we are looking for the max value and is is degenerated, then this  function gives back _ONE RANDOMLY_
    ### of the top value time events   
    def getRankOfMaxImpactRandom(self):  
        return 0  
    
    
    def getEventOfMaxImpactRandom(self):  
        return 0  


pista = SimpleCareerTrajectory('kiss_pista', 'kiss_pista.dat')

print pista.getRankOfMaxImpactAll()
