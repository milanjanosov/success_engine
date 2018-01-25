import os 
import numpy as np
from multiprocessing import Process
from CareerTrajectory.careerTrajectory import SimpleCareerTrajectory
from CareerTrajectory.careerTrajectory import MultipleImpactCareerTrajectory
import time
import gzip


files = os.listdir('Data/Book/book-authors-simple-careers')


impact_id    = 1
ifield       = 'book'
normalize    = 'no'
randomized   = False
norm_factors = {}
min_rtcnt    = 0

total    = 0
tenbooks = 0
poets    = 0

for filename in files:



    date_of_birth = 0
    date_of_death = 9999

    total += 1

    try:
        for line in gzip.open('Data/Book/book-authors-simple-profiles/' + filename.split('_')[0] + '_author_bio.dat.gz'):
            if 'Year_of_birth' in line:
                dob = int(line.strip().split('\t')[1])
            if 'Year_of_death' in line:
                dod = int(line.strip().split('\t')[1])
        
        if dob > 0:
            date_of_birth = dob
        if dod > 0:
            date_of_death = dod
    except:
        pass




    individuals_career=SimpleCareerTrajectory(filename, 'Data/Book/book-authors-simple-careers/' + filename,impact_id, normalize, norm_factors, randomized, min_rtcnt, date_of_birth, date_of_death) 

    if len(individuals_career.getImpactValues()  ) > 3:
    
        tenbooks += 1

        try:
            for line in gzip.open('Data/Book/book-authors-simple-profiles/' + filename.split('_')[0] + '_author_bio.dat.gz'):
                if 'poet' in line.strip().lower():
                    poets +=1

        except:
            pass




print 'Total: ', total
print 'More than 10: ', tenbooks
print 'Poetrs: ', poets





