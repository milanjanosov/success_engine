labelQ, DQ, meanQ, varianceQ, skewnessQ, kurtosityQ = zip(*[[ff for ff in line.strip().split('\t') ] for line in open('ResultData/5_pQ_fit/STAT_log_Q.dat') if 'label' not in line])
labelp, Dp, meanp, variancep, skewnessp, kurtosityp = zip(*[[ff for ff in line.strip().split('\t') ] for line in open('ResultData/5_pQ_fit/STAT_log_p.dat') if 'label' not in line])

Qvars = {}
pvars = {}

for index, label in enumerate(labelQ):
    Qvars[label] = variance[index]

for index, label in enumerate(labelp):
    pvars[label] = variance[index]

#x, y = zip*([(Qvars[label], pvars[label]) for label in labelQ])    


for label in labelQ[0:1]:   


    P_data = (Qvars[label], pvars[label])
    P_diag = (Qvars[label], Qvars[label])
    P_orig = (0,0)

    print norm(np.cross(P_diag - P_orig, P_orig - P_data))/norm(P_diag - P_org)


    a = abs(Qvars[label] - pvars[label])
    b = abs(pvars[label] - Qvars[label])
    c = (a**2 + b**2)**(1/2)
    T = a*b
    m = T/c


    print m

    



#y0 

