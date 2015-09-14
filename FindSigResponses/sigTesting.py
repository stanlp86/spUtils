#wrapper. Pass in dict containing corrected_traces, and associated key 


__all__ = ['getResponders','func', 'testSignificance']

import numpy  as np
from spUtils import gen_dict_extract
from collections import defaultdict
from scipy.stats import  wilcoxon as signed_wilcox
from scipy.stats import ranksums


def getResponders(traceDict,traceKey, ttest):
    
    #ARGS
    traces = gen_dict_extract(traceKey, traceDict).next() #this method only returns the first found instance

    cells = gen_dict_extract('numCells', traceDict).next()
    numOdors  = gen_dict_extract('numOdors', traceDict).next()
    trials = range(3)
    
    
    output = defaultdict(dict)

    for odor, cell, trial_list in func(numOdors,cells): 
        output[odor][cell] = trial_list
        
    return output 




def func(numOdors, cells):
    #while True:
    for cell in range(cells):

        for odor in range(numOdors):
            trial_list =  [sig_trial for sig_trial in testSignificance(traces, odor, cell, trials)]
            yield odor, cell, trial_list
               

 
def testSignificance(traces, odor, cell, trials, ttest):
    
    thresh = 0.05
   
    if ttest == signed_wilcox:
        #these must be equal in length
        baseline_period = slice(on_times[odor]-20, on_times[odor],1)
        odor_period = slice(on_times[odor],off_times[odor], 1)

        #specify portions of trace to compare
        this_cell_pre = traces[baseline_period, cell, :]
        this_cell_post = traces[odor_period, cell, :]

        #while True:
        for trial in range(trials):

            #compare mean of pre to mean of post
            mean_pre = this_cell_pre[:,trial]
            mean_post = this_cell_post[:,trial]

            out = signed_wilcox(mean_pre, mean_post)[1]
            #print out
            if out < thresh:

                yield out
                
    elif ttest == ranksums:
        print 'ttset not yet implemented'
    