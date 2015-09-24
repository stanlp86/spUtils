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
    #wrapper. Pass in dict containing corrected_traces, and associated key 


def getResponders(traceDict, traceKey, trials, test_kind = 'basic', window =10, bootstrap = False):
    
    """
    Wrapper that stores significance Test output according to odor, cell, heirarchy. The value in this dict is a list of trials that have
    met the sigTest criterion. 
    
    Wrapper specifies what test to use, and supplies any arguments it may require. because each sigTest is its own function, we can run 
    multiple tests with this wrapper without requiring separate arguments for each test. 
    
    
    """
    
    from spUtils import gen_dict_extract
    from collections import defaultdict
    
    
    #ARGS
    bootstrap = bootstrap
    traceDict = traceDict
    
    traces = gen_dict_extract(traceKey, traceDict).next() #this method only returns the first found instance

    cells = gen_dict_extract('numCells', traceDict).next()
    numOdors  = gen_dict_extract('numOdors', traceDict).next()
    
    trials = gen_dict_extract('numTrials', traceDict).next()
    test_kind = test_kind
    window = window
    
    
    output = defaultdict(dict)

    for odor, cell, trial_list in sigTestWrapper(traces, numOdors, cells, trials, test_kind, window, bootstrap): 
        output[odor][cell] = trial_list
        
    return output 


        
def getcellsBasic_sp050515a(traces, odor, cell, numTrials, window):
        on_times = [ 299,  619,  938, 1258, 1578, 1897, 2217, 2536, 2856, 3175, 3495,
           3814, 4134, 4453, 4773, 5092, 5412, 5732, 6051, 6371, 6690, 7010,
           7329]
        #on_times = [x-75 for x in on_times]
        on_times = [item-1 for item in on_times]
        off_times = [item + 20 for item in on_times]
        baselines_on = [item-10*15 for item in on_times]
        baselines_off = [item+10*15 for item in off_times]
        odor_offset_on = 160 #actual start at 150, this is to compensate for onset delay Also to avoid including responses that start
            #events that start immediately before odor rpesentation. 
        odor_offset_off = 210 #this is to ensure we capture late onset responses. 

        "this is the basic sig_test"
        high_thresh = 1.5
        low_thresh = 1.5
        
       # odor_info = gen_dict_extract('odor_info', traceDict).next()
     
        for trial in range(numTrials):
            
            #entire_interval = odor_info[trial]['pre_odor_post_interval'][odor+1]
            on = on_times[odor]
            
            
            baseline_period = slice(on-window, on,1)
            odor_period = slice(on,on+window, 1)


            pre = traces[baseline_period, cell, trial]
            post = traces[odor_period, cell, trial]



            #compare mean of pre to mean of post
            
            high = pre.mean() + high_thresh*np.std(pre)
            low = pre.mean() - low_thresh*np.std(pre)

            #condition
           
            if post.mean()>high:
                yield post.mean()-pre.mean()
            elif post.mean()<low:
                yield post.mean()-pre.mean()




def getcellsBasic(traces, odor, cell, numTrials, window):
        import numpy as np
        "this is the basic sig_test"
        high_thresh = 1.5
        low_thresh = 1.5
        
        odor_info = gen_dict_extract('odor_info', traceDict).next()
     
        for trial in range(numTrials):
            
            entire_interval = odor_info[trial]['pre_odor_post_interval'][odor+1]
            on = odor_info[trial]['on'][odor+1]
            
            
            baseline_period = slice(on-window, on,1)
            odor_period = slice(on,on+window, 1)


            pre = traces[baseline_period, cell, trial]
            post = traces[odor_period, cell, trial]



            #compare mean of pre to mean of post
            
            high = pre.mean() + high_thresh*np.std(pre)
            low = pre.mean() - low_thresh*np.std(pre)

            #condition
           
            if post.mean()>high:
                yield post.mean()-pre.mean()
            elif post.mean()<low:
                yield post.mean()-pre.mean()


def sigTestWrapper(traces, numOdors, cells, numTrials, test_kind, window, bootstrap):
    """
    Iterates over each odor-cell pair and invokes sigTest of choice.
    Returns trial list of values for trials which meet condition of sigTest. 
    This is the input to the getResponders wrapper. 
    
    
    """
    
    
    if test_kind == 'basic':
        
        for cell in range(cells):

            for odor in range(numOdors):
                trial_list =  [sig_trial for sig_trial in getcellsBasic(traces, odor, cell, numTrials, window)]
                yield odor, cell, trial_list

    elif test_kind == 'basic_sp050515a':
        
        for cell in range(cells):

            for odor in range(numOdors):
                trial_list =  [sig_trial for sig_trial in getcellsBasic_sp050515a(traces, odor, cell, numTrials, window)]
                yield odor, cell, trial_list
        
        
    """
    other elif conditions specified by test_kind can be found at the end of this cell. 
    
    """




def filterSigCells(responseDict, odor, trial_thresh = 2):
    for cell, numTrials in responseDict[odor].items():
        if len(numTrials)>=trial_thresh:
            yield cell, numTrials

