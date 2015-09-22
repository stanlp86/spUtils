

import numpy  as np

__all__ = ['get_normed_traces_byTrial','get_normed_traces_allTrials', 'normalize_allTraces', 'normalize_oneTrace', 'get_first_baseline_allTraces', 'get_first_baseline_oneTrace', 'get_second_baseline_allTraces', 'get_second_baseline_oneTrace']



#The Two functions below constitute Step 1 of our baselining routine: correcting long-term changes in baseline

def get_first_baseline_allTraces(rois, window = 150, Fluor_percentile = 5, njobs = 8):
    from scipy.stats import scoreatpercentile as score
    from joblib import Parallel, delayed
    import cPickle as pickle
    """

    params: rois- trace array, raw or neuropil subtracted, of shape [frames, cells, trial]
            window - width of sliding window in frames corresponding to secPerFrame*window seconds. 
            Fluor_percentile - percentile of fluorescence distribution at which to calculate score. 
    returns: List of tuples of size rois.shape[1]. This should be number of cells.       

    """
    
    win = window/2
    
    baseline_and_baselined_trace = Parallel(n_jobs=njobs)(delayed(get_first_baseline_oneTrace)(rois[:,cell], window, Fluor_percentile) for cell in range(rois.shape[1]))
    return baseline_and_baselined_trace




def get_first_baseline_oneTrace(trace, window, Fluor_percentile):
    from scipy.stats import scoreatpercentile as score
    """
    params: trace - 1d array of shape [frames]
            window - width of sliding window in frames corresponding to secPerFrame*window seconds. 
            Fluor_percentile - percentile of fluorescence distribution at which to calculate score.   
    """
    win = window/2
    
    baseline = np.array([score(trace[s-win:s+win], Fluor_percentile) for s in range(win,(trace.shape[0]-win))])
    baselined_trace = trace[win:-win]-baseline
    
    return baseline, baselined_trace


#The Two functions below constitute Step 2 of our baselining routine: correcting long-term changes in baseline

def get_second_baseline_allTraces(traces, SD_window = 20, SD_percentile = 5, njobs = 8):
    from scipy.stats import scoreatpercentile as score
    from joblib import Parallel, delayed
   
    
    """
    params: traces- trace array, after subtraction of first baseline. Shape should be [frames, cells, trial]
            window - width of sliding window in frames corresponding to secPerFrame*window seconds. 
            Fluor_percentile - percentile of fluorescence distribution at which to calculate score. 
    returns: tuple. First value is list of floats corresponding to baselines. Second value is list of index arrays where
             each array of indeces corresponds to all locations in trace where SD==SD_percentile
    
    """
    
    second_baseline_andIndeces = Parallel(n_jobs=njobs)(delayed(get_second_baseline_oneTrace)(traces[:,cell], SD_window, SD_percentile) for cell in range(traces.shape[1]))
    return second_baseline_andIndeces
    

def get_second_baseline_oneTrace(trace, SD_window, SD_percentile):
    from scipy.stats import scoreatpercentile as score
    """
    params: traces- trace array, after subtraction of first baseline. Shape should be [frames, cells, trial]
            window - width of sliding window in frames corresponding to secPerFrame*window seconds. 
            Fluor_percentile - percentile of fluorescence distribution at which to calculate score. 
            
    returns: second_baseline - single value, most common position in trace where SD == SD_percentile
             idx - 1d array of indeces that correspond to times in trace where SD == SD_percentile
                   Can be plugged into normalized_SD array in Normalization step. 
             rolling_SD
    
    """
    
    
    win = SD_window/2
    
    rolling_SD = np.array([trace[s-win:s+win].std() for s in range(win,(trace.shape[0]-win))])
    #Get SD value at 'percentile_val' percentile
    SD = score(rolling_SD, SD_percentile)
    SD = np.round(SD)
    
    #Find most common position in trace where SD is at the 5th percentile. 
    
    #find times where std is minimal.
    idx = np.argwhere(np.round(rolling_SD) == SD)
    
    # find the most common intensity value  at this index. This is the baseline value of the entire trace. 
    
    #get the median of the largest bin of the histogram of the range of trace[idx] values
    #specify bin size: 
    bins = np.round((idx[:,0].shape[0])/10.0)
    a,b = np.histogram(np.round(trace[win:-win][idx]), bins = bins)
    #get the range of trace[idx] values that reside within the largest bin
    bin_num = np.argwhere(b==b[a==a.max()][0])[0][0]
    left_edge = b[bin_num]
    right_edge = b[bin_num+1]
    
    #this is the median val...the baseline.
    second_baseline = score(np.unique(trace[idx].clip(left_edge,right_edge)),50)
    
    return second_baseline, np.squeeze(idx), rolling_SD

    #The Two functions below constitute Step 3 of our baselining routine: correcting long-term changes in baseline

def normalize_allTraces(baselined_traces, first_baselines, second_baselines, rolling_SDS,  
                    SD_idx_array_list, SD_window = 20, njobs = 8):
    from joblib import Parallel, delayed
    """
    params: traces - baselined_traces. Output of step 1. After subtraction of first baseline.
            first_baseline - 1d array. output from step 1. 
            second_baseline - single float. Output from step 2. 
            idx - single array of indeces where SD == SD_percentile. 
            rolling_SD:
    returns:         
    """
    
    
        
    normalized_trace_and_sd = Parallel(n_jobs=njobs)(delayed(normalize_oneTrace)(baselined_traces[:,cell], first_baselines[:,cell], second_baselines[cell], rolling_SDS[cell], SD_idx_array_list[cell], SD_window) for cell in range(baselined_traces.shape[1]))
    return normalized_trace_and_sd

def normalize_oneTrace(trace, first_baseline, second_baseline, rolling_SD, idx, SD_window = 20):
    from scipy.stats import scoreatpercentile as score
    """
    params: trace - 1d array. Output of step 1. After subtraction of first baseline.
            first_baseline - 1d array. output from step 1. 
            second_baseline - single float. Output from step 2. 
            idx - single array of indeces where SD == SD_percentile. 
            rolling_SD:
    returns:         
    """
    win = SD_window/2
    
    normed_trace = (trace - second_baseline)/(first_baseline + second_baseline) #baseline is the output of step 2
    normed_rolling_SD = (rolling_SD)/(first_baseline[win:-win] + second_baseline) #rolling SD obtained from step 2
    
    sd_vals = np.unique(np.round(normed_rolling_SD[idx], 3))
    normed_SD = score(sd_vals, 50)
    
    return normed_trace, normed_SD
    

    #This is how these steps are combined: 
"""
    #func 1: detrend
        
        step1 = get_first_baseline_allTraces(rois[:,:,0], window = 150, Fluor_percentile = 5, njobs = 8)

        baselines = np.vstack([tup[0] for tup in step1]).T #make sure size is [frames, cells]
        baselined_traces = np.vstack([tup[1] for tup in step1]).T  #make sure size is [frames, cells]
            
    #func 2: find bline
        
        step2 = get_second_baseline_allTraces(traces, window = 20, SD_percentile = 5, njobs = 8)
        
        second_baselines = np.squeeze(np.vstack([tup[0] for tup in step2]))
        SD_idx_array_list = [tup[1] for tup in step2]
        rolling_SDS = [tup[2] for tup in step2]
        
    #func 3: normalize
       
        step3 = normalize_allTraces(baselined_traces, baselines, second_baselines, rolling_SDS,  
                                SD_idx_array_list, SD_window = 20, njobs = 8)
        normalized_traces = np.vstack([tup[0] for tup in step3]).T
        normalized_SDS = np.squeeze(np.vstack([tup[1] for tup in step3]))

"""




###################################################################################################################
###################################################################################################################
###################################################################################################################
#
def get_normed_traces_allTrials(raw_rois, npils, npil_coefs, window=150, SD_window=20, SD_percentile=5, Fluor_percentile=5, njobs=8, numTrials=3, method = 2, subtracted=True):
    import numpy as np
    """
    Current conditions: raw_rois, npils, npil_coefs, window=150, 
    SD_window=20, SD_percentile=5, Fluor_percentile=5, 
    njobs=8, numTrials=3, method = 2, subtracted=True
    
    make this into a dict. 
    
    """
    #this takes a 
    out = [get_normed_traces_byTrial(raw_rois[...,trial], npils[...,trial], npil_coefs[...,trial], window, SD_window, SD_percentile, Fluor_percentile, njobs, method, subtracted) for trial in range(numTrials)] #list of dicts containing traces and stds
    
    return {'corrected_rois': np.swapaxes(np.asarray([out[trial]['corrected_rois'] for trial in range(numTrials)]).T,0,1),
            'normed_stds':np.asarray([out[trial]['normed_stds'] for trial in range(numTrials)]).T}

def get_normed_traces_byTrial(rois, npils, coefs, window, SD_window, SD_percentile, Fluor_percentile, njobs, method, subtracted):
    import sys
    import numpy as np
    from joblib import Parallel, delayed
    import cPickle as pickle
    from time import time
    
    
    
    if method == 2:
        
        #func 1: detrend
        
        step1 = get_first_baseline_allTraces(rois, window = 150, Fluor_percentile = 5, njobs = 8)

        baselines = np.vstack([tup[0] for tup in step1]).T #make sure size is [frames, cells]
        baselined_traces = np.vstack([tup[1] for tup in step1]).T  #make sure size is [frames, cells]
            
        #func 2: find bline
        
        step2 = get_second_baseline_allTraces(baselined_traces, SD_window = 20, SD_percentile = 5, njobs = 8)
        
        second_baselines = np.squeeze(np.vstack([tup[0] for tup in step2]))
        SD_idx_array_list = [tup[1] for tup in step2]
        rolling_SDS = [tup[2] for tup in step2]
        
        #func 3: normalize
        
        
        step3 = normalize_allTraces(baselined_traces, baselines, second_baselines, rolling_SDS,  
                                SD_idx_array_list, SD_window = 20, njobs = 8)
        normalized_traces = np.vstack([tup[0] for tup in step3]).T
        normalized_SDS = np.squeeze(np.vstack([tup[1] for tup in step3]))

        out = {'corrected_rois': normalized_traces,
                'normed_stds': normalized_SDS}
        return out
        
    elif method == 1:

        #for every raw cell signal in this trial fit with gaussian mixture model. To get baseline estimate
        raw_cell_gmmOut = Parallel(n_jobs=njobs)(delayed(fitGaussianMixture1D_raw)(rois[:,cell]) for cell in range(rois.shape[1]))
        npils_cell_gmmOut = Parallel(n_jobs=njobs)(delayed(fitGaussianMixture1D_raw)(npils[:,cell]) for cell in range(npils.shape[1]))

        raw_means = np.vstack([means_from_gmmOut(i) for i in raw_cell_gmmOut])[:,0]
        raw_stds = np.vstack([stds_from_gmmOut(i) for i in raw_cell_gmmOut])[:,0]

        npils_means = np.vstack([means_from_gmmOut(i) for i in npils_cell_gmmOut])[:,0]
        npils_stds = np.vstack([stds_from_gmmOut(i) for i in npils_cell_gmmOut])[:,0]

        #Normalize both cell and neighborhood
        rois_normed = rois/raw_means -1
        npils_normed = npils/npils_means-1

        #subtract neuropil or not
        #subtract neuropil or not
        if subtracted:
            corrected_rois = rois_normed - abs(npils_normed)*coefs
        else:
            corrected_rois = rois_normed 


        #get baseline estimate of corrected normed trace for event detection
        normed_cell_gmmOut = Parallel(n_jobs=njobs)(delayed(fitGaussianMixture1D_normed)(corrected_rois[:,cell]) for cell in range(corrected_rois.shape[1]))

        normed_means = np.vstack([means_from_gmmOut(i) for i in normed_cell_gmmOut])[:,0]
        normed_stds = np.vstack([stds_from_gmmOut(i) for i in normed_cell_gmmOut])[:,0]
        corrected_rois = corrected_rois-normed_means #baseline again
        
    
    
    #these are used for thresholding for events make sure correspond
    inters = {'rois_normed': rois_normed,
        'npils_normed': npils_normed,
            'raw_means': raw_means,
                'npils_means': npils_means}

    out = {'corrected_rois': corrected_rois,
        'normed_stds': normed_stds}
    return out



















findEventsParams = {'event_start_thresh': 0.5, 'std_threshold_neg': 2.25, 'minimum_length': 10, 'std_threshold_pos': 2.25, 'percentile_above_std': 80}



#designed to work with: 
#traces = normed_trace.copy()[:,np.newaxis]

def findEvents(trace, std, std_threshold_pos=1.5, std_threshold_neg = 1.5, percentile_above_std = 80, minimum_length=10, event_start_thresh = 0.0, positive=True):
    import mahotas
    import scipy as sp
    
    
    """ Modified from d_code events module by AJG
        Core event finding routine with flexible syntax.
    
    An event begins and ends at 1std from baseline and is at least 1 second long. 
    Each event's maximum is above 2std from baseline 
    
    :param: traces - 1d array
    :param: std - float
    :param: std_threshold - multiple of per-cell STD to use for an event (float)

    :param: minimum_length - minimum length of an event
    :param: alpha - optional scaling parameter for adjusting thresholds
    :event_start_thresh - where to start and end event in units of sigma from baseline.
    :returns: numpy array same shape and size of traces, with each event given a unique integer label. returns one for pos, 1 for neg. 
    
    """

    trace = trace
    std = std
   # if traces.ndim == 2:
     #   traces = np.atleast_3d(traces) # time x cells x trials
     #  
        #stds = np.atleast_2d(stds).T # cells x trials
    #time, cells, trials = traces.shape
   # print time, cells, trials, stds.shape
    pos_events = np.zeros_like(trace)
    neg_events = np.zeros_like(trace)

    event_cutoff_pos = std * float(std_threshold_pos)
    event_cutoff_neg = std * float(std_threshold_neg)*(-1)
    event_start_thresh = event_start_thresh #in sigma
    # detect  events
    
    if positive:
        
        #first find where trace deviates above /belowevent_start_thresh
        pos_events = trace > std*event_start_thresh # here we assume the mean is at 0.0 since we've already baselined. 
        # filter for minimum length
        pos_events = mahotas.label(pos_events, np.array([1,1]))[0]
        
        for single_event in range(1, pos_events.max()+1): 

            if (pos_events == single_event).sum() <= minimum_length:
                pos_events[pos_events == single_event] = 0
        pos_events = pos_events>0
        
        #filter for actual event cutoff
        pos_events = mahotas.label(pos_events, np.array([1,1]))[0]
        
        #return pos_events
        for single_event in range(1, pos_events.max()+1):

            idx = np.argwhere(pos_events==single_event)  
            #return trace[idx[:,0]]
            if sp.stats.scoreatpercentile(trace[idx[:,0]],percentile_above_std) <= event_cutoff_pos:

                pos_events[pos_events == single_event] = 0
        pos_events = pos_events>0
        
        #label and return
        pos_events = mahotas.label(pos_events, np.array([1,1]))[0]
        return pos_events
    else:
        
        neg_events = trace < (-1.0)*std *event_start_thresh
        neg_events = mahotas.label(neg_events, np.array([1,1]))[0]
        
        for single_event in range(1, neg_events.max()+1):
            if (neg_events == single_event).sum() <= minimum_length:
                neg_events[neg_events == single_event] = 0
        neg_events = neg_events>0
        
   
        neg_events = mahotas.label(neg_events, np.array([1,1]))[0]
        
        for single_event in range(1, neg_events.max()+1):
            idx = np.argwhere(neg_events==single_event)        
            if sp.stats.scoreatpercentile(trace[idx[:,0]],percentile_above_std) >= event_cutoff_neg:
                neg_events[neg_events == single_event] = 0
        neg_events = neg_events>0
        
        neg_events = mahotas.label(neg_events, np.array([1,1]))[0]

        return neg_events


def epoch_event_generator(traces, stds, cells, trials, **findEventsParams):
    
    
    return np.dstack([trial for trial in trial_event_generator(traces, stds, cells, trials, **findEventsParams)])
    
def trial_event_generator(traces, stds, cells, trials, **findEventsParams):
    for trial in range(trials):
        yield np.swapaxes(np.dstack([findEvents(traces[:,cell,trial], stds[cell,trial], **findEventsParams)
                 for cell in range(cells)]).T,1,0)


def getMaxEvents(event_array, trace_array):
    frames, cells, trials = trace_array.shape
    """This routine takes an event array and corresponding trace array
    and replaces the event labels with the average amplitude of the
    event.

    :param: event_array - 2 or 3d numpy event array (time x cells, or time x cells x trials)
    :param: trace_array - 2 or 3d numpy event array (time x cells, or time x cells x trials)
    :returns: 2d numpy array same shape and size of event_array, zero where there
              weren't events, and the average event amplitude for the event otherwise.
    """
    weighted_events = np.zeros_like(event_array, dtype=float)
    for trial in range(trials):
        for cell in range(cells):
        
            for i in np.unique(event_array[:,cell,trial])[1:]:
                #print i
                weighted_events[:,cell,trial][event_array[:,cell,trial]==i] = trace_array[:,cell,trial][event_array[:,cell,trial]==i].max()
                #print trace_array[:,cell,trial][event_array[:,cell,trial]==i].max()
    return weighted_events
