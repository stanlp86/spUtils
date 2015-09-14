



"""
Once we've extrated raw fluorescence signal for both neuropil and somata, and obtained the relationship between the two using (currently)
Linear Regression, we:

-Correct for neuropil signal
-Normalize traces
-Obtain events


#load traces, npil coefficients and odors



raw_rois = pickle.load(open('/users/stan/desktop/050515/traces/sp050515a_rois.dat'))
npils = pickle.load(open('/users/stan/desktop/050515/traces/sp050515a_npils.dat'))
npil_coefs = pickle.load(open('/users/stan/desktop/050515/traces/sp050515a_npil_coefs.dat'))

#store traces
arglist = [raw_rois, npils, npil_coefs,8,3]
sp050515a = {'sp050515a_traces' :spUtils.get_normed_traces_allTrials(*arglist)}
             
#stores events
traces = next(spUtils.gen_dict_extract('corrected_rois', sp050515a))
stds = next(spUtils.gen_dict_extract('normed_stds', sp050515a))

sp050515a['events'] = spUtils.findEventsParallel(traces, stds, std_threshold_pos=2.0, std_threshold_neg = 1.0, minimum_length=10, njobs=8)

#Store odor information
parse  odor information spUtils.parseOdorsAllTrials(csv_order, csv_names, h5, raw_tif, frameAve, trial, get_all=False)


"""