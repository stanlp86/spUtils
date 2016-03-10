
import numpy  as np

__all__ = ['extractTracesfromTrial_toDF', 'extractMetaData_to_df', 'kalman_filter_trace_parallel','kalman_filter_trace', 'getSNR', 'secScaleBar','xticksInSec','get_channel', 'get_keys','gen_dict_extract', 'findEventsParallel', 'findEvents','get_normed_traces_allTrials','open_folder','get_normed_traces','get_ransac_npil_coefs', 'fitRansac', 'reshape_by_odor', 'parseOdorsAllTrials','parseOdors','getFramePeriod','find_nearest', 'local_neighcorr']

def extractTracesfromTrial_toDF(directory, mask, trialID, trialIdx, zPos):   
    from d_code.imaging.segmentation import extractTimeCoursesFromSeries as extract
    import scipy
    import pandas as pd
    import cPickle as pickle
    from glob import glob
                  
    files = glob(directory + '*aligned.tif') 
    allSeriesNamesThisTrial = [fname for fname in files if trialID in fname]
    # now sort
    allSeriesNamesThisTrial = sorted(allSeriesNamesThisTrial, key = lambda x: x.split(trialID)[-1].split('_')[0])
    #These must absolutely be  sequenced correctly
    print allSeriesNamesThisTrial

    #Get X and Y coordinates
    a,b = scipy.ndimage.label(mask)
    coords = scipy.ndimage.measurements.center_of_mass(mask, a, range(b+1))
    #check for NAN and replace with 0.
    #replace NANs with 0's
    for icell, cell in enumerate(coords):
        if np.isnan(cell)[0]:
            coords.pop(icell)
            coords = [(0,0)]+coords

    all_series_df = []
    for ifname, fname in enumerate(allSeriesNamesThisTrial):
        series = io.imread(fname)
        #stopgap for correct sequencing?
        traceArray = extract(series, mask)

        #Save traces for workflow doc. check shape later. 
        out = {'rois': traceArray}
        pickle.dump(out, open(fname + 'rawAndNpil.dat', 'wb'))       

        numCells = traceArray.shape[1]                   
        
        #Make a list of dfs 
        cell_list = [pd.DataFrame(traceArray[:,cell]) for cell in range(numCells)]

        #append other attributes to each cell df. 
        for icell, cellDf in enumerate(cell_list):

            cellDf['cellID'] = icell
            cellDf['trial'] = trialIdx
            cellDf['odorIdx'] = ifname + 1
            #cell location 
            cellDf['xPos'] = coords[icell][1]
            cellDf['yPos'] = coords[icell][0]
            cellDf['zPos'] = zPos  

        series_df = pd.concat(cell_list)                  
        all_series_df.append(series_df)                   
                           
    trial_df = pd.concat(all_series_df)
    pickle.dump(trial_df,open(directory + 'tracesDF'+ trialID +'.dat', 'wb'))

def extractMetaData_to_df(sub_series_file, channel, slice_number, numSlices, blank_flyback):
    import cPickle as pickle
    import pandas as pd
    from collections import defaultdict
    import tifffile
    import numpy as np
    
    numChannels = 2
    if blank_flyback == True:
        slice_select = numSlices+1
    else:
        slice_select = numSlices
    with tifffile.TiffFile(sub_series_file) as tif:
        metadict = defaultdict(list)
        metadict['channel'] = channel
        stack_frames = range(len(tif))
        frames_by_chan = stack_frames[(channel + numChannels*slice_number)::2*slice_select]
        for frame in frames_by_chan:
            metadata = tif[frame].tags['image_description'].value.split('\n')
            for item, line in enumerate(metadata):
                try:
                    key = metadata[item].split('=')[0]
                    value = metadata[item].split('=')[1]
                except:
                    pass
                metadict[key].append(value)
                
        df = pd.DataFrame.from_dict(metadict, orient = 'columns')
        
        #return df
        pickle.dump(df, open(sub_series_file.split('.')[0] + '_slice_' + str(slice_number) + '_channel_' + str(channel) + '.dat', 'wb')) 




def findEventsParallel(traces, stds, std_threshold_pos=2.0, std_threshold_neg = 1.6, minimum_length=10, njobs=8):
    from joblib import Parallel, delayed
    import numpy as np
    frames, cells, trials = traces.shape    
    
        
    events = Parallel(n_jobs=njobs)(delayed(findEvents)(traces[:,cell,:], stds[cell,:], 
        std_threshold_pos, std_threshold_neg, minimum_length) for cell in range(cells))


    pos_list = [events[cell]['pos_events'] for cell in range(cells)]
    neg_list = [events[cell]['neg_events'] for cell in range(cells)]

    return dict([('pos_events', np.swapaxes(np.dstack(pos_list), 2,1)),('neg_events', np.swapaxes(np.dstack(neg_list), 2,1))])
    

def findEvents(traces, stds, std_threshold_pos=2.0, std_threshold_neg = 1.6, minimum_length=10):
    import mahotas
    
    """Core event finding routine with flexible syntax.


    :param: traces - 2 or 3d numpy array of baselined and normalized traces (time x cells, or time x cells x trials)
    :param: stds - 1 or 2d numpy event array of per-cell std values (cells, or cells x trials)
    :param: std_threshold - multiple of per-cell STD to use for an event (float)

    :param: minimum_length - minimum length of an event
    :param: alpha - optional scaling parameter for adjusting thresholds

    :returns: numpy array same shape and size of traces, with each event given a unique integer label. returns one for pos, 1 for neg. 
    """

    if traces.ndim == 2:
        traces = np.atleast_3d(traces) # time x cells x trials
        stds = np.atleast_2d(stds).T # cells x trials
    time, cells, trials = traces.shape
   
    pos_events = np.zeros_like(traces)
    neg_events = np.zeros_like(traces)

    
####################################################################################################################################
   
    
    # detect  events
    for trial in range(trials):
        for cell in range(cells):
            
            pos_events[:,cell,trial] = traces[:,cell,trial] > (stds[cell, trial] * float(std_threshold_pos)) # here we assume the mean is at 0.0 since we've already baselined. 
            
            neg_events[:,cell,trial] = traces[:,cell,trial] < (-1.0)*(stds[cell, trial] * float(std_threshold_neg))
    # filter for minimum length
    pos_events = mahotas.label(pos_events, np.array([1,1])[:,np.newaxis,np.newaxis])[0]
    neg_events = mahotas.label(neg_events, np.array([1,1])[:,np.newaxis,np.newaxis])[0]
    
    for single_event in range(1, pos_events.max()+1):
        if (pos_events == single_event).sum() <= minimum_length:
            pos_events[pos_events == single_event] = 0
    pos_events = pos_events>0
    
    for single_event in range(1, neg_events.max()+1):
        if (neg_events == single_event).sum() <= minimum_length:
            neg_events[neg_events == single_event] = 0
    neg_events = neg_events>0
    
    # finally label the event array and return it. 
    pos_events = np.squeeze(mahotas.label(pos_events>0, np.array([1,1])[:,np.newaxis,np.newaxis])[0])
    neg_events = np.squeeze(mahotas.label(neg_events>0, np.array([1,1])[:,np.newaxis,np.newaxis])[0])
    
    
    return dict([('pos_events',pos_events),('neg_events',neg_events)])


def get_normed_traces_allTrials(raw_rois,npils,npil_coefs,njobs,numTrials,subtracted=True):
    import numpy as np
    

    out = [get_normed_traces(raw_rois[...,trial], npils[...,trial], npil_coefs[...,trial],8, subtracted=subtracted) for trial in range(numTrials)] #list of dicts containing traces and stds
    return {'corrected_rois': np.swapaxes(np.asarray([out[trial]['corrected_rois'] for trial in range(numTrials)]).T,0,1),
            'normed_stds':np.asarray([out[trial]['normed_stds'] for trial in range(numTrials)]).T}

def get_normed_traces(rois, npils, coefs, njobs,subtracted):
    
    import sys
    import numpy as np
    from joblib import Parallel, delayed
    import cPickle as pickle
    from time import time
    
    """
    this function is designed for trial by trial runs.
    args: 
        rois, npils: [frames x cells] arrays for a field of view. 
        coefs: [cells] array; refers to output of get_ransac_npil_coefs
        njobs: specifies number of cores. 
        subtracted: if True, implements neuropil subtraction. 

    returns: 
    out = {'corrected_rois': corrected_rois,
        'normed_means': normed_means,
        'normed_stds': normed_stds}
    """
    
    #for every raw cell signal in this trial fit with gaussian mixture model. to get baseline estimate
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

def fitGaussianMixture1D_raw(data, n=2, set_mean_priors=True):
    from sklearn.mixture import GMM
    import numpy as np
    """Routine for fitting a 1d array to a mixture of `n` gaussians.
        if 'set_mean_priors' is True (the default), we initialize the GMM
        model with means equal to the first point (the 'background' cell)
        and all ROIs larger than the mean.  Otherwise, we have random means.
        After fitting, we return the means, stds, and weights of the GMM,
        along with the BIC, AIC, and the model itself.
        :param: data - 1d array of data to fit
        :param: n - number of gaussians to fit, defaults to 2
        :param: set_mean_priors - boolean, if true, initializes the means of a mixture of 2 gaussians
        :returns: tuple of (means, stds, weights, BIC, AIC, GMM model object)
    """
    a=data
    max_counts = max(np.histogram(a, bins = 100)[0]) #counts
    idx = np.argwhere(np.histogram(a, bins = 100)[0]==max_counts)
    mode = np.histogram(a, bins = 100)[1][idx][0][0]
    
    if set_mean_priors:
        g = GMM(n_components=n, init_params='wc', n_init=5)
        g.means_ = np.zeros((n, 1))
        g.means_[0,0] = mode
        g.means_[1,0] = data[data>mode].mean()
    else:
        g = GMM(n_components=n, n_init=5)
    
    g.fit(data)

    return (np.squeeze(g.means_.flatten()),
            np.squeeze(np.sqrt(g.covars_).flatten()),
            np.squeeze(g.weights_).flatten(),
            g.bic(data),
            g.aic(data),
            g.predict(data),
            g.predict_proba(data),
            g)


def fitGaussianMixture1D_normed(data, n=2, set_mean_priors=True):
    from sklearn.mixture import GMM
    import numpy as np

    
    if set_mean_priors:
        g = GMM(n_components=n, init_params='wc', n_init=5)
        g.means_ = np.zeros((n, 1))
        g.means_[0,0] = data.mean()
        g.means_[1,0] = data[data>data.mean()].mean()
    else:
        g = GMM(n_components=n, n_init=5)
    
    g.fit(data)

    return (np.squeeze(g.means_.flatten()),
            np.squeeze(np.sqrt(g.covars_).flatten()),
            np.squeeze(g.weights_).flatten(),
            g.bic(data),
            g.aic(data),
            g.predict(data),
            g.predict_proba(data),
            g)



def means_from_gmmOut(out_byCell):
    means, stds, weights, bic, aic, labels, priors, model = out_byCell
    return means.min()

def stds_from_gmmOut(out_byCell):
    means, stds, weights, bic, aic, labels, priors, model = out_byCell
    return stds.min() #fix this its possible that the wrong distribution will be returned. for quiet cells.



#################################################################################################################


def parseOdorsAllTrials(csv_order, csv_names, h5, raw_tif, frameAve, sigSampRate, trial, get_all = False):
    
    from glob import glob
    if get_all == False:
        #just do one trial; pass filename for one raw_tif
        return parseOdors(csv_order, csv_names, h5, raw_tif, frameAve, trial = trial)
    
    else:
        
        #get raw_tif directory
        raw_tif_dir = raw_tif
        files = glob(raw_tif_dir + '*.tif')
        if files == []:
            print 'bad dir'
        else:
            list_of_dicts = []
            for trial, raw_tif in enumerate(files):
                #here we're using only 1 h5 file. this is wrong. but doesn't matter for now since onnset is the same. for all trials. 
                # don't worry about offset for now. 
                list_of_dicts.append(parseOdors(csv_order, csv_names, h5, raw_tif, sigSampRate,frameAve, trial = trial))
            #here we pad all cell_odor-trial intervals to same shape by adding a frame to the end. #doesn't affect anything other than plotting using interval. 
            numOdors = list_of_dicts[0]['numOdors']
            for odor in range(1,numOdors+1):
                
                padVal = max((trial['pre_odor_post_interval'][odor][1]-trial['pre_odor_post_interval'][odor][0] for trial in list_of_dicts))
                for i, trial in enumerate(list_of_dicts):
                    dif = trial['pre_odor_post_interval'][odor][1]-trial['pre_odor_post_interval'][odor][0]
                    
                    if dif==padVal:
                        continue
                    appendVal = padVal-dif
                    list_of_dicts[i]['pre_odor_post_interval'][odor][1] = list_of_dicts[i]['pre_odor_post_interval'][odor][1]+appendVal
                        
    return list_of_dicts


#################################################################################################################
def parseOdors(csv_order, csv_names, h5, raw_tif, sigSampRate, frameAve, trial):
    import pandas as pn
    import h5py
    from spUtils import getFramePeriod
    import cPickle as pickle
    import numpy as np
    import traces as tm
    
    
    frameAve= float(frameAve)
    sampleRate =float(sigSampRate)
    trial = trial
    
    parseDict = {}
    secPerFrame = getFramePeriod(raw_tif)
    parseDict['secPerFrame'] = secPerFrame
    parseDict['trial'] = int(trial)+1
    
    #Read in files
    odor_signal = h5py.File(h5)
    odor_signal = odor_signal[str(odor_signal.keys()[1])][1]
    
    #truncate odor_signal so it matches original length of video file:
    #odor_signal = odor_signal[:int(7743*secPerFrame*sampleRate)]
    
    #get names and stimulus order
    odor_names = np.array(pn.read_csv(csv_names, header = None))
    order = np.array(pn.read_csv(csv_order, header = None))
    num_odors = odor_names.shape[0]
    #initialize dicts
    
    #return_these:
    frame_times = {'on':{}, 'off':{}}
    names_dict = {'odor_names':odor_names}
    order_dict = {'odor_order':order[:,trial]}
    interval = {'pre_odor_post_interval':{}}
    
    offset = np.around(15/secPerFrame/frameAve).astype('int')
    
    #find deflections
    derivative = np.gradient(odor_signal, 2)
    on_times  = tm.findLevels(derivative, 1000, mode='rising')[0]
    off_times  = tm.findLevels(derivative, -1000, mode='falling')[0]
    
    #
    on_times = np.round(on_times/sampleRate/secPerFrame/frameAve)
    off_times = np.round(off_times/sampleRate/secPerFrame/frameAve)
    
    #fill dict
    
    
    
    for frame, array in [('on', on_times), ('off', off_times)]:
        
        for odor in range(1,num_odors+1):
            index = np.argwhere(order[:,trial]==odor)[0]
            frame_times[frame][odor] = array[index][0].astype(int)

    #make interval for each odor

    for odor in range(1,num_odors+1):
        interval['pre_odor_post_interval'][odor] = [frame_times['on'][odor]-offset, frame_times['off'][odor]+offset]

    [parseDict.update(_) for _ in [frame_times, names_dict, order_dict, interval, {'numOdors':num_odors}, {'offset':offset}]]

    return parseDict
#################################################################################################################
#for each odor get reshaped array:
#requires traces_dict and trials to be in register
# could fix to just use trial id.

def reshape_by_odor(traces, list_of_trials, numTrials, odor):
    import numpy as np
    odor = odor
    frame_list_allTrials = []
    for trial in list_of_trials:
        frame_list_allTrials.append(trial['pre_odor_post_interval'][odor])
    
    #choose segment to plot
    all_trials_by_odor = []

    for trial in range(numTrials):
        a,b = frame_list_allTrials[trial]
        all_trials_by_odor.append(traces[a:b,:,trial])
    
    #reshape array appropriately.
    return np.dstack(all_trials_by_odor)

#################################################################################################################
def getFramePeriod(raw_tif):
    import tifffile
    
    page_dict = tifffile.TiffFile(raw_tif)[0].tags
    meta = page_dict['image_description'].value
    secPerFrame = [value for value in meta.split('\n') if 'scanimage.SI5.scanFramePeriod' in value]
    secPerFrame = float(secPerFrame[0].split(' = ')[1])

    return secPerFrame

def get_channel(fname, channel, offset = True):
    import tifffile
    if offset == True:
        with tifffile.TiffFile(fname) as tif:
            data = tif.asarray()[channel::2,...] # extract channel
            metadata = tif[0].tags['image_description'].value.split('\n')
            offset = [int(item) for item in metadata[35].split('[')[1].split() if item.isdigit()][channel]
            data += offset
            num_frames = data.shape[0]
        return (data, metadata, num_frames)
    else:
        with tifffile.TiffFile(fname) as tif:
            data = tif.asarray()[channel::2,...] # extract channel
            metadata = tif[0].tags['image_description'].value.split('\n')
            
            num_frames = data.shape[0]
            return (data, metadata, num_frames)




def find_nearest(array,value):
    import numpy as np
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def local_neighcorr(data, mask, corr_thresh, neigh_size, cell):
    import copy
    import numpy as np
    import scipy
    
    
    """
        Takes a temporally smoothed series and a mask (either binary or labelled) and performs a cross correlation for
        all pixels in local neighborhood aruond a cell's center.
        
        :param data: X by Y by time
        :param mask: 2-D labeled image of cell masks
        :param window:
        :param corr_thresh:
        :param neigh_size:
        :param cell:
        :returns:
        
        """
    
    
    #a: labelled mask, b:  number of labels.
    a,b = scipy.ndimage.label(mask)
    labeled_mask = mask
    binary_mask = mask.astype(bool).astype(int)
    
    
    #get coordinates for all but the first cell which corresponds to background.
    coords = np.vstack(scipy.ndimage.measurements.center_of_mass(mask, a, range(b+1)))
    
    #params
    window = 1
    corr_thresh = float(corr_thresh)
    size = neigh_size + window
    totalFrames = 333
    frameStart = 0
    numFrames = slice(frameStart,frameStart+totalFrames,1)
    
    numNeighbors = (window*2+1)**2-1
    neighborhood = (2*neigh_size + 2*window)
    
    #get center for the neuropil neighborhood.
    neigh_pos1 = coords[:][cell,0]
    neigh_pos2 = coords[:][cell,1]
    
    #get cell-specific local region of timeseries
    #binary_mask_long[neigh_pos1-size:neigh_pos1+size,
    #neigh_pos2-size:neigh_pos2+size, numFrames]
    
    
    
    #makke neighborhood for cell; mask ROIS that have already been selected
    #during segmentation; finally, crop masked arrray to particular neighborhood?
    # or should we mask? NEVER CHANGE SIZE JUST MASK>:
    #make a new mask; all 1's outside local neighborhood: use this mask
    # to remask the masked array.
    
    #broadcast mask to fit mov length.
    #make cell mask for fov.
    binary_mask_long = np.repeat(mask.astype(bool).astype(int)[:,:,np.newaxis], totalFrames,2)
    
    # make neigh mask for fov.
    neigh_mask = np.zeros_like(binary_mask_long).astype(bool)
    neigh_mask[neigh_pos1-size:neigh_pos1+size,
               neigh_pos2-size:neigh_pos2+size, numFrames]=1
        
    #check to see if local neighborhood is an exception case
    if neigh_mask[neigh_pos1-size:neigh_pos1+size,
                 neigh_pos2-size:neigh_pos2+size, numFrames].any()==False:
       
       print 'failed cell, empty neigh_mask', cell
       return cell

            ###
            #exception for failed cells goes here

            ###

    else:
        
        neigh_mask = np.invert(neigh_mask)
            
        #first mask entire fov with cell mask
        full_FOV_cell_masked = np.ma.array(data[:,:,:], mask = binary_mask_long)
        
        #then mask the masked FOV with neighborhood mask.
        local_FOV_neigh_and_cell_masked = np.ma.array(full_FOV_cell_masked, mask = neigh_mask)
        #local_FOV_neigh_and_cell_masked is now the same shape as data but masked
        #everywhere outside neighborhood; within neighborhood it's masked anywhere there
        #is a labelled roi.
        
        #now, for each pixel in this mask, average all pixels that is in the immedaite surround
        
        #gegnerate mask for center pixel.
        center_mask = np.zeros((window*2+1, window*2+1,1))
        center_mask[window,window,:] = 1
        center_mask = np.repeat(center_mask, local_FOV_neigh_and_cell_masked.shape[2], 2)
        
        
        # print center_mask.shape
        #just do for i and j in the neighborhood region
        corr_scores = np.empty((np.argwhere(neigh_mask[:,:,0]==0).shape[0], 3))
        try:
            for k, (a,b) in enumerate(np.argwhere(neigh_mask[:,:,0]==0)):
                
                
                lBound = a-window
                rBound = a+window+1
                tBound = b-window
                bBound = b+window+1
                
                
                
                mask_center = np.ma.array(local_FOV_neigh_and_cell_masked[lBound:rBound,tBound:bBound,:], mask = center_mask.astype(bool))
                mask_surround = np.ma.array(local_FOV_neigh_and_cell_masked[lBound:rBound,tBound:bBound,:], mask =
                                            np.invert(center_mask.astype(bool)))
                    
                neigh_ave = mask_center.mean(0).mean(0)
                this_pixel = mask_surround.mean(0).mean(0)
                
                corr_scores[k] = a,b, np.correlate(this_pixel, neigh_ave, mode = 'valid')
            
            #corr_scores = np.reshape(corr_scores, (neighborhood,neighborhood, 3))
            scores = corr_scores[:,2]
            norm_scores = scores/float(scores.max())
            
            thresh_scores = norm_scores.copy()
            thresh_scores[thresh_scores>corr_thresh] = 0
            
            
            #replace local neighborhood in original binary neighborhood mask (where 1s are neighborhood)
            #with thresh values. where high correlations are 0 and cell rois are 0
            #this leaves a 512 by 512 mask with only low correlates pixels labelled by 1s around the cell.
            substitution = np.invert(neigh_mask[:,:,0]).astype(float)
            # make exception for mismatched local neighs
            substitution[np.argwhere(substitution[:,:]==1)[:,0], np.argwhere(substitution[:,:]==1)[:,1]] = thresh_scores
        except:
            print 'failed cell, local mask mismatch', cell
            return cell
    return  (substitution.astype(bool))


###################################_Npil_Utils_##############################################################

def get_ransac_npil_coefs(npils,raw_rois, njobs=8):
    
    from joblib import Parallel, delayed
    from time import time
    import cPickle as pickle
    
    npils = npils
    raw_rois = raw_rois
    frames,cells,trials = raw_rois.shape
    cells_by_trial_list = []
    
    
    for trial in range(trials):
        
        tic = time()
        all_cells = Parallel(n_jobs = njobs)(delayed(fitRansac)(npils, raw_rois, cell = cell, trial = trial) for cell in range(cells))
        cells_by_trial_list.append(all_cells)

        print time()-tic
    return np.vstack(np.dstack(cells_by_trial_list))


def fitRansac(npils,raw_rois,cell,trial,min_samples = .10):
    
    from sklearn import linear_model
    import cPickle as pickle
    
    npils = npils
    raw_rois = raw_rois
    
    
    frames,cells,trials = raw_rois.shape
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), max_trials = 10000, min_samples = min_samples)
    X = npils[:,cell,trial, np.newaxis]
    y = raw_rois[:,cell,trial]
    model_ransac.fit(X,y)
    
    
    return model_ransac.estimator_.coef_[0][0]

def xticksInSec(framerate, secInterval, plotLengthInFrames):
    import numpy as np
    import matplotlib.pylab as plt
    plotLengthInFrames = int(plotLengthInFrames)
    secInterval = int(secInterval)
    xtickframetimes = np.arange(0, plotLengthInFrames, framerate*secInterval)
    xtickssectimes = np.arange(0, plotLengthInFrames/framerate, secInterval).astype(int)
    return plt.xticks(xtickframetimes,xtickssectimes)
    
def secScaleBar(framerate, seconds, startposX, ypos, textyoffset, textString, fontsize):
    from matplotlib import pyplot as plt
    framescale = framerate*seconds #sec
    startpos = startposX
    length = np.arange(startpos,startpos + framescale)
    scalebar = np.ones_like(length)*ypos
    s = textString
    fontsize  = fontsize
    plt.plot(length, scalebar)
    plt.annotate(xy = (startposX,ypos-textyoffset), s = '5 sec', fontsize = 14)

def getSNR(traces,maxEvents, cell,trial):
    import numpy as np
    premasked_traces = traces[:,cell, trial]
    eventMags = maxEvents[...,cell,trial]

    #Get absolute event mags.
    #mask events to get baseline. 
    mask = eventMags.astype(bool)

    masked_traces = premasked_traces[premasked_traces == mask]
    absoluteEventMags = np.ma.MaskedArray(premasked_traces, np.invert(mask))
    baseline = np.ma.MaskedArray(premasked_traces, mask)

    #get baseline and its SD
    blineMean = baseline.mean()
    sd = np.std(baseline)

    #subtract mean from absolute magnitude of each event to get signal. 

    signal = np.unique(eventMags)[1:]-blineMean
    noise = sd

    SNR = signal/noise
    return SNR

def kalman_filter_trace(traces, cell, trial):
    import numpy as np
    from pykalman import KalmanFilter as kfilt

    trace = traces[:,cell, trial]
    
    
    # create a Kalman Filter by hinting at the size of the state and observation
    # space.  If you already have good guesses for the initial parameters, put them
    # in here.  The Kalman Filter will try to learn the values of all variables.
    kf = kfilt(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=0.01 * np.eye(2))
    states_pred = kf.em(trace).smooth(trace)[0]
    filtered_trace = states_pred[:,0]
    return filtered_trace

def kalman_filter_trace_parallel(traces,njobs):
    import numpy as np
    from joblib import Parallel, delayed
    
    njobs = njobs
    
    frames,cells, trials = traces.shape
    kalman_filtered_traces_list = Parallel(n_jobs=njobs)(delayed(kalman_filter_trace)(traces, cell, trial) for cell in range(cells) for trial in range(trials))
    kalman_filtered_traces = np.array(kalman_filtered_traces_list).T
    kalman_filtered_traces = np.reshape(kalman_filtered_traces.copy(), (frames,cells, trials))
    return kalman_filtered_traces

###################################_Query_Utils_##############################################################

def gen_dict_extract(key, var):
    """
    returns value(s) of passed key regardless of where it is nested
    """
    if hasattr(var,'iteritems'):
        for k, v in var.iteritems():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result

def get_keys(dl, keys_list):
    if isinstance(dl, dict):
        keys_list += dl.keys()
        map(lambda x: get_keys(x, keys_list), dl.values())
    elif isinstance(dl, list):
        map(lambda x: get_keys(x, keys_list), dl)    

def open_folder(path):
    import subprocess
    subprocess.call(["open", "-R", path])                   