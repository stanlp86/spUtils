
#QC Helper Functions
""" 
Most of our code for processing imaging data has largely been  written to operate on dictionaries and arrays. 
Because tabular (pandas) data manipulation and storage is convenient on so many levels, here we provide convenience Functions
for either converting between 2 formats, or directly operating on tabular data using our codebase. 

Ultimately, the goal is to move away from our codebase, and port everything to pandas.

"""
from spUtils import get_normalized_traces_submodule 
import numpy as np
import pandas as pd
import graphlab as gl
gl.canvas.set_target('ipynb')



get_normed_traces_arglist = dict(window = 112, 
                            SD_window = 38,
                            SD_percentile = 5,
                            Fluor_percentile = 5,
                            njobs = 8,
                            method = 2,
                            subtracted=False
                            )



def normTrial(group, offset):
    from scipy.stats.mstats import mquantiles
    
    offset = offset# temporary fix. 
    
    trialID = group['trial'].ix[0]
    
    #get trial as array
    trial = np.squeeze(group.loc[trialID][['0']].to_panel().as_matrix().T)
    
    #preprocess trial. 
    #clip alignment vals. but def. get rid of offset fix. this should havppen during initial trace extraction. 
    trial = trial - offset #temporary fix.
    min_, max_ = mquantiles(trial.flatten(),  prob=[0.01, 100])
    trial[trial<min_]=min_
    
    numCells = trial.shape[1]
    raw_rois = trial
    npils = trial
    npil_coefs = trial[0,:]

    

    singleTrial = get_normalized_traces_submodule.get_normed_traces_byTrial(raw_rois, npils, npil_coefs, **get_normed_traces_arglist)
    
    #reindex this trial for compatability
    iterables = [[trialID], range(numCells)]
    idx = pd.MultiIndex.from_product(iterables, names=['trial', 'cellID'])
    
    #reformat output and add back to df
    baselined1_traces = pd.DataFrame(singleTrial['baselined1_traces'].T, index = idx).stack()
    corrected_rois = pd.DataFrame(singleTrial['corrected_rois'].T, index = idx).stack()
    normed_stds = pd.DataFrame(singleTrial['normed_stds'].T, index = idx)
    normed_stds.columns = ['normed_stds']
    #place in df
    group = group.join(normed_stds.loc[trialID])
    group['normed_traces']=corrected_rois
    group['baselined1_traces']=baselined1_traces
    
    return group

def saveDF(directory, epochIDX, current_epoch):
    store = pd.HDFStore(directory + 'epoch_{}_DF.h5'.format(epochIDX))
    store['epoch_{}'.format(epochIDX)] = current_epoch
    store.close()

def trialDFs_to_SFrame(directory, saveName = 'experiment'):
    files = glob(directory + '*.dat')
    slice0 = gl.SFrame(pickle.load(open(files[0])))
    epoch = int(re.split('tracesDF_e|_0', files[0])[-2])
    slice0['epoch'] = epoch

    for fname in files[1:]:
        df = gl.SFrame(pickle.load(open(fname)))
        epoch = int(re.split('tracesDF_e|_0', fname)[-2])
        df['epoch'] = epoch

        slice0 = slice0.append(df)
        #trial = re.split('tracesDF_|_.dat', fname)[-2]#[int(s) for s in re.split('tracesDF |_',fname) if s.isdigit()][0]

    slice0.save(directory + saveName)


#Convert odor information from pandas to dict. 

def extractOdorInfo(sframePath):
    def changeTimingReference(x):
        trialstart = x['trialFrame'][0]
        odorstart = x['odorOn'][0]
        odorstop = x['odorOff'][0]
        x['trialOdorOn']=trialstart+odorstart
        x['trialOdorOff'] = trialstart + odorstop
        return x

    #Filter odor relevant information from master sframe and read into pandas dataframe
    odorInfo = sframe[sframe['cellID']==0]['odorPos', 'odorID', 'odorOn', 'odorOff', 'trial', 'epoch']
    odorInfo = odorInfo.to_dataframe()
    #get rid of sframe index, reindex by trial and corresponding frames. 
    groups = odorInfo.groupby('trial').apply(lambda x: x.reset_index(drop = True))
    #turn index of Trial frames into its own column
    groups = groups.reset_index(1)
    #Regroup by odorID and trial. and reset_index. This resorts trialFrames such that the frames corresponding to each 
    #odor presentation are referenced with respect to that trial; not just the odor presentation. 
    #the new index is 0-241 reflecting all frames for that presentation. 
    groups = groups.groupby(['odorID', 'trial']).apply(lambda x: x.reset_index(drop = True))
    groups.rename(columns={'level_1': 'trialFrame'}, inplace=True)
    odorInfo = groups.groupby(['odorID', 'trial']).apply(lambda x: changeTimingReference(x))
    return odorInfo

def odorInfo_to_dict(odorInfo, trialID):
    """
    Utility function for reformatting tabular odor info into dict format supplied to qc functions. 
    Operates on a trial by trial basis.

    args: odor info (dataframe)
    output: 
    -trialID (int)
    -num_odors (int)
    -odor-order (list)
    -odor on (dict, odorID's are keys)
    -odor off (dict, odorID's are keys)
    -pre_odor_interval_post (dict, odorID's are keys; values are: list of 2)

    """
    
    num_odors = odorInfo.odorID.max()
    num_trials = odorInfo.trial.max()

    parseDict ={}
    #keys:

    parseDict['trialID'] = trialID
    frame_times = {'on':{}, 'off':{}}
    interval = {'pre_odor_post_interval':{}}
    order_dict = {'odor_order':[]}

    for odor in range(1,num_odors+1):
    #odorPeriod timing
        fullLength = len(odorInfo.xs((1,1)).trialFrame)
        startFrame = odorInfo.xs((odor,trialID)).trialFrame[0]
        stopFrame = odorInfo.xs((odor,trialID)).trialFrame[fullLength-1]
        interval['pre_odor_post_interval'][odor] = [startFrame, stopFrame]

        #odor onset timing
        onFrame = odorInfo.xs((odor,trialID)).trialOdorOn[0]
        offFrame = odorInfo.xs((odor,trialID)).trialOdorOff[0]
        frame_times['on'][odor], frame_times['off'][odor] = onFrame, offFrame

    #get odor order for this trial
    trial_order = []
    for odorPos in range(1,num_odors+1):
        trial_order.append(odorInfo[(odorInfo['trial']==trialID) & (odorInfo['odorPos']==odorPos)]['odorID'].ix[0])
    order_dict = {'odor_order':trial_order}
    #update odor Dict
    temp = [parseDict.update(_) for _ in [frame_times, order_dict, interval, {'numOdors':num_odors}]]
    return parseDict

#generate mask
