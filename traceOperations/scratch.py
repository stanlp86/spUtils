
import scipy as sp
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import seaborn as sns



__all__ = ['readme','get_baselined_trace', 'get_params', 'compare_params', 'get_allParams_oneTrace', 
            'storeValsParamsTrialsbyCell', 'retrieve_param_oneTrace', 'retrieveParamforallCells', 
            'plotParamAve']

def readme():

    """
    The only routine that actually carries out the comparison is: 
    compare_params. It uses get_baselined_trace and get_param_difs to execute normalization. 
    The rest are utility routines: 
    storeValsParamsTrialsbyCell adapts get_allParams_oneTrace which uses compare_params to store all metric values for all parameter settings as a dict where keys are cells and values are list of lists. 

    To retrieve data, 
    RetrieveParamforallCells uses retrieve_param_oneTrace
    the output is plotted using plotParamAve


    """

def get_baselined_trace(trace, order, cutoff, plots = False):
    b, a = sp.signal.butter(order, cutoff, 'low',analog=False)
    lf = sp.signal.lfilter(b, a, trace)

    baseline_mask = lf.copy()
    #cutoff all values above median of baseline trace. 
    baseline_mask[baseline_mask>sp.median(baseline_mask)]=sp.median(baseline_mask)
    
    normed_trace = trace/baseline_mask-1
########################   PLOT   ################################################
    if plots:
    
        plt.figure(figsize = (20,5))

        plt.plot(baseline_mask, linewidth = 1, color = 'k', label = 'baseline estimate')
        plt.plot(trace, linewidth = 0.4, label= 'raw trace')
        plt.axhline(trace.mean(), label = 'mean', linewidth = 0.4)
        #plt.ylim(500,4000)
        #plt.xlim(xleft,xright)
        plt.legend()

        #Normalized trace
        plt.figure(figsize = (20,5))

        plt.plot(normed_trace, linewidth = 0.4, label = 'normalized trace')
        plt.axhline(trace.mean())
        plt.axhline(0, linewidth = 0.4)
        plt.ylim(-1,5)
        #plt.xlim(xleft,xright)
        plt.legend()
        return normed_trace
    else:
        return normed_trace

def standardize(trace):
    
    standardized_trace = (trace-trace.mean())/sp.std(trace)
    return standardized_trace

################################################################################################

def get_params(trace, numBins=50, offset=0.1):


    "Params: trace, numBins=50, offset=0.1 \n Returns: FWHM, base_dist, neg_dist, pos_dist, peak, mean, var, skew, kurt " 
    
    import scipy as sp
    import traces as tm
    from scipy import stats
    from scipy import linspace
    from scipy.interpolate import UnivariateSpline
    from scipy.stats import describe

       
    #manually correct filtered artifact. 
    offset = trace.shape[0]*offset
    #get min_max
    n_samples, min_max, mean, var, skew, kurt = stats.describe(trace[offset:])
    

    n, bins = sp.histogram(trace[offset:], 50)

    l = linspace(min_max[0], min_max[1], numBins*2)
    fit = UnivariateSpline(bins[:-1],n)

    thresh = n.max()/2
    curve = fit(l)

    tophalf = curve.copy()
    tophalf[tophalf<thresh]=0


    up = tm.findLevels(tophalf, 0, mode='rising')[0]
    down = tm.findLevels(tophalf, 0, mode='falling')[0]
    peak = n.max()
    FWHM = sp.linalg.norm(l[down]-l[up])
    base_dist = min_max[1]-min_max[0]
    neg_dist = abs(min_max[0])
    pos_dist = abs(min_max[1])
    return FWHM, base_dist, neg_dist, pos_dist, peak, mean, var, skew, kurt  


########################################################################################

def compare_params(trace, F_order=1, window=0.1):
    """Params: raw trace for 1 cell-trial, F_order, and window for butterfilter 
    Returns: differences for FWHM, base_dist, neg_dist, pos_dist, peak, mean, var, skew, kurt
    between control_trace and normed_trace
      """
    control_trace = trace/trace.mean()-1
    normed_trace = get_baselined_trace(trace, F_order,window)
    normed_trace = normed_trace-normed_trace.mean()
    a = get_params(normed_trace, numBins=50, offset=0.1)
    b = get_params(control_trace, numBins=50, offset=0.1)
    all_param_difs = [sp.linalg.norm(a[i]-b[i])/abs(b[i]) for i in range(len(a))]
    return all_param_difs

########################################################################################
def get_allParams_oneTrace(trace_array, cell, trial, window_vals, F_order_vals):
    """
    Generator.
    RETURNS: list of length identical to number of values provided for parameter search
    
    """
    F_order = F_order_vals
    trace = trace_array[:,cell,trial]
    window_vals = window_vals
   
    for window in window_vals:
        yield compare_params(trace, F_order, window=window)
    
########################################################################################
def storeValsParamsTrialsbyCell(trace_array,numCells, numTrials, window_vals, F_order_vals):
    
    """
    This returns a dict with key - cell, and value - 
    array of shape (number of search values x numParams x numTrials)
    """
    paramsDict = defaultdict(list)

    for cell in range(numCells):
        trial_list =  []
        for trial in range(numTrials):
            trial_list.append([a for a in get_allParams_oneTrace(trace_array, cell, trial, window_vals=window_vals, F_order_vals = F_order_vals)])
            paramsDict[cell] = sp.dstack(trial_list)
    return paramsDict

########################################################################################
def retrieve_param_oneTrace(paramsDict, cell, trial, param):
    """ for cell-trial pair, returns one param. 
        PARAMS: above. 
        RETURNS: difference between control and normed trace for a given parameter
    """
    vals_pars_array = paramsDict[cell][:,:,trial]
    
    params = {0:'FWHM', 1:'base_dist', 2:'neg_dist', 3:'pos_dist', 4:'peak', 5:'mean', 6:'var', 7:'skew', 8:'kurt'}    
    params = {key:value for value,key in params.items()}
    param = params[param]
    
    return vals_pars_array[:,param]

########################################################################################
def retrieveParamforallCells(paramsDict, param, numCells, trial):
    cell_list =  []
    for cell in range(numCells):
        cell_list.append([a for a in retrieve_param_oneTrace(paramsDict, cell, trial, param)])
        cells_array = sp.squeeze(sp.dstack(cell_list))
    return cells_array 

########################################################################################
def plotParamAve(cells_array, label):
    
    y = cells_array.mean(1)
    ystd = cells_array.std(1)
    x = range(y.shape[0])
    l = sp.linspace(0.001, .75, 8)
    
    plt.plot(l,y, linewidth = 0.5, label = label)
    plt.errorbar(l,y, yerr=ystd, linewidth = 0.2)
    plt.legend()





