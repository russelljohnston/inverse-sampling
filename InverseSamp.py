############################################################
#Created by Russell Johnston
#rwi.johnston@gmail.com
#
#Uses the porbability integral transform to sample the distribution
#of any input data from its CDF.
#https://en.wikipedia.org/wiki/Inverse_transform_sampling
#
#requires as input:
# data:  a dataset in form of 1D array you want to Monte Carlo
# nsamp: number of random samples
#
# num_bins: can be adjusted to suit how finely you bin the
#           histogram of your data
###########################################################
import numpy as np
from scipy.interpolate import interp1d


def InverseSamp(data,nsamp,num_bins):
    #create a histogram of your data
#    num_bins = 80

    counts, bin_edges = np.histogram(data, bins=num_bins, normed=False)
    #returns the total in each bin (counts) and the bin location (bin_edges)

    #now create the cumulative distribtion fucntion of this
    cdf = np.cumsum(counts)
    #normalise it between 0 and 1
    cdf[0]=0.
    cdf = cdf/float(cdf[num_bins-1])

    #interpolate the cdf and the bin location
    interp = interp1d(cdf,bin_edges[1:])

    #now you can randomly sample the CDF between 0 and 1 uniformly
    #to obtain your randomly sampled distribution
    
    dran =  np.random.uniform(0,1, nsamp)
    

    samp = interp(dran)

    return samp