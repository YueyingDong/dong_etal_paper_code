import warnings

# cv for image processing
import gc 
import cv2

# math packages
import math, pywt
import random

# importing packages for data analysis
import pandas as pd
import numpy as np
import pingouin as pg
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

#plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap


#for readin
import glob 
import os   
import pickle
from datetime import datetime


#for signal processing
from scipy.interpolate import CubicSpline,interp1d
import scipy.signal as signal
import scipy.stats as stats
from scipy.signal import filtfilt, butter
from scipy.fft import rfft, rfftfreq
from scipy.stats import sem 
from scipy.stats import norm
from scipy.stats import ttest_rel
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
from scipy.io import loadmat

#other packages
import ast
import itertools
from itertools import groupby
from more_itertools import consecutive_groups
import more_itertools as mit
from operator import itemgetter


#some shared settings
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'arial'


#dataframe functions
def subset(df, subset_dict={'col1': ['val1', 'val2'],
                             'col2': ['val1', 'val2']}):
    
    for col, vals in subset_dict.items():
        df = df[df[col].isin(vals)]

    return df

def cohenD(group1,group2):
    # Calculate means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    # Calculate standard deviations
    std1 = np.std(group1, ddof=1)  # Sample standard deviation
    std2 = np.std(group2, ddof=1)

    # Calculate sample sizes
    n1 = len(group1)
    n2 = len(group2)

    # Calculate pooled standard deviation
    pooledstd = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    cohend = (mean1 - mean2) / pooledstd
    return cohend

#some key custom functions

def consecutive(data, stepsize=10,
                findSame = False): # helper function for findGazeShift, deBlink, and more
    """
    @ data = row vector
    @ stepsize =  the larger the step size, the more difference it allows for two groups
                to be considered within the same cluster
    @ findSame = if True, find concecutive data point of the same value, this is set true
                 for finding blink counts using the blink mask (a df of 0 and 1 )
    """
    # this will split an array of numbers into many subarrays of consecutive numbers
    # the stepsize controls for how close of the two numbers for them to be considered
    # in the same cluster; e.g. [1,2,3,4,5,10] would be clustered together if stepsize = 5


    #data is the array of timepoint where the saccade velocity array crossed the threshold
    if findSame:
        stepsize = 0
        return np.split(data, np.where(abs(np.diff(data)) > stepsize)[0]+1)
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)


def smooth(x,window_len=11,window='hanning'): #smoothing function
    try:
        x = x.values
    except:
        pass
    # about np windows:
    # https://numpy.org/doc/stable/reference/routines.window.html
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    #this is to prepare the input for convolve i.e. add 1/2 window length padding before and after
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]] 
    
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    #sliding window
    y=np.convolve(w/w.sum(),s,mode='valid')
    
    #get rid of the paddings
    return y[int(window_len/2):int(-0.5*window_len)] #select the data points to get rid of the delay


def fs(width,height):#setting plot size
    plt.rcParams['figure.figsize'] = (width,height)

# A function to prettify my figures, make spines thicker, etc
def style_spines(axes, which_spines='all', color='#666666', width=3):
    try:
        for a in np.ravel(axes):
            spines = a.spines.values() if which_spines == 'all' else [a.spines[s] for s in which_spines]
            for spine in spines:
                spine.set_color(color)
                spine.set_linewidth(width)
    except KeyError:
        raise KeyError("which_spines should be a 1. 'all', or 2. a list containing 'left', 'bottom', 'right', 'top'")
        