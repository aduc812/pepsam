####################################################################################
# This file defines a set of filtering routines
# to be used in pemongo package
#
####################################################################################



def filter_wrapper(y,set_of_filters):
    '''\
a routine which does all the dirty filtering

arguments:

    set_of_filters --  a tuple of definitions of filters to be applied to the data 
        filter definition is a dictionaty with one element: 
            its key is a string of filter type 
                for example 'adjavg' , 'fourier' , 'golay', 'FIR' or 'cosmic'
            its value is a filter parameter or a tuple of filter parameters (if several required)
 
'''
# support single dict input
    if isinstance(set_of_filters,dict):
         set_of_filters=tuple((set_of_filters,))
    for filter_ in set_of_filters:
        if  'adjavg' in filter_:
            min_period=filter_['adjavg']
            y=adjavg_lowpass_filter(y, min_period=min_period)
        elif 'fourier' in filter_:
            y=fft_lowpass_filter(y, min_period=filter_['fourier']) 
        elif 'cosmic' in filter_:
            y=cosmic_filter (y,tolerance=filter_['cosmic'])
        elif 'golay' in filter_: 
            y=savitzky_golay(y, window_size=filter_['golay'][0], order=filter_['golay'][1])
        elif 'FIR' in filter_:
            #(cutoff,width,attenuation,s_rate)
            try:
                try:
                    s_rate=filter_['FIR'][3]           
                except IndexError:
                    s_rate=100.0
                try:
                    att=filter_['FIR'][2]           
                except IndexError:
                    att=60.0
                try:
                    width=filter_['FIR'][1]           
                except IndexError:
                    width=5.0
                cutoff=filter_['FIR'][0]
            except TypeError:
                s_rate=100.0
                width=5.0
                att=60.0
                cutoff=filter_['FIR']
                 
            y=FIR(y, sample_rate_hz=s_rate, cutoff_hz=cutoff, width_hz=width, attenuation_db=att)  
        else:
            raise ValueError('No such filter:'+filter_.items()[0][0])
    return y

####################################################################################
# This is a FFT lowpass filtering routine by v_2e from askSAGE
#
####################################################################################
def fft_lowpass_filter(signal, min_period=2):
    """Applies a simple as a log FFT lowpass filter to a signal.

    INPUT:

    - ``signal`` -- a list of data representing the sampled signal;

    - ``min_period`` -- a threshold for the lowpass filter (the minimal period
      of the oscillations which should be left intact) expressed in a number of 
      samples per one full oscillation.

    EXAMPLES:

    If you have a sampling frequency equal to 1 Hz (one sample per second)
    and want to filter out all the frequencies higher than 0.5 Hz (two samples 
    per one oscillation period) you should call this function like this::

        sage: fft_lowpass_filter(signal, min_period=2)

    If you, for example, have a sampling frequency of 1 kHz (1000 samples per 
    second) and wish to filter out all frequencies hihger than 50 Hz, you should
    use the value of ``min_period`` = <sampling frequency> / <cutoff frequency> = 1000 Hz / 50 Hz = 20::

        sage: fft_lowpass_filter(signal, min_period=20)

    """
    import scipy

    signal_fft = scipy.fft(signal)
    spectrum_array_length = len(signal_fft)

    i = 0                                          # This is used instead of ' for i in range()'
    while i < spectrum_array_length:               # because the 'signal' array can be rather long
        if i >= spectrum_array_length/min_period : 
            signal_fft[i] = 0
        i += 1

    signal_filtered = scipy.ifft(signal_fft)

    fft_filtered_signal = []

    filtered_signal_length = len(signal_filtered)
    i = 0
    while i < filtered_signal_length:
        fft_filtered_signal.append(float(signal_filtered[i]))
        i += 1

    norm = max(signal)/max(fft_filtered_signal)              # In case we need 
    
    
                                                             # to renormalize 
    fft_filtered_signal_length = len(fft_filtered_signal)    # the filtered signal
    i = 0                                                    # in order to obtain
    while i < fft_filtered_signal_length:                    # the same amplitude
        fft_filtered_signal[i] *= norm                       # as the inintial
        i += 1                                               # signal had.

    return fft_filtered_signal


# End of fft lowpass filter
####################################################################################



####################################################################################
# adjacent-averaging lowpass filter

def adjavg_lowpass_filter(signal, min_period=2):
    """Applies a simple as a log FFT lowpass filter to a signal.

    INPUT:

    - ``signal`` -- a list of data representing the sampled signal;

    - ``min_period`` -- a threshold for the lowpass filter (the minimal period
      of the oscillations which should be left intact) expressed in a number of 
      samples per one full oscillation."""
    import numpy as np
    
    signal_filt=signal
    for data_index in range (0,len(signal)):
        window=np.ones(min_period)
        for window_index in range (0,min_period):
            dwi=data_index-min_period//2+window_index # data in-window index
            if dwi<0:
                window[window_index]=signal[0]
            else: 
                if dwi>=len(signal):
                    window[window_index]=signal[-1]
                else:
                    window[window_index]=signal[dwi]
        signal_filt[data_index]= float(sum(window)) / len(window)

    return signal_filt
        

# End of adjacent-averaging lowpass filter
####################################################################################
 
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
# end of Savitsky-Golay data filter 
####################################################################################

#------------------------------------------------
#  FIR filter
#------------------------------------------------
def FIR(y, sample_rate_hz=1.0, cutoff_hz=1.0, width_hz=1.0, attenuation_db=60.0):
    '''\
finite impulse response (FIR) filter
Agruments: 
    y - an numpy array of data to be filtered
Keywords:
    sample_rate_hz - sample rate in hertz. defaults to 1.0
    cutoff_hz - cutoff frequency in Hz. defaults to 1.0
    width_hz - width of transition from pass to stop, in hertz.
         defaults to 1.0
    attenuation_db - The desired attenuation in the stop band, in dB. 
         defaults to 60.0
'''
    from scipy.signal import kaiserord, lfilter, firwin, freqz
    from numpy import array
    # The Nyquist rate of the signal.
    nyq_rate = sample_rate_hz/2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = width_hz/nyq_rate

    # The desired attenuation in the stop band, in dB.
    #attenuation_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(attenuation_db, width)

    # The cutoff frequency of the filter.
    #cutoff_hz = 10.0

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))


    # try to preserve units
    try: 
        yunits=y.units
    except AttributeError:
        yunits=None
    # Use lfilter to filter y with the FIR filter.

    filtered_y = lfilter(taps, 1.0, y)
    #corrected_y=filtered_y
    
    # Correct phase delay (in units of sample rate)
    delay = 0.5 * (N-1) 
    
    corrected_y=filtered_y.copy()
    for i,val in enumerate(filtered_y):
        corrected_y[i-delay]=filtered_y[i]
    
    if yunits is not None:
        return array(corrected_y)*yunits
    else:
        return array(corrected_y)


###################################################################################
# Cosmic ray removal filter
# searches for separate outliers in data, which are away from both neighbours 
# more than tolerance*(average difference between neighboring points)
# and sets them to average of neighbors

#if ((abs(DifftoNext[i-1]*tolerance)>avgdiff)&&(abs(DifftoNext[i]*tolerance)>avgdiff)&&(DifftoNext[i-1]*DifftoNext[i]<0))

def cosmic_filter (data,tolerance=5):
    '''#
Cosmic ray removal filter
searches for separate outliers in data, which are away from both neighbours 
more than tolerance*(average difference between neighboring points)
and sets them to average of neighbors.
The bigger is the tolerance, the less data are affected.
Makes no sence to make it lower than tolerance 1
'''
    import numpy as np
    
# correct default tolerance
    if tolerance<=0:
        tolertance=5.0
    else:
        tolerance=float(tolerance)
# calculate average distance between closest neighbours
    distances=[]
    for i in range(0,len(data)-1):
        distances.append(data[i]-data[i+1])
    avgdist=np.sum(np.abs(distances))/float(len(distances))
# Find separate outliers
    newdata=np.array(data)
    for i in range(0,len(data)-2):
        if (np.abs(distances[i])>avgdist*tolerance) and (np.abs(distances[i+1])>avgdist*tolerance) and (np.abs(distances[i]+distances[i+1]) <= (np.abs(distances[i])+np.abs(distances[i+1])/2)):
            newdata[i+1]=(newdata[i]+newdata[i+2])/2
    return newdata


# end of Cosmic ray removal filter
####################################################################################

