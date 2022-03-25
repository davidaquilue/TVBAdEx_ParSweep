import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as signal
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from numba import njit
from scipy.integrate import simps
from matplotlib.pyplot import cm


def remove_results(folder_sim):
    """Deletes all the folder_sim folder. Free disk space after loading results of each simulation"""
    try:
        shutil.rmtree(folder_sim)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (folder_sim, e))


def mean_FC(FR, do_plot=False):
    """Returns the average value of the FC matrix obtained from Pearson correlation of timetraces in FR."""
    FC = np.corrcoef(np.transpose(FR))
    FC_mean = np.mean(FC) - np.trace(FC) / FC.size  # Delete the relevance of the diagonal elements.
    if do_plot:
        plt.imshow(FC)
        plt.title('FC matrix')
        plt.show()
        plt.close()
    return FC_mean


def mean_PLI(FR, do_plot=False):
    nnodes = FR.shape[1]
    sig = np.transpose(FR)
    PLI = np.zeros((nnodes, nnodes))
    hilb_amplitude_region = np.zeros_like(sig)
    hilb_phase_region = np.zeros_like(sig)

    for reg in range(nnodes):
        hilb = signal.hilbert(sig[reg])
        hilb_amplitude_region[reg] = np.abs(hilb)
        hilb_phase_region[reg] = np.angle(hilb)

    for i_reg in range(nnodes):
        for j_reg in range(i_reg, nnodes):
            phase_lags = hilb_phase_region[i_reg] - hilb_phase_region[j_reg]
            PLI[i_reg][j_reg] = np.abs(np.mean(np.sign(phase_lags)))
            PLI[j_reg][i_reg] = PLI[i_reg][j_reg]

    PLI_mean = np.mean(PLI) - np.trace(PLI) / PLI.size  # Not interested about the contribution of diagonal elements
    if do_plot:
        plt.imshow(PLI)
        plt.title('PLI matrix')
        plt.show()
        plt.close()
    return PLI_mean


# ========================================= UP-STATES ========================================= #
def detect_UP(train_cut, ratioThreshold=0.4,
              sampling_rate=1., len_state=50.,
              gauss_width_ratio=10.):
    """
    detect UP states from time signal
    (population rate or population spikes or cell voltage trace)
    return start and ends of states.
    
    Written by Trang-Anh Nghiem

    Parameters
    ----------
    train_cut: array
        array of shape (N, ) containing the time trace on which we detect upstates

    ratioThreshold: float
        Over which % of the FR maximum value in the time trace of a region we consider an up-state

    sampling_rate: float
        Sampling rate of the time trace. Usually 1 / dt. In ms**(-1)

    len_state: float
        Minimum length (in ms) of time over threshold to be considered up-state (I think)

    gauss_width_ratio: float
        Width ratio of the Gaussian Kernel used in the filter for detecting up-states.

    Returns
    -------
    idx: array
        indexes where there is a change of state.
    train_shift: array
        time trace of the filtered signal - ratioThreshold * np.max(train_filtered)
    train_bool: array
        array containing 1s when up state and 0s when downstate
    """
    # convolve with Gaussian
    time = range(len(train_cut))  # indexes
    gauss_width = gauss_width_ratio * sampling_rate

    # We obtain a gauss filter
    gauss_filter = np.exp(-0.5 * ((np.subtract(time, len(train_cut) / 2.0) / gauss_width) ** 2))
    gauss_norm = np.sqrt(2 * np.pi * gauss_width ** 2)
    gauss_filter = gauss_filter / gauss_norm

    # We filter the signal by convolving the gauss_filter
    train_filtered = signal.fftconvolve(train_cut, gauss_filter)
    train_filt = train_filtered[int(len(train_cut) / 2.0): \
                                int(3 * len(train_cut) / 2.0)]
    thresh = ratioThreshold * np.max(train_filt)

    # times at which filtered signal crosses threshold
    train_shift = np.subtract(train_filt, thresh)
    idx = np.where(np.multiply(train_shift[1:], train_shift[:-1]) < 0)[0]

    # train of 0 in DOWN state and 1 in UP state
    train_bool = np.zeros(len(train_shift))
    train_bool[train_shift > 0] = 1  # assign to 1 in UP states

    # cut states shorter than min length
    idx = np.concatenate(([0], idx, [len(train_filt)]))
    diff_remove = np.where(np.diff(idx) < len_state * sampling_rate)[0]
    idx_start_remove = idx[diff_remove]
    idx_end_remove = idx[np.add(diff_remove, 1)] + 1

    for ii_start, ii_end in zip(idx_start_remove, idx_end_remove):
        train_bool[ii_start:ii_end] = np.ones_like(train_bool[ii_start:ii_end]) * train_bool[ii_start - 1]
        # assign to same state as previous long

    idx = np.where(np.diff(train_bool) != 0)[0]
    idx = np.concatenate(([0], idx, [len(train_filt)])) / sampling_rate
    return idx, train_shift, train_bool


@njit
def obtain_updown_durs(train_bool, dt):
    N = train_bool.size
    up_durs = np.empty(0)  # Something of the like up_durs = []
    down_durs = np.empty(0)
    current_up_duration = 0
    current_down_duration = 0
    for i in range(1, N):  # We sweep over all the values of the train_bool signal
        if train_bool[i - 1] == train_bool[i]:  # If 2 consecutive equal values -> increase state duration
            if train_bool[i - 1] == 1:
                current_up_duration += dt
            else:
                current_down_duration += dt
        else:  # If 2 consecutive NOT equal values -> increase state duration + store duration + restore
            if train_bool[i - 1] == 1:
                up_durs = np.append(up_durs, current_up_duration)
                current_up_duration = 0
            else:
                down_durs = np.append(down_durs, current_down_duration)
                current_down_duration = 0
        if i == N - 1:  # Regardless of the value of the last time point, we have to store the last duration.
            if train_bool[i] == 1:
                current_up_duration += dt
                up_durs = np.append(up_durs, current_up_duration)
                current_up_duration = 0
            else:
                current_down_duration += dt
                down_durs = np.append(down_durs, current_down_duration)
                current_down_duration = 0

    if up_durs.size == 0:  # If no up-states, return duration of 0
        mean_up_durs = 0
    else:
        mean_up_durs = np.mean(up_durs)

    if down_durs.size == 0:  # If no down-states, return duration of 0
        mean_down_durs = 0
    else:
        mean_down_durs = np.mean(down_durs)

    return mean_up_durs, mean_down_durs


def mean_UD_duration(FR, dt, ratio_threshold=0.3,
                     len_state=20,
                     gauss_width_ratio=10,
                     units='s'):
    """ Returns mean duration of Up and Down states in the TVB-AdEx simulation using Trang-Anh's detect_UP function.

    Parameters
    ----------
    FR: ndarray
        numpy array of shape (N, M) containing the FR time traces of M regions of the TVB-AdEx simulation.

    dt: float
        time step used to integrate the TVB-AdEx (in ms).

    ratio_threshold: float
        Over which % of the FR maximum value in the timetrace of a region we consider an up-state

    len_state: float
        Minimum length (in ms) of time over threshold to be considered up-state (I think)

    gauss_width_ratio: float
        Width ratio of the Gaussian Kernel used in the filter for detecting up-states.

    units: str
        Units for the results. Either 'ms' or 's'.

    Returns
    -------
    up_mean_len: float
        Average length of up-states throughout the whole simulation, in s.
    down_mean_len: float
        Average length of up-states throughout the whole simulation, in s.
    """
    # TODO: Accurately determine the best set of parameters for the TVB adex

    N, M = FR.shape
    sampling_rate = 1 / dt
    up_mean_len = 0
    down_mean_len = 0
    for m in range(M):  # Sweep over regions
        _, _, train_bool = detect_UP(FR[:, m], ratioThreshold=ratio_threshold,
                                     sampling_rate=sampling_rate,
                                     len_state=len_state,
                                     gauss_width_ratio=gauss_width_ratio)

        mean_up_m, mean_down_m = obtain_updown_durs(train_bool, dt)
        # We update the running value that will be transformed in mean of means at the end
        up_mean_len += mean_up_m
        down_mean_len += mean_down_m

    if units == 'ms':
        up_mean_len = up_mean_len / M
        down_mean_len = down_mean_len / M
    elif units == 's':
        up_mean_len = up_mean_len / (1000 * M)  # Since dt is in ms
        down_mean_len = down_mean_len / (1000 * M)
    else:
        raise ValueError('Choose units between s or ms')
    return up_mean_len, down_mean_len


def psd_fmax_ampmax(frq, psd, type_mean='avg_psd', prominence=1, which_peak='both'):
    """ Returns frequency at which PSD peaks and the amplitude of the peak. Makes use of scipy.signal.find_peaks.

    Parameters
    ----------
    frq: array_like
        Vector of frequencies, array of shape (N, )

    psd: array_like
        Array containing the PSD (or PSDs). If single PSD, array of shape (N, ). If multiple (M)
        PSDs, array of shape (M, N). So, in each row a different PSD

    type_mean: str
        For multiple PSDs.
        'avg_psd': For looking for the max of the average of all the PSDs
        'avg_slopes': For looking for the f_max_i and amp_max_i of each of the M PSD and obtaining mean(f_max_i) and
        mean(amp_max_i).

    prominence: Understand how prominences work and note it down here. Give a valuable range as well

    which_peak: str
        To choose which peak we want the function to return:
        - 'max_prominence': Returns the peak with the maximum prominence
        - 'max_amplitude': Returns the peak with the maximum power value
        - 'both': Returns both peaks in a list

    Returns
    -------
    f_max: list
        Frequency at which we find the peak of the PSD. 2 elements if 'both'
    amp_max: list
        Amplitude of the chosen peak of the PSD. 2 elements if 'both'
    """
    good_idxs = frq > 0
    frq = frq[good_idxs]

    if not len(psd.shape) == 1 and type_mean == 'avg_psd':  # 2D PSDs and avg_psd, obtain the average of PSD
        psd = np.mean(psd, axis=0)
        psd = psd[good_idxs]

    if len(psd.shape) == 1 or type_mean == 'avg_psd':  # Now PSD is only a vector
        idx_peaks, props = signal.find_peaks(psd, prominence=prominence)
        # TODO: Find how to obtain a reliable prominence for our simulations
        if idx_peaks.size == 0:  # It might be that the peak finding algorithm does not find any
            if which_peak == 'max_amplitude' or which_peak == 'max_prominence':
                return [np.nan], [np.nan]
            elif which_peak == 'both':
                return [np.nan, np.nan], [np.nan, np.nan]
            else:
                raise ValueError('Choose a correct value of which_peak')
        else:
            # We take the peak with maximum prominence:
            idx_max_peak_prom = idx_peaks[np.argmax(props['prominences'])]
            amp_max_prom = psd[idx_max_peak_prom]

            # Or we take the peak that has the highest PSD value:
            idx_max_peak = idx_peaks[np.argmax(psd[idx_peaks])]
            amp_max = psd[idx_max_peak]

            if which_peak == 'max_prominence':
                return [frq[idx_max_peak_prom]], [amp_max_prom]
            elif which_peak == 'max_amplitude':
                return [frq[idx_max_peak]], [amp_max]
            elif which_peak == 'both':
                return [frq[idx_max_peak], frq[idx_max_peak_prom]], [amp_max, amp_max_prom]
            else:
                raise ValueError('Choose a correct value of which_peak')

    elif not len(psd.shape) == 1 and type_mean == 'avg_slopes':  # We want PSD to be 2D array to obtain means of each
        f_max_m = np.array([])
        amp_max_m = np.array([])
        f_max_m_prom = np.array([])
        amp_max_m_prom = np.array([])

        psd = psd[:, good_idxs]
        M, N = psd.shape
        flag = False
        for m in range(M):
            aux_psd = psd[m]
            idx_peaks, props = signal.find_peaks(aux_psd, prominence=1)

            if idx_peaks.size == 0:  # Avoid errors if the algorithm cannot find any peaks
                flag = True
                break

            idx_max_peak = idx_peaks[np.argmax(psd[idx_peaks])]
            idx_max_peak_prom = idx_peaks[np.argmax(props['prominences'])]

            f_max_m = np.append(f_max_m, frq[idx_max_peak])
            f_max_m_prom = np.append(f_max_m_prom, frq[idx_max_peak_prom])

            amp_max_m = np.append(amp_max_m, psd[idx_max_peak])
            amp_max_m_prom = np.append(amp_max_m_prom, psd[idx_max_peak_prom])

        if flag:
            return [np.nan], [np.nan]
        else:
            if which_peak == 'max_amplitude':
                return [np.mean(f_max_m)], [np.mean(amp_max_m)]
            elif which_peak == 'max_prominence':
                return [np.mean(f_max_m_prom)], [np.mean(amp_max_m_prom)]
            elif which_peak == 'both':
                return [np.mean(f_max_m), np.mean(f_max_m_prom)], [np.mean(amp_max_m), np.mean(amp_max_m_prom)]
            else:
                raise ValueError('Choose a correct value of which_peak')

    else:
        print('Check size of array if compatible with type of average. No computation done.')
        return [np.nan], [np.nan]


def fit_psd_slope(frq, psd, range_fit=(0.1, 1000), type_mean='avg_psd'):
    """ Fits PSD to b/f^a which corresponds to a linear relationship in log(PSD) vs log(frq).
    
    Parameters
    ----------
    frq: array_like
        Vector of frequencies, array of shape (N, )
    
    psd: array_like
        Array containing the PSD (or PSDs). If single PSD, array of shape (N, ). If multiple (M)
        PSDs, array of shape (M, N). So, in each row a different PSD

    range_fit: tuple
        Sets the range of frequencies that will be used to fit the params.
        
    type_mean: str
        For multiple PSDs.
        'avg_psd': For obtaining the fit on the average of all the PSDs
        'avg_slopes': For fitting each of the M PSD to b/f^a and then obtaining mean(a) and mean(b).
        
    Returns
    -------
    a: float
        Slope obtained after fitting.
    b: float
        Intercept obtained after fitting
    """
    f_0 = frq[frq > range_fit[0]]
    f_f = f_0[f_0 < range_fit[1]]
    
    if not len(psd.shape) == 1 and type_mean == 'avg_psd':
        psd = np.mean(psd, axis=0)
        
    if len(psd.shape) == 1 or type_mean == 'avg_psd':  # Now both have same dimensions
        psd_0 = psd[frq > range_fit[0]]
        psd_f = psd_0[f_0 < range_fit[1]]
        X = np.array([np.log(f_f)[0::10]]).T  # We take every 10 values to speed up the fitting
        Y = np.array([np.log(psd_f)[0::10]]).T

        reg = LinearRegression().fit(X, Y)
        a = -reg.coef_[0, 0]
        b_prime = reg.intercept_
        b = np.exp(b_prime)
        score = reg.score(X, Y)

        return a, b, score
        
    elif not len(psd.shape) == 1 and type_mean == 'avg_slopes':
        a_m = np.array([])
        b_m = np.array([])
        score_m = np.array([])
        M, N = psd.shape
        for m in range(M):
            # This could be optimized but don't want to waste much time
            aux_psd = psd[m]
            psd_0 = aux_psd[frq > range_fit[0]]
            psd_f = psd_0[f_0 < range_fit[1]]
            X = np.array([np.log(f_f)[0::10]]).T  # We take every 10 values to speed up the fitting
            Y = np.array([np.log(psd_f)[0::10]]).T

            reg = LinearRegression().fit(X, Y)
            a_m = np.append(a_m, -reg.coef_[0, 0])
            b_prime = reg.intercept_
            b_m = np.append(b_m, np.exp(b_prime))
            score_m = np.append(score_m, reg.score(X, Y))
        a = np.mean(a_m)
        b = np.mean(b_m)
        score = np.mean(score_m)

        return a, b, score
        
    else:
        print('Check size of array if compatible with type of average. No computation done.')
        return 0, 0, 0


# ========================================= Plotting ========================================= #
def PSD(time_s, FR, returnplot=False, ax=None, c=None, lab=None):
    """Function that takes the time and FR of a TVB simulation and returns the psd analysis with the plot
    Inputs:
    - time_s:     Time vector resulting from the TVB simulation
    - FR:         Firing rate array (Exc or Inh) resulting from the TVB simulation
    - returnplot: Boolean that to either modify axis or just perform analysis
    If returnplot = True:
    - ax:         Axes on which we want to plot the psd
    - c:          Color to plot the PSD
    - lab:        Label for the plot

    Outputs:
    analysis: Tuple containing the signals from the psd analysis
    ax: Returns the modified axes. Only if returnplot=True.

    """
    no = type(None)
    if (returnplot and type(ax) == no) or (returnplot and type(c) == no) or (returnplot and type(lab) == no):
        raise ValueError('If returnplot, then provide axes')
    f_sampling = 1. * len(time_s) / time_s[-1]  # time in seconds, f_sampling in Hz
    frq = np.fft.fftfreq(len(time_s), 1 / f_sampling)
    pwr_region = []

    sig = np.transpose(FR)
    nnodes = len(sig)
    for reg in range(nnodes):
        pwr_region.append(np.abs(np.fft.fft(sig[reg])) ** 2)

    mean_Hz = np.mean(pwr_region, axis=0)
    std = np.std(pwr_region, axis=0)
    high = mean_Hz[frq > 0] + std[frq > 0] / np.sqrt(nnodes)
    low = mean_Hz[frq > 0] - std[frq > 0] / np.sqrt(nnodes)

    # ylims
    ymax = np.amax(mean_Hz) * 100
    ymin = np.amin(mean_Hz) / 100
    if returnplot:
        ax.loglog(frq[frq > 0], mean_Hz[frq > 0], color=c, label=lab)
        ax.fill_between(frq[frq > 0], high, low, color=c, alpha=0.4)
        ax.set(xlabel='Frequency(Hz)', ylabel='Power', xlim=(0.1, 100), ylim=(ymin, ymax))
        analysis = (mean_Hz, std, high, low)
        return analysis, ax

    else:
        analysis = (mean_Hz, std, high, low)
        return analysis


# ========================================= Power in Bands. First tests ========================================= #
def rel_power_bands(frq, psd, bands=None, do_plot=False):
    """ Returns the relative powers in each band range with the associated plot.

    Modified classical version from https://raphaelvallat.com/bandpower.html
    Accuracy could be improved by using multitaper.

    Function could be generalized to take in multiple PSDs and perform averaging of relative power.

    Parameters
    ----------
    frq: array_like
        Vector of frequencies, array of shape (N, )

    psd: array_like
        Array containing the PSD, of shape (N, )

    bands: dict
        Dictionary containing the label and low, high limits of frequency bands, e.g. {'alpha': (8, 12)}

    do_plot: bool
        Determines whether a plot with shaded areas for each band should be returned.

    Returns
    -------
    rel_powers: dict
        containing the relative power for each of the bands, e.g. {'alpha': 0.5}
    """

    if bands is None:  # Default bands value, to not have a mutable argument
        bands = {'theta': (0.5, 4), 'delta': (4, 8)}
    good_idxs = frq > 0  # Only interested in real part of the PSD
    frq = frq[good_idxs]
    psd = psd[good_idxs]

    rel_powers = {}
    freq_res = frq[1] - frq[0]
    total_power = simps(psd, dx=freq_res)

    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set(xlim=(0.4, 200), xlabel='Frequency (Hz)', ylabel='Power spectral density')
        color = iter(cm.rainbow(np.linspace(0, 1, len(bands))))
        ax.loglog(frq, psd, 'k')

    for i, band in enumerate(bands):
        low, high = bands[band]
        idx_band = np.logical_and(frq >= low, frq <= high)
        power_band = simps(psd[idx_band], dx=freq_res)
        rel_powers[band] = power_band / total_power

        if do_plot:
            c = next(color)
            plt.fill_between(frq, psd, where=idx_band, color=c, alpha=0.5, label=band)
    if do_plot:
        plt.title('Relative power for bands')
        plt.legend()
        plt.show()
        plt.close()
    return rel_powers


def corr_dmn(fr, dmn_regions=None):
    """ Returns a data frame containing the statistics of the correlation values between pairs of nodes:
    - Correlations between nodes of the Default Mode Network (DMN)
    - Correlations between nodes of the DMN and other nodes
    - Correlations between nodes not of the DMN

    The last one being, usually, the largest group of the three.

    Parameters
    ----------
    fr: ndarray
        Numpy array of shape (N, M) containing the firing rates of M regions.

    dmn_regions: list
        Contains the indexes of the M regions that belong to the Default Mode Network.


    Returns
    -------
    data: pandas.DataFrame
        Contains M(M-1) rows with the values of correlation and to which group each value belongs.
    """

    if dmn_regions is None:
        dmn_regions = [28, 29, 52, 53,  # mPFC
                       50, 51, 20, 21]  # precuneus / posterior cingulate
    corrs_in_DMN = []
    corrs_out_DMN = []
    corrs_in_out_DMN = []

    # We make use of the FR_exc obtained
    N, M = fr.shape
    FC = np.corrcoef(fr.T)

    for i in range(M):
        # for j in range(i): # If we want to take only one triangular portion of the matrix.
        for j in range(M):
            if i == j:  # Not interested in diagonal
                continue
            corr = FC[i, j]
            if i in dmn_regions and j in dmn_regions:
                corrs_in_DMN.append(corr)
            elif any(n in dmn_regions for n in (i, j)):
                corrs_in_out_DMN.append(corr)
            else:
                corrs_out_DMN.append(corr)

    # Saving results in a dataframe
    df_in_DMN = pd.DataFrame({'group': np.repeat('DMN', len(corrs_in_DMN)),
                              'corr': corrs_in_DMN})

    df_in_out_DMN = pd.DataFrame({'group': np.repeat('DMN with others', len(corrs_in_out_DMN)),
                                  'corr': corrs_in_out_DMN})

    df_out_DMN = pd.DataFrame({'group': np.repeat('others', len(corrs_out_DMN)),
                               'corr': corrs_out_DMN})

    data = pd.concat([df_in_DMN, df_in_out_DMN, df_out_DMN])

    return data


def plot_corr_dmn(df_corrs_dmn, type_plot='box', jitter=True):
    """ Shows box or violin plot of the statistics of correlation between nodes in the TVB.
    Separates in three groups:
    - Correlations between nodes of the Default Mode Network (DMN)
    - Correlations between nodes of the DMN and other nodes
    - Correlations between nodes not of the DMN

    The last one being, usually, the largest group of the three.

    Parameters
    ----------
    df_corrs_dmn: pandas.DataFrame
        Containing M rows. Each row will have a correlation value and the label of the cathegory of the value.

    type_plot: str
        Either 'box' or 'violin'.

    jitter: bool
        Determines whether we want to plot all the data points.

    Returns
    -------
    fig: matplotlib.pyplot fig object
    ax: matplotlib.pyplot axes object
        Shows statistics of correlations between the different nodes in the network.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set(ylim=(-1.1, 1.1))
    if type_plot == 'box':
        sns.boxplot(x='group', y='corr', data=df_corrs_dmn, ax=ax)

    elif type_plot == 'violin':
        sns.violinplot(x='group', y='corr', data=df_corrs_dmn, ax=ax)
    else:
        raise ValueError('Incorrect choice of type_plot (either box or violin)')

    if jitter:
        sns.stripplot(x='group', y='corr', data=df_corrs_dmn, color="orange", jitter=0.2, size=2, ax=ax)

    return fig, ax
