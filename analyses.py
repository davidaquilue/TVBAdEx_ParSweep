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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.tri import Triangulation
import scipy.signal as signal
from scipy.optimize import curve_fit


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
              gauss_width_ratio=10., min_for_up=0.2):
    """
    detect UP states from time signal
    (population rate or population spikes or cell voltage trace)
    return start and ends of states.
    
    Written by Trang-Anh Nghiem. Modified with min_for_up by David Aquilue

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

    min_for_up: float
        A value under which there is no up-state. That way, if we have high relative variations
        near 0 value but the FR is not higher than 0.3 there will be no up-state.
        However, take into account that this will modify the functioning of the algorithm, possibly
        underestimating up state duration.

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
    train_shift = np.subtract(train_filt, thresh) - min_for_up
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


# ========================================= PSD Metrics ========================================= #
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


# ========================================= COUNTING AMOUNT OF UD States ========================================= #
def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)


def exponential(x, A, k):
    return A * np.exp(-k * x)


def sing_bi_modal_fit(time_trace, do_plot=False):
    """This function returns either 0 if U-D or 1 if AI state. Will return 0 as well if broken point.
    That way we will count the number of U-D states in one simulation by summing all the values.

    When histogram is bimodal: U-D, value 0
    When histogram is single mode: AI, value 1

    The process this algorithm does is:
        Tries to fit the histogram of the input time-trace to a Gaussian curve
        Then tries to fit the histogram of time-trace to a sum of two Guassians
        If any of the two fits cannot be done, the state will be assigned to the only possible fit
        If both fits can be done, then the fit with better R2 score on the data is selected as state.

    However, if no fit is possible, then it will return a np.nan value.

    Parameters
    -------
    time_trace: ndarray
        Array containing the FR time trace of a single region.

    do_plot: bool
        Select if one wants to obtain a plot or not.

    Returns
    -------
    val: 0, 1 or np.nan
        Depending on if the time-trace provided is U-D, AI state or no possible fit was possible to apply.
    """
    if np.max(time_trace) > 120:
        return 0
    else:
        range_bins = (0, 100)
        n_bins = 101

        hist, bins = np.histogram(time_trace, bins=n_bins, range=range_bins)
        mean_hist = np.mean(hist)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        sing_failed = False
        expected_sing = (10, 1, 5000)
        try:
            params_sing, _ = curve_fit(gauss, center, hist, expected_sing)
        except:
            sing_failed = True
        else:
            params_sing, _ = curve_fit(gauss, center, hist, expected_sing)
            SS_res = np.sum((hist - gauss(center, *params_sing)) ** 2)
            SS_tot = np.sum((hist - mean_hist))
            R_2_sing = 1 - SS_res / SS_tot

        bi_failed = False
        expected_bi = (4, 1, 5000, 15, 1, 2000)
        try:
            params_bi, _ = curve_fit(bimodal, center, hist, expected_bi)
        except:
            bi_failed = True
        else:
            params_bi, _ = curve_fit(bimodal, center, hist, expected_bi)
            SS_res = np.sum((hist - bimodal(center, *params_bi)) ** 2)
            SS_tot = np.sum((hist - mean_hist))
            R_2_bi = 1 - SS_res / SS_tot
        x_fit = np.linspace(center[0], center[1], center.size)

        if do_plot:
            plt.bar(center, hist, align='center', width=width)
            if not sing_failed:
                plt.plot(center, gauss(center, *params_sing), color='red', lw=3)
            if not bi_failed:
                plt.plot(center, bimodal(center, *params_bi), color='green', lw=3)
            plt.show()
            plt.close()

        # Now it's time to handle the results of the fits
        if bi_failed and sing_failed:
            # If both of them failed, we try one last resort. Some U-D histogram's look like exponential
            all_failed = False
            try:
                params_exp, _ = curve_fit(exponential, center, hist, (10000, 100))
            except:
                # If the fit to the exponential does not work. We return a nan as total failure of fit
                return np.nan
            else:
                return 0  # We have U-D case

        elif bi_failed and not sing_failed:
            return 1  # Single mode didn't fail, return AI state

        elif sing_failed and not bi_failed:
            return 0  # Bimodal didn't fail, return U-D state

        else:
            # If neither of them failed, see which R_2 score is better
            if R_2_bi > R_2_sing:
                return 0
            else:
                return 1


def sing_bi_modal_peaks(time_trace, do_plot=False):
    """This function returns either 0 if U_D or 1 if AI state. Will return 0 as well if broken point.
    That way we will count the number of U-D states in one simulation by summing all the values.

    When histogram is bimodal: U-D, value 0
    When histogram is single mode: AI, value 1

    The process this algorithm does is:
        Find the peaks in the histogram
        Take the two peaks with maximum prominence
        If one of the peaks appears @ less than 5Hz, then UD state
        Else AI

    Parameters
    -------
    time_trace: ndarray
        Array containing the FR time trace of a single region.

    do_plot: bool
        Select if one wants to obtain a plot or not.

    Returns
    -------
    val: 0, 1
        Depending on if the time-trace provided is U-D or AI state.
    """
    if np.max(time_trace) > 120:
        return 1
    else:
        range_bins = (0, 100)
        n_bins = 101

        hist, bins = np.histogram(time_trace, bins=n_bins, range=range_bins)
        mean_hist = np.mean(hist)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        idx_peaks, props = signal.find_peaks(np.concatenate(([min(hist)], hist, [min(hist)])), prominence=100)
        idx_peaks -= 1
        if idx_peaks.size > 1:
            idx_peaks = idx_peaks[np.argpartition(props['prominences'], -2)[-2:]]

        if do_plot:
            plt.bar(center, hist, align='center', width=width)
            plt.plot(idx_peaks, hist[idx_peaks], 'r*')
            plt.show()
            plt.close()

        if np.min(idx_peaks) <= 5:  # Discuss this value
            return 0
        else:
            return 1


def count_ratio_AI(FR, type_alg='peaks', do_plot=False):
    running_count = 0
    for fr_idx in range(68):
        time_trace = FR[:, fr_idx]
        if type_alg == 'peaks':
            running_count += sing_bi_modal_peaks(time_trace, do_plot=do_plot)
        elif type_alg == 'fits':
            running_count += sing_bi_modal_fit(time_trace, do_plot=do_plot)
        else:
            raise ValueError('Select an adequate type_alg. Either peaks or fits')
    return running_count / 68


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


# ======================================== Power in Bands ======================================== #
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


# ======================================== DMN and FC ======================================== #
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
        for j in range(i): # If we want to take only one triangular portion of the matrix.
        #for j in range(M):
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


def plot_corr_dmn(df_corrs_dmn, type_plot='violin', jitter=False):
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


def multiview_z_score(cortex, hemisphere_left, hemisphere_right, idx_regions_without_seed,
                      z_score, zlim, seed_region=None, title='', figsize=(15, 10), **kwds):
    """Function that will plot the parcellation of the brain color coded with the z_score values.
    In theory, one will have a seed region with index = seed_region that will have been used to obtain
    the correlations and z-scores of all the other regions. This index seed_region will be the only region
    index not included in the region array.

    It can also be used to obtain colormaps over the brain's surface if correctly adapted. In that case,
    input idx_regions_without seed should contain all nodes and seed_region=None should work.

    Parameters
    ----------
    cortex: tvb.datatypes.cortex.Cortex object
        Cortex object. This and both hemispheres can be readily obtained with the following line of code:
        cortex, conn, hem_left, hem_right = prepare_surface_regions_human(inputs)

    hemisphere_left: np.ndarray
        Hemisphere object obtained as previously mentioned

    hemisphere_right: np.ndarray
        Hemisphere object obtained as previously mentioned

    idx_regions_without_seed: np.ndarray or list
        Contains the indexes of all the regions (except the seed region) for which we have obtained
        a z_score. Typically, 67 elements in our parcellation.

    z_score: np.ndarray or list
        Contains the z-scores of all the regions. Usually 68 elements, although one of them (corresponding
        to seed region) might have extreme values, even infinite value. It will be masked.

    zlim: float or tuple
        If float the range of the colormap will be (-zlim, zlim) if tuple range will be (zlim[0], zlim[1])

    seed_region: int
        Index of the region that has been used as seed to obtain the z-scores.

    title: str
        If we want to give a title to the figure

    Returns
    -------
    fig: matplotlib.pyplot.figure object
        Contains the five graphs of the brain color graded using the z-score.
    """
    fig = plt.figure(figsize=figsize)

    # Obtaining the triangulation from the cortex object
    cs = cortex
    vtx = cs.vertices  # Gives the three vertices of the many triangles in the parcellation
    tri = cs.triangles
    rm = cs.region_mapping
    x, y, z = vtx.T

    data = np.zeros((cs.region_mapping_data.array_data.shape[0],))  # data will be used for coloring
    lh_tri = tri[np.unique(np.concatenate([np.where(rm[tri] == i)[0] for i in hemisphere_left]))]
    lh_vtx = vtx[np.concatenate([np.where(rm == i)[0] for i in hemisphere_left])]
    lh_x, lh_y, lh_z = lh_vtx.T
    lh_tx, lh_ty, lh_tz = vtx[lh_tri].mean(axis=1).T
    rh_tri = tri[np.unique(np.concatenate([np.where(rm[tri] == i)[0] for i in hemisphere_right]))]
    rh_vtx = vtx[np.concatenate([np.where(rm == i)[0] for i in hemisphere_right])]
    rh_x, rh_y, rh_z = rh_vtx.T
    rh_tx, rh_ty, rh_tz = vtx[rh_tri].mean(axis=1).T
    tx, ty, tz = vtx[tri].mean(axis=1).T

    if type(idx_regions_without_seed) == list or type(idx_regions_without_seed) == np.ndarray:
        for r in idx_regions_without_seed:
            data[rm == r] = z_score[r]  # Set the z-score value for all except seed region. For coloring

    # Assign np.nan to seed region. Will be masked for different color
    if seed_region:
        data[rm == seed_region] = np.nan
        data = np.ma.masked_values(data, np.nan)

    # We define the views we want to see
    views = {
        'lh-lateral': Triangulation(-x, z, lh_tri[np.argsort(lh_ty)[::-1]]),
        'lh-medial': Triangulation(x, z, lh_tri[np.argsort(lh_ty)]),
        'rh-medial': Triangulation(-x, z, rh_tri[np.argsort(rh_ty)[::-1]]),
        'rh-lateral': Triangulation(x, z, rh_tri[np.argsort(rh_ty)]),
        'both-superior': Triangulation(y, x, tri[np.argsort(tz)]),
    }

    def plotview(i, j, k, viewkey, z=None, zlim=None, shaded=False, viewlabel=False):
        """Function that plots one of the views."""
        cmap = plt.cm.get_cmap("RdBu_r").copy()
        cmap.set_bad('green')
        v = views[viewkey]
        small_ax = plt.subplot(i, j, k)
        if z is None:
            raise ValueError("No z-score array")
        if not viewlabel:
            small_ax.axis('off')
        kwargs = {'shading': 'gouraud'} if shaded else {'edgecolors': 'k', 'linewidth': 0.13}
        tc = small_ax.tripcolor(v, z, cmap=cmap, **kwargs)
        if type(zlim) is tuple:
            tc.set_clim(vmin=zlim[0], vmax=zlim[1])
        else:
            tc.set_clim(vmin=-zlim, vmax=zlim)
        small_ax.set_aspect('equal')

        return small_ax, tc

    plotview(2, 3, 1, 'lh-lateral', data, zlim=zlim, **kwds)
    plotview(2, 3, 4, 'lh-medial', data, zlim=zlim, **kwds)
    plotview(2, 3, 3, 'rh-lateral', data, zlim=zlim, **kwds)
    plotview(2, 3, 6, 'rh-medial', data, zlim=zlim, **kwds)
    ax, im = plotview(1, 3, 2, 'both-superior', data, zlim=zlim, **kwds)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    if title:
        plt.gcf().suptitle(title, y=0.85, fontsize=20)
    fig.tight_layout()
    return fig


def ratio_most_active_from_dmn(fr, dmn_regions=None, k=10):
    """Function that selects the k regions with highest mean FR and checks how many DMN regions are in
    the top 10."""

    if dmn_regions is None:
        dmn_regions = [28, 29, 52, 53,  # mPFC
                       50, 51, 20, 21]  # precuneus / posterior cingulate

    mean_t_FR = np.mean(fr, axis=0)  # (M, ) of mean FR of each region

    k_most_active = np.argpartition(mean_t_FR, -k)[-k:]
    count_in_dmn = 0
    for active_region in k_most_active:
        if active_region in dmn_regions:
            count_in_dmn += 1

    ratio_most_active = count_in_dmn / k

    return ratio_most_active


def ratio_zscore_from_dmn(FC, seed=50, dmn_regions=None, k=10):
    """Function that selects the k regions more correlated with the seed region and checks how many DMN
    regions are in that top 10. Minimum ratio of 0.1 since the seed region will have largest zscore."""

    if dmn_regions is None:
        dmn_regions = [28, 29, 52, 53,  # mPFC
                       50, 51, 20, 21]  # precuneus / posterior cingulate

    corrs_with_seed = FC[:, seed]
    zscores = np.arctanh(corrs_with_seed)
    k_more_correlated = np.argpartition(zscores, -k)[-k:]

    count_in_dmn = 0
    for active_region in k_more_correlated:
        if active_region in dmn_regions:
            count_in_dmn += 1

    ratio_zscore = count_in_dmn / k

    return ratio_zscore


def hist_FR_dmn(fr, dmn_regions=None):
    """ Returns two average histogram of FRs: one for the nodes in the DMN and the other for the nodes outside
    the DMN.

    Parameters
    ----------
    fr: ndarray
        Numpy array of shape (N, M) containing the firing rates of M regions.

    dmn_regions: list
        Contains the indexes of the M regions that belong to the Default Mode Network.


    Returns
    -------
    hist_DMN: ndarray
        Averaged (and normalized) histogram of FRs of the nodes in the DMN

    hist_others: ndarray
        Averaged (and normalized) histogram of FRs of the nodes NOT in the DMN

    bins: ndarray
        Array containing the histogram bins' edges.
    """

    if dmn_regions is None:
        dmn_regions = [28, 29, 52, 53,  # mPFC
                       50, 51, 20, 21]  # precuneus / posterior cingulate

    if np.amax(fr) > 120:  # If broken point, larger range for histogram
        range_bins = (0, 200)
    else:
        range_bins = (0, 80)

    bins_edges = np.arange(range_bins[0], range_bins[1] + 1)

    N, M = fr.shape

    hist_DMN = np.zeros(bins_edges.size - 1)
    hist_others = np.zeros_like(hist_DMN)

    for node in range(M):
        hist_node, _ = np.histogram(fr[:, node], bins=bins_edges)
        if node in dmn_regions:
            hist_DMN += hist_node / np.sum(hist_node)
        else:
            hist_others += hist_node / np.sum(hist_node)

    hist_DMN /= len(dmn_regions)
    hist_others /= M - len(dmn_regions)

    return hist_DMN, hist_others, bins_edges


def plot_violin_hist_FR(hist_DMN, hist_others, bins_edges, jitter=False, bw_violin=0.3):
    """Plot in violin form the results from the function hist_FR_dmn"""
    center = (bins_edges[:-1] + bins_edges[1:]) / 2

    factor = 10000  # Actually quite fast. Takes around 90ms with this factor.
    inted_hist_DMN = np.array([int(factor * i) for i in hist_DMN])
    inted_hist_others = np.array([int(factor * i) for i in hist_others])

    inted_hists = [inted_hist_DMN, inted_hist_others]
    labels = ['DMN', 'others']
    dfs = []
    for id_hist, inted_hist in enumerate(inted_hists):
        data = []
        for idx_in_hist, number_elements in enumerate(inted_hist):
            for ii in range(number_elements):
                data.append({'group': labels[id_hist], 'FR': center[idx_in_hist]})

        dfs.append(pd.DataFrame(data))

    final_data = pd.concat(dfs)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.violinplot(x='group', y='FR', data=final_data, ax=ax, bw=bw_violin)

    if bins_edges[-1] > 150:
        ax.set(ylim=(-2, 200))
    else:
        ax.set(ylim=(-2, 60))

    # We maintain the problems of the histogram. We will only have points in the centers of the bins.
    # This problem can be seen when using jitter
    if jitter:
        sns.stripplot(x='group', y='FR', data=final_data, color="orange", jitter=0.2, size=2, ax=ax)
    return fig, ax
