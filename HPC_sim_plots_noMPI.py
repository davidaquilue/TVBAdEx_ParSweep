import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tvb_model_reference.src.nuu_tools_simulation_human as tools

from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
from analyses import *

parameters = Parameter()
start_time = time.time()

steps = 2  # Let's say 15 at the moment.
S_vals = np.linspace(0, 0.5, steps)
b_vals = np.linspace(0, 120, steps)
E_L_e_vals = np.linspace(-80, -60, steps)
E_L_i_vals = np.linspace(-80, -60, steps)
T_vals = np.linspace(5, 40, steps)

S_vals = list(S_vals)
b_vals = list(b_vals)
E_L_i_vals = list(E_L_i_vals)
E_L_e_vals = list(E_L_e_vals)
T_vals = list(T_vals)

lst = [S_vals, b_vals, E_L_i_vals, E_L_e_vals, T_vals]

combinaison = list(itertools.product(*lst))

del lst
del S_vals
del E_L_e_vals
del E_L_i_vals
del T_vals
del b_vals

Job_proc = np.array(combinaison)

# Set the parameters of the simulation:
run_sim = 5000.0  # ms, length of the simulation
cut_transient = 2000.0  # ms, length of the discarded initial segment
Iext = 0.000315  # External input

# Define a location to save the files wll make sure they already created.
folder_root = './results/'

for simnum in range(len(Job_proc)):
    parameters.parameter_coupling['parameter']['a'] = Job_proc[simnum][0]
    parameters.parameter_model['b_e'] = Job_proc[simnum][1]
    parameters.parameter_model['E_L_i'] = Job_proc[simnum][2]
    parameters.parameter_model['E_L_e'] = Job_proc[simnum][3]
    parameters.parameter_model['T'] = Job_proc[simnum][4]

    parameters.parameter_model['external_input_ex_ex'] = Iext
    parameters.parameter_model['external_input_in_ex'] = Iext

    label_sim = '_a_' + str(Job_proc[simnum][0]) + '_b_' + str(Job_proc[simnum][1]) + '_ELI_' + \
                str(Job_proc[simnum][2]) + '_ELE_' + str(Job_proc[simnum][3]) + '_T_' + str(Job_proc[simnum][4]) + '/'
    print(label_sim)
    file_name = folder_root + label_sim
    parameters.parameter_simulation['path_result'] = file_name

    result = tools.get_result(file_name, cut_transient, run_sim)
    time_s = result[0][0] * 1e-3  # from ms to sec

    # Decide which one to analyze since they usually exhibit very similar behavior
    FR_exc = result[0][1][:, 0, :] * 1e3  # from KHz to Hz; Excitatory firing rate
    FR_inh = result[0][1][:, 1, :] * 1e3  # from KHz to Hz; Inhibitory firing rate

    del result

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(time_s, FR_inh[:], label=r'$v_i$', c='r')
    ax.plot(time_s, FR_exc[:], label=r'$v_e$', c='g')
    ax.set(title=(f'S: {Job_proc[simnum][0]}, b:{Job_proc[simnum][1]}, E_L_i: {Job_proc[simnum][2]}'
                  f'E_L_e: {Job_proc[simnum][3]}, T: {Job_proc[simnum][4]}'),
           xlabel='Time(s)', ylabel='FR(Hz)',
           xlim=(2, 5), ylim=(0, 5 + max([np.amax(FR_exc), np.amax(FR_inh)])))
    plt.show()


    # Start analyzing the results
    store_npy = []  # Empty list that we will convert to array
    for i in range(len(Job_proc[simnum])):
        store_npy.append(Job_proc[simnum][i])  # We will store the param values in each file. Same order

    # Mean and std deviation of regions
    store_npy.append(np.mean(FR_exc))
    store_npy.append(np.std(FR_exc))

    # Average FC
    m_FC = mean_FC(FR_exc, do_plot=True)
    store_npy.append(m_FC)
    print(f'Mean of FC: {m_FC}')

    # Average PLI
    m_PLI = mean_PLI(FR_exc, do_plot=True)
    store_npy.append(m_PLI)
    print(f'Mean of PLI: {m_PLI}')

    plot_corr_DMN(FR_exc, dmn_regions=None)

    # Length of Up-Down states
    up_mean_len, down_mean_len = mean_UD_duration(FR_exc, dt=parameters.parameter_integrator['dt'],
                                                  ratio_threshold=0.4, len_state=50, gauss_width_ratio=10)
    store_npy.append(up_mean_len)
    store_npy.append(down_mean_len)
    print(f'Up mean length: {up_mean_len}s. Down mean length: {down_mean_len}s')

    # Global Maxima
    store_npy.append(np.amax(FR_exc))  # We'll see here if the 200 Hz fixed point appears.

    # Obtaining PSDs
    f_sampling = 1. * len(time_s) / time_s[-1]  # time in seconds, f_sampling in Hz
    frq = np.fft.fftfreq(len(time_s), 1 / f_sampling)
    psd = np.abs(np.fft.fft(FR_exc.T)) ** 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    _, ax = PSD(time_s, FR_exc, returnplot=True, ax=ax, c='r', lab='PSD')

    # Frequency at peak and amplitude of peak of power spectra.
    f_max, amp_max = psd_fmax_ampmax(frq, psd, type_mean='avg_psd', prominence=1, which_peak='both')
    for f, amp in zip(f_max, amp_max):
        ax.semilogy(f, amp, 'b*', label='peak')
        store_npy.append(f)
        store_npy.append(amp)

    # Slope of power spectra
    a, b, score = fit_psd_slope(frq, psd, range_fit=(1, 1000), type_mean='avg_psd')
    store_npy.append(a)
    store_npy.append(score)  # Store the score to know the goodness of fit

    pred = b / frq[frq > 0.1] ** a
    ax.semilogy(frq[frq > 0.1], pred, 'g', label='scale-free fit')
    ax.set_xlim((0.1, 1000))
    plt.legend()
    plt.title('PSD, peaks and fit of slope')
    plt.show()

    # Relative powers in each band
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
             'beta': (12, 30), 'gamma': (30, 100)}

    dict_rel_powers = rel_power_bands(frq, np.mean(psd, axis=0), bands, do_plot=True)

    for band in dict_rel_powers:  # Print and store results
        print('Relative power in ' + band + ': ' + str(dict_rel_powers[band]))
        store_npy.append(dict_rel_powers[band])

    # Finally, save the results.
    print(store_npy)
    print(len(store_npy))
    filename = folder_root + label_sim[:-1] + '.npy'  # Indexing to get rid of / in label_sim
    # Maybe too long a name, will it take extra space? Could be relevant, we only store a few values in the file.
    np.save(filename, np.array(store_npy))

    print(f'Time of analysis: {time.time() - start_time}s')