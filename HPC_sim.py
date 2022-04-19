import time
import itertools
import numpy as np
import tvb_model_reference.src.nuu_tools_simulation_human as tools

from mpi4py import MPI
from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
from analyses import *

parameters = Parameter()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# To run the script with 3 cores type in terminal: mpiexec -n 3 python3 script.py

# To simulate more processes than processors available one can use
# $ mpiexec -n 20 --oversubscribe python3 script.py. 
# That way maximum rank will be 19 and size will be 20.


# TODO: This will be changed once we have the folder with the chunks in the data partition of JUSUF
# This will have been done outside and here we will only need to select the chunk of the parameter space that we want
# to run.
steps = 2  # Let's say 15 at the moment.

# We will round the points in the parameter space to have only three floating point values. Will be better for the
# titles of the files and for the plots.
# HAVE THIS IN MIND TO PLOT AND ANALYZE, WE WILL HAVE TO ALWAYS ROUND TO AVOID ERRORS
S_vals = np.round(np.linspace(0, 0.5, steps), 3)
b_vals = np.round(np.linspace(0, 120, steps), 3)
E_L_i_vals = np.round(np.linspace(-80, -60, steps), 3)
E_L_e_vals = np.round(np.linspace(-80, -60, steps), 3)
T_vals = np.round(np.linspace(5, 40, steps), 3)

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

L = len(combinaison)
Rem = L % size

job_len = L // size

if rank < Rem:
    Job_proc = np.array(combinaison[rank + rank * job_len: rank + 1 + (1 + rank) * job_len])
else:
    Job_proc = np.array(combinaison[Rem + rank*job_len: Rem + (1 + rank) * job_len])
        
#print('I am the process number : ', rank, ' and len of my list is : ', Job_proc.shape[0])

# Set the parameters of the simulation:
run_sim = 5000.0  # ms, length of the simulation
cut_transient = 2000.0  # ms, length of the discarded initial segment
Iext = 0.000315  # External input

# TODO: Change these two folders everytime we execute a new job (new chunk of the parameter space)
# Define a location to save the files wll make sure they already created.
folder_root = '/media/master/Nuevo vol/Internship/Data/hpc_tvbadex/results/'  # Lab computer with external disk
# When running in HPC, we will create a directory in the partition that allows us to write huge numbers of files.
# In that folder we will create as many folders as chunks of the entire parameter space we have.
# In each chunk folder we will generate a '_a_0.0_b_0.0_ELI_0.0_ELE_0.0_T_0.0_COMPLETED.txt file after each simulation
# has been completed.
indicator_folder = '/media/master/Nuevo vol/Internship/Data/hpc_tvbadex/results/'

for simnum in range(len(Job_proc)):
    start_time = time.time()
    parameters.parameter_coupling['parameter']['a'] = Job_proc[simnum][0]
    parameters.parameter_model['b_e'] = Job_proc[simnum][1]
    parameters.parameter_model['E_L_i'] = Job_proc[simnum][2]
    parameters.parameter_model['E_L_e'] = Job_proc[simnum][3]
    parameters.parameter_model['T'] = Job_proc[simnum][4]

    parameters.parameter_model['external_input_ex_ex'] = Iext
    parameters.parameter_model['external_input_in_ex'] = Iext

    label_sim = '_a_' + str(Job_proc[simnum][0]) + '_b_' + str(Job_proc[simnum][1]) + '_ELI_' + \
                str(Job_proc[simnum][2]) + '_ELE_' + str(Job_proc[simnum][3]) + '_T_' + str(Job_proc[simnum][4]) + '/'
    
    file_name = folder_root + label_sim
    parameters.parameter_simulation['path_result'] = file_name
    
    # Set up simulator with new parameters
    simulator = tools.init(parameters.parameter_simulation, parameters.parameter_model,
                           parameters.parameter_connection_between_region,
                           parameters.parameter_coupling,
                           parameters.parameter_integrator,
                           parameters.parameter_monitor)

    # Run simulations
    tools.run_simulation(simulator, run_sim, parameters.parameter_simulation, parameters.parameter_monitor)

    sim_time = time.time()
    print(f'Time of simulation: {sim_time - start_time}s')

    result = tools.get_result(file_name, cut_transient, run_sim)
    time_s = result[0][0] * 1e-3  # from ms to sec

    FR_exc = result[0][1][:, 0, :] * 1e3  # from KHz to Hz; Excitatory firing rate
    FR_inh = result[0][1][:, 1, :] * 1e3  # from KHz to Hz; Inhibitory firing rate
    
    del result

    # We can then also delete result to free some space in disk.
    remove_results(folder_root + label_sim)  # Delete the folders, get space free as soon as possible

    # Start analyzing the results
    store_npy = []  # Empty list that we will convert to array
    for i in range(len(Job_proc[simnum])):
        store_npy.append(Job_proc[simnum][i])  # We will store the param values in each file. Same order

    # ========================================= EXCITATORY FR ========================================= #
    # Mean and std deviation of regions
    store_npy.append(np.mean(FR_exc))
    store_npy.append(np.std(FR_exc))

    # Average FC 
    store_npy.append(mean_FC(FR_exc))
    
    # Average PLI
    store_npy.append(mean_PLI(FR_exc))
    
    # Length of Up-Down states
    up_mean_len, down_mean_len = mean_UD_duration(FR_exc, dt=parameters.parameter_integrator['dt'],
                                                  ratio_threshold=0.3, len_state=20, gauss_width_ratio=10)
    store_npy.append(up_mean_len)
    store_npy.append(down_mean_len)

    # Global Maxima
    store_npy.append(np.amax(FR_exc))  # We'll see here if the 200 Hz fixed point appears.
    
    # Obtaining PSDs     
    f_sampling = 1.*len(time_s)/time_s[-1]  # time in seconds, f_sampling in Hz
    frq = np.fft.fftfreq(len(time_s), 1/f_sampling)
    psd = np.abs(np.fft.fft(FR_exc.T))**2

    # Frequency at peak and amplitude of peak of power spectra.
    f_max, amp_max = psd_fmax_ampmax(frq, psd, type_mean='avg_psd', prominence=1, which_peak='both')
    for f, amp in zip(f_max, amp_max):
        store_npy.append(f)
        store_npy.append(amp)
    
    # Slope of power spectra
    a, b, score = fit_psd_slope(frq, psd, range_fit=(0.1, 1000), type_mean='avg_psd')
    store_npy.append(a)
    store_npy.append(score)  # Store the score to know the goodness of fit

    # Relative powers in each band
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
             'beta': (12, 30), 'gamma': (30, 100)}

    dict_rel_powers = rel_power_bands(frq, np.mean(psd, axis=0), bands, do_plot=False)

    for band in dict_rel_powers:  # Store results
        store_npy.append(dict_rel_powers[band])

    # ========================================= INHIBITORY FR ========================================= #
    store_npy.append(np.mean(FR_inh))
    store_npy.append(np.std(FR_inh))
    store_npy.append(mean_FC(FR_inh))
    store_npy.append(mean_PLI(FR_inh))
    up_mean_len, down_mean_len = mean_UD_duration(FR_inh, dt=parameters.parameter_integrator['dt'],
                                                  ratio_threshold=0.3, len_state=20, gauss_width_ratio=10)
    store_npy.append(up_mean_len)
    store_npy.append(down_mean_len)
    store_npy.append(np.amax(FR_inh))
    f_sampling = 1. * len(time_s) / time_s[-1]  # time in seconds, f_sampling in Hz
    frq = np.fft.fftfreq(len(time_s), 1 / f_sampling)
    psd = np.abs(np.fft.fft(FR_inh.T)) ** 2
    f_max, amp_max = psd_fmax_ampmax(frq, psd, type_mean='avg_psd', prominence=1, which_peak='both')
    for f, amp in zip(f_max, amp_max):
        store_npy.append(f)
        store_npy.append(amp)
    a, b, score = fit_psd_slope(frq, psd, range_fit=(0.1, 1000), type_mean='avg_psd')
    store_npy.append(a)
    store_npy.append(score)

    dict_rel_powers = rel_power_bands(frq, np.mean(psd, axis=0), bands, do_plot=False)

    for band in dict_rel_powers:  # Store results
        store_npy.append(dict_rel_powers[band])

    # ========================================= PREDICTIONS ========================================= #
    FC_exc = np.corrcoef(FR_exc.T)
    store_npy.append(ratio_most_active_from_dmn(FR_exc))
    store_npy.append(ratio_zscore_from_dmn(FC_exc))
    FC_inh = np.corrcoef(FR_inh.T)
    store_npy.append(ratio_most_active_from_dmn(FR_inh))
    store_npy.append(ratio_zscore_from_dmn(FC_inh))

    store_npy.append(count_ratio_AI(FR_exc))  # Still under construction

    # ========================================= SAVING RESULTS ========================================= #
    # Finally, save the results.
    print(store_npy)
    filename = folder_root + label_sim[:-1] + '.npy'  # Indexing to get rid of / in label_sim
    np.save(filename, np.array(store_npy))

    # In order to manage simulation errors and problems that might arise during HPC computations, we save a .txt
    # file that will indicate if the simulation of a combination of parameters has been completed correctly.
    ind_filename = indicator_folder + label_sim[:-1] + 'COMPLETED.txt'
    open(ind_filename, 'a').close()

    print(f'Time of analysis: {time.time() - sim_time}s')
    print(f'Total time: {time.time() - start_time}')

    # ========================================= OUTPUT INDEXES ========================================= #

    # +===================================+=============+=============+=====+
    # |        Parameter / Metric         | idx for Exc | idx for Inh | idx |
    # +===================================+=============+=============+=====+
    # | 'a'                               |             |             |   0 |
    # +-----------------------------------+-------------+-------------+-----+
    # | 'b_e'                             |             |             |   1 |
    # +-----------------------------------+-------------+-------------+-----+
    # | 'E_L_i'                           |             |             |   2 |
    # +-----------------------------------+-------------+-------------+-----+
    # | 'E_L_e'                           |             |             |   3 |
    # +-----------------------------------+-------------+-------------+-----+
    # | 'T'                               |             |             |   4 |
    # +-----------------------------------+-------------+-------------+-----+
    # | mean of FR                        |           5 |          23 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | std of FR                         |           6 |          24 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | mean of FC matrix                 |           7 |          25 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | mean of PLI matrix                |           8 |          26 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | mean length UP state              |           9 |          27 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | mean length DOWN state            |          10 |          28 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | global maxima of FR               |          11 |          29 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | f @ peak with max power           |          12 |          30 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | power of peak with max power      |          13 |          31 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | f @ peak with max prominence      |          14 |          32 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | power of peak with max prominence |          15 |          33 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | slope of PSD fit                  |          16 |          34 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | score of PSD fit                  |          17 |          35 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | delta relative power              |          18 |          36 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | theta relative power              |          19 |          37 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | alpha relative power              |          20 |          38 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | beta relative power               |          21 |          39 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | gamma relative power              |          22 |          40 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | ratio DMN with mean FR            |          41 |          43 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | ratio DMN with zscore             |          42 |          44 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | counting ratio UD states          |          45 |             |     |
    # +-----------------------------------+-------------+-------------+-----+

