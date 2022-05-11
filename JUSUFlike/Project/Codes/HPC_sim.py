import time
import numpy as np
import tvb_model_reference.src.nuu_tools_simulation_human as tools

from analyses import *
from mpi4py import MPI
from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
from generate_data_chunks import cores_per_job, chunks_per_job


parameters = Parameter()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# When running in HPC, we will have two folders in the partition that allows us to write huge numbers of files.
# One folder will be for storing the results (divided by chunks, one extra folder/chunk)
# In the other folder, we will generate a '_a_0.0_b_0.0_ELI_0.0_ELE_0.0_T_0.0_COMPLETED.txt file after each simulation
# has been completed. It will allow us to manage job fails.


initial_chunk = 64  # From which of the 128 chunks we start to compute. CHANGE EVERYTIME NEW JOB, STEPS OF 16.
chunk_folder = '../Data/data_chunks/'  # TODO Change when we want to run Remaining Chunks
folder_assignment = '../Data/'
folder_results = '/p/scratch/icei-hbp-2022-0005/Results/' # TODO Change in JUSUF to scratch
folder_indicators = '/p/scratch/icei-hbp-2022-0005/Indicators/'

# We load the assignment matrix:
assignment_matrix = np.load(folder_assignment + 'assignment.npy')
n_chunk = initial_chunk + assignment_matrix[rank, 0]  # Which chunk we compute is given by rank and assignment.

rel_pos_chunk = assignment_matrix[rank, 1]  # The relative position in the chunk. It will act as the rank did.
cores_per_chunk = cores_per_job / chunks_per_job  # The total cores dedicated to a chunk. It will act as size did.

folder_root = folder_results + 'chunk_' + str(n_chunk) + '/'
folder_indicator = folder_indicators + 'chunk_' + str(n_chunk) + '/'

chunk = np.load(chunk_folder + 'chunk_' + str(n_chunk) + '.npy')

L = chunk.shape[0]
Rem = int(L % cores_per_chunk)

core_len = int(L // cores_per_chunk)  # How many simulations each core will have to do.

# Now we build a list in core_proc containing all the different combinations of values that the core will execute.
if rel_pos_chunk < Rem:
    core_proc = chunk[rel_pos_chunk + rel_pos_chunk * core_len: rel_pos_chunk + 1 + (1 + rel_pos_chunk) * core_len, :]
else:
    core_proc = chunk[Rem + rel_pos_chunk*core_len: Rem + (1 + rel_pos_chunk) * core_len, :]

print('I am the process number : ', rel_pos_chunk, ' and len of my list is : ', core_proc.shape[0])
print(core_len)
# Set the parameters of the simulation:
run_sim = 5000.0  # ms, length of the simulation
cut_transient = 2000.0  # ms, length of the discarded initial segment
Iext = 0.000315  # External input

for simnum in range(len(core_proc)):
    start_time = time.time()
    parameters.parameter_coupling['parameter']['a'] = core_proc[simnum][0]
    parameters.parameter_model['b_e'] = core_proc[simnum][1]
    parameters.parameter_model['E_L_i'] = core_proc[simnum][2]
    parameters.parameter_model['E_L_e'] = core_proc[simnum][3]
    parameters.parameter_model['T'] = core_proc[simnum][4]

    parameters.parameter_model['external_input_ex_ex'] = Iext
    parameters.parameter_model['external_input_in_ex'] = Iext

    label_sim = '_a_' + str(core_proc[simnum][0]) + '_b_' + str(core_proc[simnum][1]) + '_ELI_' + \
                str(core_proc[simnum][2]) + '_ELE_' + str(core_proc[simnum][3]) + '_T_' + str(core_proc[simnum][4]) + '/'

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
    # print(f'Time of simulation: {sim_time - start_time}s')

    result = tools.get_result(file_name, cut_transient, run_sim)
    time_s = result[0][0] * 1e-3  # from ms to sec

    FR_exc = result[0][1][:, 0, :] * 1e3  # from KHz to Hz; Excitatory firing rate
    FR_inh = result[0][1][:, 1, :] * 1e3  # from KHz to Hz; Inhibitory firing rate

    del result

    # We can then also delete result to free some space in disk.
    remove_results(folder_root + label_sim)  # Delete the folders, get space free as soon as possible

    # Start analyzing the results
    store_npy = []  # Empty list that we will convert to array
    for i in range(len(core_proc[simnum])):
        store_npy.append(core_proc[simnum][i])  # We will store the param values in each file. Same order

    # ========================================= EXCITATORY FR ========================================= #
    # Mean and std deviation of regions
    mean_FR_e = np.mean(FR_exc)
    std_FR_e = np.std(FR_exc)
    store_npy.append(mean_FR_e)
    store_npy.append(std_FR_e)

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
    mean_FR_i = np.mean(FR_inh)
    std_FR_i = np.std(FR_inh)
    store_npy.append(mean_FR_i)
    store_npy.append(std_FR_i)
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

    weights_mat = np.array(simulator.connectivity.weights)
    tracts_mat = np.array(simulator.connectivity.tract_lengths)
    # Correlations between FC matrix and SC matrix
    store_npy.append(np.corrcoef(x=FC_exc.flatten(), y=weights_mat.flatten())[0, 1])
    store_npy.append(np.corrcoef(x=FC_inh.flatten(), y=weights_mat.flatten())[0, 1])
    # Correlations between FC matrix and tract lengths matrix.
    # Reason: longer tract, more time to reach effect, less synchronized.
    store_npy.append(np.corrcoef(x=FC_exc.flatten(), y=tracts_mat.flatten())[0, 1])
    store_npy.append(np.corrcoef(x=FC_inh.flatten(), y=tracts_mat.flatten())[0, 1])
    # ============================= SAVING CV AND MEANS AND std of means ============================== #
    # Coefficient of variation
    store_npy.append(std_FR_e / mean_FR_e)
    store_npy.append(std_FR_i / mean_FR_i)

    # variance of the mean (how the mean changes between. How much the means of the node vary
    store_npy.append(np.std(np.mean(FR_exc, axis=0)))
    store_npy.append(np.std(np.mean(FR_inh, axis=0)))

    # And also mean of variance. How much each node varies
    store_npy.append(np.mean(np.std(FR_exc, axis=0)))
    store_npy.append(np.mean(np.std(FR_inh, axis=0)))
    # ========================================= SAVING RESULTS ========================================= #
    # Finally, save the results.
    filename = folder_root + label_sim[:-1] + '.npy'  # Indexing to get rid of / in label_sim
    np.save(filename, np.array(store_npy))

    # In order to manage simulation errors and problems that might arise during HPC computations, we save a .txt
    # file that will indicate if the simulation of a combination of parameters has been completed correctly.
    ind_filename = folder_indicator + label_sim[:-1] + '_COMPLETED.txt'
    open(ind_filename, 'a').close()

    # print(f'Time of analysis: {time.time() - sim_time}s')
    # print(f'Total time: {time.time() - start_time}')

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
    # | counting ratio UD states          |             |             |   45|
    # +-----------------------------------+-------------+-------------+-----+
    # | correlation between FC and SC     |          46 |          47 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | correlation between FC and tracts |          48 |          49 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | Coefficient of variation(std/mean)|          50 |          51 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | Standard deviation of means       |          52 |          53 |     |
    # +-----------------------------------+-------------+-------------+-----+
    # | Mean of standard deviation        |          54 |          55 |     |
    # +-----------------------------------+-------------+-------------+-----+
