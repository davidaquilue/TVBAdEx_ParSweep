"""
Revisions for Neuroinformatics

This script is used to perform different simulations to answer the comments of
the reviewers of the paper.

The first simulation consists of running the TVB model for a very long period
of time, to prove that the UD are metastable.

The second simulation consists in running multiple, different, iterations of
the same model and parameter combination, to check that the model tends to
converge before the 2.5s and that initialization artifacts have little impact
in the analyses of the parameter space.
"""
import multiprocessing
import tvb_model_reference.src.nuu_tools_simulation_human as tools
from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import \
    Parameter
from JUSUFlike.Project.Codes.analyses import *
import json


def obtain_metrics(run_params):
    parameters = Parameter()
    S, b, E_L_i, E_L_e, T = run_params['hyperparams']
    cut_transient, run_sim, time_step = run_params['simulation_params']
    folder_root = run_params['folder_root']

    parameters.parameter_coupling['parameter']['a'] = S
    parameters.parameter_model['b_e'] = b
    parameters.parameter_model['E_L_i'] = E_L_i
    parameters.parameter_model['E_L_e'] = E_L_e
    parameters.parameter_model['T'] = T
    print('Parameter models defined')

    parameters.parameter_integrator['dt'] = time_step
    parameters.parameter_integrator['noise_parameter']['dt'] = time_step
    print('Changed Time Step')

    label_sim = f'be_{b}_S_{S}_ELE_{E_L_e}_ELI_{E_L_i}_T_{T}_dt_{time_step}'

    file_name = folder_root + '\\' + label_sim
    parameters.parameter_simulation['path_result'] = file_name

    result = tools.get_result(file_name, cut_transient, run_sim)
    time_s = result[0][0] * 1e-3  # from ms to sec

    FR_exc = result[0][1][:, 0, :] * 1e3  # from KHz to Hz; Excitatory
    # firing rate

    # Start analyzing the results
    dict_metrics = {}

    # Mean and std deviation of regions
    dict_metrics['mean_FR_e'] = np.mean(FR_exc)
    dict_metrics['std_FR_e'] = np.std(FR_exc)


    # Average FC
    dict_metrics['mean_FC'] = mean_FC(FR_exc)

    # Average PLI
    dict_metrics['mean_PLI'] = mean_PLI(FR_exc)

    # Length of Up-Down states
    up_mean_len, down_mean_len = mean_UD_duration(FR_exc, dt=
    parameters.parameter_integrator['dt'],
                                                  ratio_threshold=0.3,
                                                  len_state=20,
                                                  gauss_width_ratio=10)
    dict_metrics['up_mean_len'] = up_mean_len
    dict_metrics['down_mean_len'] = down_mean_len

    # Global Maxima
    dict_metrics['FR_max'] = np.amax(FR_exc)  # We'll see here if the 200 Hz fixed point appears.

    # Obtaining PSDs
    f_sampling = 1. * len(time_s) / time_s[-1]  # time in seconds, f_sampling in Hz
    frq = np.fft.fftfreq(len(time_s), 1 / f_sampling)
    psd = np.abs(np.fft.fft(FR_exc.T)) ** 2

    # Frequency at peak and amplitude of peak of power spectra.
    f_max, amp_max = psd_fmax_ampmax(frq, psd, type_mean='avg_psd',
                                     prominence=1, which_peak='both')
    # We might have some of them. Let's keep whatever the output
    dict_metrics['f_max'] = f_max
    dict_metrics['amp']= amp_max

    # Slope of power spectra
    a, b, score = fit_psd_slope(frq, psd, range_fit=(0.1, 1000),
                                type_mean='avg_psd')
    dict_metrics['slope_psd'] = a
    dict_metrics['score_slope_psd'] = score  # Store the score to know the goodnessf fit

    # Relative powers in each band
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
             'beta': (12, 30), 'gamma': (30, 100)}

    dict_rel_powers = rel_power_bands(frq, np.mean(psd, axis=0), bands,
                                      do_plot=False)

    # Set up the simulator just to have the connectivity weights and
    # tractlengths
    simulator = tools.init(parameters.parameter_simulation, parameters.parameter_model,
                           parameters.parameter_connection_between_region,
                           parameters.parameter_coupling,
                           parameters.parameter_integrator,
                           parameters.parameter_monitor)

    FC_exc = np.corrcoef(FR_exc.T)
    weights_mat = np.array(simulator.connectivity.weights)
    # Correlations between FC matrix and SC matrix
    dict_metrics['corrFCSC'] = np.corrcoef(x=FC_exc.flatten(),
                                           y=weights_mat.flatten())[0, 1]

    # Finally, save also the mean of the variance
    # And also mean of variance. How much each node varies
    dict_metrics['mean_of_std_e'] = np.mean(np.std(FR_exc, axis=0))

    for band in dict_rel_powers:  # Store results
        dict_metrics[f'rel_bp_{band}'] = dict_rel_powers[band]

    with open(file_name + '\\analyses.json', 'w') as file:
        json.dump(dict_metrics, file, indent=4)


if __name__ == '__main__':
    # 2. Then run a parameter sweep, for each of the two time steps, over the
    # values of b_e, to demonstrate that the time step does not have much of
    # an effect.

    # First with smaller dt
    parameters = Parameter()
    folder_root = (r'C:\Users\daqui\Desktop\temp_data\ParSweep_Paper'
                   r'\StudyStepSize\stochastic_sweep')

    cut_transient = 2000.0
    run_sim = 5000.0
    time_step = 0.01
    # Type here the values of the hyperparameters to be run.
    # The order is: S, b, E_L_i, E_L_e, T
    b_vals = [0, 20, 40, 60, 80]
    list_run_params = []
    for b in b_vals:
        hyperparams = (0.3, b, -64.0, -64.0, 19.0)
        list_run_params.append({'hyperparams': hyperparams,
                                'simulation_params': (cut_transient,
                                                      run_sim, time_step),
                                'folder_root': folder_root})

    obtain_metrics_small_dt = True
    if obtain_metrics_small_dt:
        num_processes = 3
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use pool.map to parallelize the function across the parameter
            # sets
            results = pool.map(obtain_metrics, list_run_params)

    # Then repeat with larger dt
    folder_root = (r'C:\Users\daqui\Desktop\temp_data\ParSweep_Paper'
                   r'\StudyStepSize\stochastic_sweep')

    cut_transient = 2000.0
    run_sim = 5000.0
    time_step = 0.1
    # Type here the values of the hyperparameters to be run.
    # The order is: S, b, E_L_i, E_L_e, T
    b_vals = [0, 20, 40, 60, 80]
    list_run_params = []
    for b in b_vals:
        hyperparams = (0.3, b, -64.0, -64.0, 19.0)
        list_run_params.append({'hyperparams': hyperparams,
                                'simulation_params': (cut_transient,
                                                      run_sim, time_step),
                                'folder_root': folder_root})

    obtain_metrics_large_dt = True
    if obtain_metrics_large_dt:
        num_processes = 3
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use pool.map to parallelize the function across the parameter
            # sets
            results = pool.map(obtain_metrics, list_run_params)
