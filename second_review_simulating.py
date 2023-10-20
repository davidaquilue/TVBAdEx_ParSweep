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



def run_tvb(run_params, deterministic=False):
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

    # We want to obtain the deterministic simulation
    if deterministic:
        parameters.parameter_integrator['noise_parameter']['nsig'][-1] = 0.0
        parameters.parameter_integrator['stochastic'] = False

    label_sim = f'be_{b}_S_{S}_ELE_{E_L_e}_ELI_{E_L_i}_T_{T}_dt_{time_step}'

    file_name = folder_root + '\\' + label_sim
    parameters.parameter_simulation['path_result'] = file_name

    # Set up simulator with new parameters
    simulator = tools.init(parameters.parameter_simulation,
                           parameters.parameter_model,
                           parameters.parameter_connection_between_region,
                           parameters.parameter_coupling,
                           parameters.parameter_integrator,
                           parameters.parameter_monitor)

    print('Initiated Simulator')

    # Run simulations
    print('Start running')
    tools.run_simulation(simulator, run_sim, parameters.parameter_simulation,
                         parameters.parameter_monitor)


if __name__ == '__main__':
    # Initialize the parameters object
    parameters = Parameter()

    # 1. First we run two simulations, with different time steps,
    # but without stochastic noise.
    # Select a directory to store the simulations
    folder_root = (r'C:\Users\daqui\Desktop\temp_data\ParSweep_Paper'
                   r'\StudyStepSize\deterministic')
    cut_transient = 0.0
    run_sim = 1000.0
    time_step = 0.01
    # Type here the values of the hyperparameters to be run.
    # The order is: S, b, E_L_i, E_L_e, T
    hyperparams1 = (0.3, 120.0, -64.0, -64.0, 19.0)

    run_params = {'hyperparams': hyperparams1,
                 'simulation_params': (cut_transient, run_sim, time_step),
                 'folder_root': folder_root}

    simulate_deterministic = True

    if simulate_deterministic:
        run_tvb(run_params, deterministic=True)

    # Then one with a larger time step

    # Select a directory to store the simulations
    folder_root = (r'C:\Users\daqui\Desktop\temp_data\ParSweep_Paper'
                   r'\StudyStepSize\deterministic')
    cut_transient = 0.0
    run_sim = 1000.0
    time_step = 0.1
    # Type here the values of the hyperparameters to be run.
    # The order is: S, b, E_L_i, E_L_e, T
    hyperparams1 = (0.3, 120.0, -64.0, -64.0, 19.0)

    run_params = {'hyperparams': hyperparams1,
                 'simulation_params': (cut_transient, run_sim, time_step),
                 'folder_root': folder_root}

    simulate_deterministic = True
    if simulate_deterministic:
        run_tvb(run_params, deterministic=True)


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

    sim_small_dt = False
    if sim_small_dt:
        num_processes = multiprocessing.cpu_count() - 2
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use pool.map to parallelize the function across the parameter
            # sets
            results = pool.map(run_tvb, list_run_params)

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

    sim_large_dt = False
    if sim_large_dt:
        num_processes = multiprocessing.cpu_count() - 2
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use pool.map to parallelize the function across the parameter
            # sets
            results = pool.map(run_tvb, list_run_params)
