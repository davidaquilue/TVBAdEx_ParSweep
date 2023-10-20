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

from JUSUFlike.Project.Codes.processing_results import *
import tvb_model_reference.src.nuu_tools_simulation_human as tools
from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import \
    Parameter


def run_tvb(run_params):
    parameters = Parameter()
    S, b, E_L_i, E_L_e, T = run_params['hyperparams']
    cut_transient, run_sim = run_params['simulation_params']
    folder_root = run_params['folder_root']

    parameters.parameter_coupling['parameter']['a'] = S
    parameters.parameter_model['b_e'] = b
    parameters.parameter_model['E_L_i'] = E_L_i
    parameters.parameter_model['E_L_e'] = E_L_e
    parameters.parameter_model['T'] = T
    print('Parameter models defined')

    label_sim = f'be-{b}_S-{S}_ELE-{E_L_e}_ELI-{E_L_i}_T-{T}/'

    file_name = folder_root + label_sim
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


def run_tvb_initconds(run_params):
    parameters = Parameter()
    S, b, E_L_i, E_L_e, T = run_params['hyperparams']
    cut_transient, run_sim = run_params['simulation_params']
    folder_root = run_params['folder_root']
    E_init, I_init, n_it = run_params['initial_conditions']

    parameters.parameter_coupling['parameter']['a'] = S
    parameters.parameter_model['b_e'] = b
    parameters.parameter_model['E_L_i'] = E_L_i
    parameters.parameter_model['E_L_e'] = E_L_e
    parameters.parameter_model['T'] = T
    print('Parameter models defined')

    parameters.parameter_model['initial_condition']['E'] = [E_init, E_init]
    parameters.parameter_model['initial_condition']['I'] = [I_init, I_init]
    print('Initial Conditions defined')

    label_sim = f'{n_it}_be-{b}_S-{S}_ELE-{E_L_e}_ELI-{E_L_i}_T-{T}/'

    file_name = folder_root + label_sim
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

    # 1. First we will want to run a single, very long simulation

    # Select a directory to store the simulations
    tests_root = (r'C:\Users\daqui\Desktop\temp_data\ParSweep_Paper'
                  r'\TestingUpDownStates')
    cut_transient = 95000.0
    run_sim = 100000.0

    # Type here the values of the hyperparameters to be run.
    # The order is: S, b, E_L_i, E_L_e, T
    hyperparams1 = (0.23, 120.0, -64.0, -64.0, 19.0)
    hyperparams2 = (0.23, 0.0, -78.667, -80.0, 40.0)

    list_hyperparams = [hyperparams1, hyperparams2]
    folder_root = tests_root + '\\Long\\'

    list_runparams = [{'hyperparams': set_hyperparams,
                       'simulation_params': (cut_transient, run_sim),
                       'folder_root': folder_root} for set_hyperparams in
                      list_hyperparams]

    sim_long = True

    if sim_long:
        run_tvb(list_runparams[0])
        num_processes = 1  # Use the number of available CPU cores
        with multiprocessing.Pool(processes=num_processes) as pool:
            # pool.map to parallelize the function across the parameter sets
            results = pool.map(run_tvb, list_runparams)

    # 2. Then, we will run multiple, single and short simulations, where each
    # simulation contains a different random initialization

    # Initial conditions to be chosen at random, in a certain range.
    # See that they induce transient states.
    n_inconds = 50

    # Let us say that it is between 0 and 30. See what happens
    max_init = 30
    min_init = 0

    cut_transient = 0
    run_sim = 4000

    set_hyperparams = hyperparams1
    folder_root = tests_root + '\\Initial_conds\\'

    list_runparams = []
    for n in range(n_inconds):
        rand_init_conds = (np.random.uniform(min_init, max_init),
                           np.random.uniform(min_init, max_init),
                           n)
        list_runparams.append({'hyperparams': set_hyperparams,
                               'simulation_params': (cut_transient, run_sim),
                               'folder_root': folder_root,
                               'initial_conditions': rand_init_conds})

    sim_init_conds = False
    if sim_init_conds:
        num_processes = multiprocessing.cpu_count() - 2
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use pool.map to parallelize the function across the parameter
            # sets
            results = pool.map(run_tvb_initconds, list_runparams)
