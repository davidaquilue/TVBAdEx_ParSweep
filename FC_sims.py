"""Used to simulate many times the same conditions to obtain statistics for the DMN."""
import time
import itertools
import random
import numpy as np
import tvb_model_reference.src.nuu_tools_simulation_human as tools

from mpi4py import MPI
from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter

parameters = Parameter()

start_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# To run the script with 3 cores type in terminal: mpiexec -n 3 python3 script.py

stimval_max = 0.1 / 1000  # 0.1 Hz if the model runs with kHz.
stimvals = [stimval_max/3, stimval_max/2 , stimval_max]
stim_lens = [50, 500, 1000, 2000]  # in ms
stim_region = 5

lst = [stimvals, stim_lens]

combinaison = list(itertools.product(*lst))

L = len(combinaison)
Rem = L % size

job_len = L // size

if rank < Rem:
    Job_proc = combinaison[rank + rank * job_len: rank + 1 + (1 + rank) * job_len]
else:
    Job_proc = combinaison[Rem + rank * job_len: Rem + (1 + rank) * job_len]

print('I am the process number : ', rank, ' and len of my list is : ', len(Job_proc))

# Set the parameters of the simulation:
run_sim = 5000.0  # ms, length of the simulation
cut_transient = 2000.0  # ms, length of the discarded initial segment
Iext = 0.000315  # External input

# Define a location to save the files wll make sure they already created.
folder_root = '/media/master/Nuevo vol/Internship/Data/hpc_tvbadex/results_test_DMN/'

for simnum in range(len(Job_proc)):
    stimval = Job_proc[simnum][0]
    stimlen = Job_proc[simnum][1]

    weights = list(np.zeros(68))
    weights[stim_region] = stimval

    parameters.parameter_stimulus['tau'] = stimlen  # Stimulus duration
    parameters.parameter_stimulus['T'] = 1e9  # Interstimulus interval
    parameters.parameter_stimulus['weights'] = weights
    parameters.parameter_stimulus['variables'] = [0]  # kick FR_exc
    parameters.parameter_stimulus['onset'] = 2500.0  # All stimuli start at 2.5s

    label_sim = '_stimval_' + str(stimval) + '_stimlen_' + str(stimlen) + '/'

    file_name = folder_root + label_sim
    parameters.parameter_simulation['path_result'] = file_name

    # Set up simulator with new parameters
    simulator = tools.init(parameters.parameter_simulation, parameters.parameter_model,
                           parameters.parameter_connection_between_region,
                           parameters.parameter_coupling,
                           parameters.parameter_integrator,
                           parameters.parameter_monitor,
                           parameter_stimulation=parameters.parameter_stimulus)

    # Run simulations
    tools.run_simulation(simulator, run_sim, parameters.parameter_simulation, parameters.parameter_monitor)
