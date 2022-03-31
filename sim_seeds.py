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

# To simulate more processes than processors available one can use
# $ mpiexec -n 20 --oversubscribe python3 script.py. 
# That way maximum rank will be 19 and size will be 20.

steps = 20
b_vals = [0]
seeds = [i for i in range(20, 20+steps)]

lst = [b_vals, seeds]

combinaison = list(itertools.product(*lst))


L = len(combinaison)
Rem = L % size

job_len = L // size

if rank < Rem:
    Job_proc = combinaison[rank + rank * job_len: rank + 1 + (1 + rank) * job_len]
else:
    Job_proc = combinaison[Rem + rank*job_len: Rem + (1 + rank) * job_len]
        
print('I am the process number : ', rank, ' and len of my list is : ', len(Job_proc))

# Set the parameters of the simulation:
run_sim = 5000.0  # ms, length of the simulation
cut_transient = 2000.0  # ms, length of the discarded initial segment
Iext = 0.000315  # External input

# Define a location to save the files wll make sure they already created.
folder_root = '/media/master/Nuevo vol/Internship/Data/hpc_tvbadex/results_test_DMN/'

for simnum in range(len(Job_proc)):
    random.seed(Job_proc[simnum][1])
    np.random.seed(Job_proc[simnum][1])
    parameters.parameter_model['b_e'] = Job_proc[simnum][0] + np.random.rand()
    print(parameters.parameter_model['b_e'])
    parameters.parameter_simulation['seed'] = Job_proc[simnum][1] + np.random.rand()
    E_in = 5 * np.random.rand()
    I_in = 10 + E_in
    parameters.parameter_model['initial_condition']['E'] = [E_in, E_in]
    parameters.parameter_model['initial_condition']['I'] = [I_in, I_in]

    label_sim = '_b_' + str(Job_proc[simnum][0]) + '_s_' + str(Job_proc[simnum][1]) + '/'
    
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
