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

T_list = [12, 19, 26]
E_L = - 64
a_list = [0.3, 0.4, 0.5]
b_list = np.linspace(0, 120, 6).tolist()


lst = [T_list, a_list, b_list]
lst_corners = [T_list, [0.0], [0.0, 120.0]]

combinaison1 = list(itertools.product(*lst))
combinaison2 = list(itertools.product(*lst_corners))

combinaison = combinaison1 + combinaison2

L = len(combinaison)
Rem = L % size

job_len = L // size

if rank < Rem:
    Job_proc = combinaison[rank + rank * job_len: rank + 1 + (1 + rank) * job_len]
else:
    Job_proc = combinaison[Rem + rank * job_len: Rem + (1 + rank) * job_len]

print('I am the process number : ', rank, ' and len of my list is : ', len(Job_proc))

# Set the parameters of the simulation:
run_sim = 5000.0  # ms, length of the simulation We run for longer to have a larger margin for computing FC
cut_transient = 2000.0  # ms, length of the discarded initial segment
Iext = 0.000315  # External input

# Define a location to save the files
folder_root = '/media/master/Nuevo vol/Internship/Data/hpc_tvbadex/results_for_plotting/'

for simnum in range(len(Job_proc)):
    parameters.parameter_model['T'] = Job_proc[simnum][0]
    parameters.parameter_coupling['parameter']['a'] = Job_proc[simnum][1]
    parameters.parameter_model['b_e'] = Job_proc[simnum][2]
    parameters.parameter_model['E_L_i'] = E_L
    parameters.parameter_model['E_L_e'] = E_L

    parameters.parameter_model['external_input_ex_ex'] = Iext
    parameters.parameter_model['external_input_in_ex'] = Iext

    label_sim = '_a_' + str(Job_proc[simnum][1]) + '_b_' + str(Job_proc[simnum][2]) + '_ELI_' + \
                str(E_L) + '_ELE_' + str(E_L) + '_T_' + str(Job_proc[simnum][0]) + '/'
    print(label_sim)
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
