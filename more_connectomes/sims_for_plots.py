"""Used to simulate many times the same conditions to obtain statistics for the DMN."""
import time
import itertools
import random
import numpy as np
import tvb_model_reference.src.nuu_tools_simulation_human as tools
from tqdm import tqdm

from mpi4py import MPI
from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter

parameters = Parameter()

start_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# To run the script with 3 cores type in terminal: mpiexec -n 3 python3 script.py
#ELS_vals = [[-78.667, -78.667]]
#b_list = [0]

#ELS_A = [0]
#T_list = [19, 40]
#a_list = [0, 0.1, 0.25, 0.5]
#lst1 = [#T_list, a_list, b_list, ELS_A]
#combinaison1 = list(itertools.product(*lst1))

#ELS_B = [2]
#a_list = [0.3, 0.4]
#lst2 = [T_list, a_list, b_list, ELS_B]
#combinaison2 = list(itertools.product(*lst2))

#ELS_C = [3]
#a_list = [0.2, 0.3]
#lst3 = [T_list, a_list, b_list, ELS_C]
#combinaison3 = list(itertools.product(*lst3))

#ELS_D = [4]
#a_list = [0.4]
#lst4 = [T_list[1], a_list, b_list, ELS_D]
#combinaison4 = list(itertools.product(*lst4))

#combinaison = combinaison1 #+ combinaison2 + combinaison3 + combinaison4

T_list = [40] #[5.0, 40.0]
a_list = [0] #[0.0, 0.5]
b_list = [120] #[0.0, 120.0]
ELI_list = [-80] #[-80.0, -60.0]
ELE_list = [-80] #[-80.0, -60.0]
lst = [T_list, a_list, b_list, ELI_list, ELE_list]

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
run_sim = 5000.0  # ms, length of the simulation We run for longer to have a larger margin for computing FC
cut_transient = 2000.0  # ms, length of the discarded initial segment
Iext = 0.000315  # External input

# Define a location to save the files
# TODO Change this to save them in another folder
folder_root = '/media/master/Nuevo vol/Internship/Data/new_connectome_BOLD/'

for simnum in tqdm(range(len(Job_proc))):

    T = Job_proc[simnum][0]
    a = Job_proc[simnum][1]
    b = Job_proc[simnum][2]
    #ELS = Job_proc[simnum][3]
    #E_L_i = ELS_vals[ELS][0]
    #E_L_e = ELS_vals[ELS][1]
    E_L_i = Job_proc[simnum][3]
    E_L_e = Job_proc[simnum][4]
    parameters.parameter_model['T'] = T
    parameters.parameter_coupling['parameter']['a'] = a
    parameters.parameter_model['b_e'] = b
    parameters.parameter_model['E_L_i'] = E_L_i
    parameters.parameter_model['E_L_e'] = E_L_e

    # We change the connectome
    parameters.parameter_connection_between_region['path'] = '/home/master/Desktop/tests_hpc/more_connectomes/tvb_model_reference/data/DH_20120806'

    parameters.parameter_model['external_input_ex_ex'] = Iext
    parameters.parameter_model['external_input_in_ex'] = Iext
    parameters.parameter_model['initial_condition']['W_e'] = [0, 0]

    label_sim = '_a_' + str(a) + '_b_' + str(b) + '_ELI_' + \
                str(E_L_i) + '_ELE_' + str(E_L_e) + '_T_' + str(T) + '/'

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
