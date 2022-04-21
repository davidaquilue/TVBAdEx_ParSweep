import numpy as np
import itertools
import os

# Executing this file will generate different chunks of the parameter space and save them as .npy files in the
# /Project/Data/data_chunks/ folder.
# It will additionally generate the different directories (one for each chunk) in the /Scratch/Results/ and
# the /Scratch/Indicators
number_chunks = 4
# TODO these folder names will still need to be changed
folder_chunks = '../Data/data_chunks/'
folder_indicators = '../../Scratch/Indicators/'
folder_results = '../../Scratch/Results/'


steps = 2
S_vals = np.round(np.linspace(0, 0.5, steps), 3)
b_vals = np.round(np.linspace(0, 120, steps), 3)
E_L_e_vals = np.round(np.linspace(-80, -60, steps), 3)
E_L_i_vals = np.round(np.linspace(-80, -60, steps), 3)
T_vals = np.round(np.linspace(5, 40, steps), 3)

S_vals = list(S_vals)
b_vals = list(b_vals)
E_L_i_vals = list(E_L_i_vals)
E_L_e_vals = list(E_L_e_vals)
T_vals = list(T_vals)

lst = [S_vals, b_vals, E_L_i_vals, E_L_e_vals, T_vals]

combinaison = np.array(list(itertools.product(*lst)))

print(f'Complete data array shape: {combinaison.shape}')

# Let's divide it into different jobs
comb_div = np.array_split(combinaison, number_chunks)

for chunk_id, chunk in enumerate(comb_div):
    # Save the .npy data chunk
    file_name = folder_chunks + 'chunk_' + str(chunk_id) + '.npy'
    np.save(file_name, chunk)
    # Create both Indicator and Results folder for each chunk
    indicators_chunk_path = folder_indicators + 'chunk_' + str(chunk_id) + '/'
    if not os.path.exists(indicators_chunk_path):
        os.mkdir(indicators_chunk_path)

    results_chunk_path = folder_results + 'chunk_' + str(chunk_id) + '/'
    if not os.path.exists(results_chunk_path):
        os.mkdir(results_chunk_path)
