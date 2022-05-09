import numpy as np
import itertools
import os

# Executing this file will generate different chunks of the parameter space and save them as .npy files in the
# /Project/Data/data_chunks/ folder.
# It will additionally generate the different directories (one for each chunk) in the /Scratch/Results/ and
# the /Scratch/Indicators
number_chunks = 128
thresh_silence = 4
# TODO these folder names will still need to be changed
folder_chunks = '../Data/data_chunks/'
folder_indicators = '/p/scratch/icei-hbp-2022-0005/Indicators/'
folder_results = '/p/scratch/icei-hbp-2022-0005/Results/'
folder_assignment = '../Data/'

# In order to be able to parallelize the processing, change in structure. Will divide space in 128 chunks of 5280
# combinations in each chunk. In each job we will execute 16 chunks making use of 8 nodes (8*128cores=1024cores).
# So, for each chunk, we will use 64 cores. To do that we need to store a (2, 1024) array where the first row will
# let us know which chunk the core will work on and the second row will let us know the relative position inside of
# the chunk.
cores_per_job = 1024
chunks_per_job = 16
cores_per_chunk = cores_per_job / chunks_per_job  # This will be the new "size" that we used in HPC_sim
steps = 16

if __name__ == '__main__':
    assignment_matrix = np.zeros((cores_per_job, 2), dtype=np.int8)  # File that will be used to assign the chunk and
    # relative position in chunk to each core.

    count_chunk = 0  # counter for assignment of chunks
    count_rel_pos = 0  # counter for relative position in chunk
    for i in range(cores_per_job):
        assignment_matrix[i, 0] = int(count_chunk)
        assignment_matrix[i, 1] = int(count_rel_pos)
        count_rel_pos += 1
        if count_rel_pos == cores_per_chunk:
            count_rel_pos = 0
            count_chunk += 1

    np.save(folder_assignment + 'assignment.npy', assignment_matrix)

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

    # However, those that result in silent behavior, we don't want to calculate them.
    idx_keep = combinaison[:, 3] > combinaison[:, 2] - thresh_silence  # We keep E_L_e > E_L_i - thresh_silence
    combinaison = combinaison[idx_keep, :]  # And eliminate those combinations that are not needed.

    print(f'Complete data array shape: {combinaison.shape}')

    # Let's divide it into different chunks
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
