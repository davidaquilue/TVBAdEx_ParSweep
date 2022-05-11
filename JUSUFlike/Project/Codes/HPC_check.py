import numpy as np
from mpi4py import MPI
from generate_data_chunks import cores_per_job, chunks_per_job
from processing_results import check_clean_chunk, batch_files

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_chunk = rank
# I will check each chunk in parallel. Each process will process a chunk
indicator_folder = '/p/scratch/icei-hbp-2022-0005/Indicators/chunk_' + str(n_chunk) + '/'
results_folder = '/p/scratch/icei-hbp-2022-0005/Results/chunk_' + str(n_chunk) + '/'
parsw_chunk = '../Data/data_chunks/chunk_' + str(n_chunk) + '.npy'
rem_chunk_folder = '../Data/rem_chunks/'
batches_folder = '../FinalResults/'
check_clean_chunk(indicator_folder, results_folder, parsw_chunk, rem_chunk_folder, verbose=True)
batch_files(results_folder, batches_folder, batch_size=5280, n_cols=56, name_batch=str(n_chunk))



