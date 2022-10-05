# How to make use of ranks to parallelize the code using MPI

import numpy as np
ranks = list(np.arange(0, 12))  # Array with different processes

combinaison = list(np.random.randn(97))

L = len(combinaison)
Np = len(ranks)  # Also known as size

Rem = L % Np  # Remainder of division
print(f'Combinations: {L}. Ranks: {Np} and remainder: {Rem}')

job_len = L//Np

# My idea, divide the remainder simulations throughout other processes, so that one does not have to do much more work
# thank the others. The only information each process will have will be:
# - its rank
# - the total number of processes aka size (here called Np)
# - the combinations list (and thus the length of the list, here L)

# Making use of the rank, which goes from 0 to Np it might be possible to distribute the remaining simulations.

for rank in ranks:
    # To each process having a rank smaller than the Rem value, we give it one simulation more
    if rank < Rem:
        rank_len = rank + 1 + (1 + rank) * job_len - (rank + rank*job_len)  # Clearly can be simplified
        print(f'I am rank: {rank} and take from {rank + rank*job_len} to {rank + 1 + (1 + rank) * job_len}')
        print(f'My length is: {rank_len}')
    # To the others, we just stay with the job_len
    else:
        rank_len = Rem + (1 + rank) * job_len - (Rem + rank*job_len)  # Clearly can be simplified
        print(f'I am rank{rank} and take from {Rem + rank*job_len} to {Rem + (1 + rank) * job_len}')
        print(f'My length is: {rank_len}')

# This should add up to the total number!