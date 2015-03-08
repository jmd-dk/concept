# Run the code
import timeloop

# Print out success message
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
master = not rank
if master:
    os.system('printf "\033[1m\033[92mCO\033[3mN\033[0m\033[1m\033[92mCEPT'
              + ' finished successfully\033[0m\n"')
