# Cython declarations and values of parameters common to all modules

# Importing the cython module for use in this module and all others
import cython
# List of dummy Cython compiler directives for use in pure Python
cython_directives = ['boundscheck', 'wraparound', 'cdivision']
# Create dummy decorators for dummy directives
def dummy_function(*args, **kwargs):
    def dummy_decorator(fun):
        return fun
    return dummy_decorator

# Constants which have to be specified directly
cython.declare(dt='double')
cython.declare(G='double')
cython.declare(N='int')
cython.declare(boxsize='double')
cython.declare(ewald_gridsize='int')
cython.declare(softening='double')
dt = 0.0001           # The time step
G_Newton = 1         # Newtons constant
N = 2                # Number of particles
boxsize = 3          # Linear size of the simulation box
ewald_gridsize = 64  # Linear size of the grid of Ewald corrections
softening = 0        # Softening length

# Derived and internally defined constants
cython.declare(ewald_file='str')
cython.declare(boxsize2='double')
cython.declare(softening2='double')
cython.declare(machine_eps='double')
ewald_file = '.ewald_gridsize=' + str(ewald_gridsize) + '_precision=64.npy'  # Name of file storing the Ewald grid
boxsize2 = boxsize**2
softening2 = softening**2
from numpy import finfo
machine_eps = finfo('float64').eps  # Machine epsilon

# Frequently used imports common to pure Python and Cython
from numpy import array, zeros, shape

# MPI setup
from mpi4py import MPI
cython.declare(nprocs='int')
cython.declare(rank='int')
cython.declare(master='bint')
# Functions for (collective) communication
Abort = MPI.COMM_WORLD.Abort
Allgather = MPI.COMM_WORLD.Allgather
Bcast = MPI.COMM_WORLD.Bcast
Reduce = MPI.COMM_WORLD.Reduce
Scatter = MPI.COMM_WORLD.Scatter
nprocs = MPI.COMM_WORLD.size  # Number of processes started with mpiexec
rank = MPI.COMM_WORLD.rank    # The unique rank of the running process
master = not rank             # Flag for identifying the master/root process (that which have rank 0)
# Function for easily partitioning of multidimensional arrays
from numpy import prod, unravel_index
@cython.locals(# Arguments
               array_shape='tuple',
               # Locals
               problem_size='int',
               local_size='int',
               errmsg='str',
               indices_start='int[::1]',
               indices_end='int[::1]',
               )
def partition(array_shape):
    """ This function takes in the shape of an array as the argument
    and returns the start and end indices corresponding to the local chunk
    of the array which should be processed by running process,
    base on rank and nprocs.
    """
    # Raise an exception if nprocs > problem_size
    problem_size = prod(array_shape)
    if problem_size < nprocs:
        errmsg = ('Cannot partition the workload because the number of\nprocesses ('
                  + str(nprocs) + ') is larger than the problem size (' + str(problem_size) + ').')
        raise ValueError(errmsg)
    # Partition the local shape based on the rank
    local_size = int(problem_size/nprocs)
    indices_start = array(unravel_index(local_size*rank, array_shape), dtype='int32')
    indices_end = array(unravel_index(local_size*(rank+1)-1, array_shape), dtype='int32') + 1
    return indices_start, indices_end
