# This module contains imports, Cython declarations and values
# of parameters common to all other modules. Each module should have
# 'from commons import *' as its first statement.

##############################################################################
# Imports common to pure Python and Cython                                   #
##############################################################################
from __future__ import division  # Needed for Python3 division in Cython
from numpy import array, empty, zeros, concatenate, delete
import h5py

##############################################################################
# Cython-related stuff                                                       #
##############################################################################
import cython
# Declarations exclusively to either pure Python or Cython
if not cython.compiled:
    # Dummy Cython compiler directives (decorators)
    def dummy_function(*args, **kwargs):
        def dummy_decorator(func):
            return func
        return dummy_decorator
    for directive in ('boundscheck', 'wraparound', 'cdivision'):
        setattr(cython, directive, dummy_function)
    # Dummy Cython functions
    for func in ('address', ):
        setattr(cython, func, lambda _: _)
else:
    # Lines in triple quotes will be executed in .pyx files.
    """
    # Get full access to all of Cython
    cimport cython
    # Import the signed integer type ptrdiff_t
    from libc.stddef cimport ptrdiff_t
    # Function type definitions
    ctypedef double  (*scalar_func)(double, double, double)
    ctypedef double* (*vector_func)(double, double, double)
    """
# Seperate but equivalent imports and definitions in pure Python and Cython
if not cython.compiled:
    # Mathematical functions
    from numpy import pi, sqrt, exp, sin
    from scipy.special import erfc
    # Allocate a (module level) global vector
    vector = empty(3)
    # Import all user specified constants
    from params import *
else:
    # Lines in triple quotes will be executed in .pyx files.
    """
    # Mathematical functions
    from libc.math cimport M_PI as pi
    from libc.math cimport sqrt, exp, sin, erfc
    # Allocate a (module level) global vector
    from cpython.mem cimport PyMem_Malloc
    cython.declare(vector='double*')
    vector = <double*> PyMem_Malloc(3*sizeof(double))
    # Import all user specified constants 
    from params cimport *
    """

##############################################################################
# Derived and internally defined constants                                   #
##############################################################################
cython.declare(use_PM='bint',
               ewald_file='str',
               boxsize2='double',
               two_ewald_gridsize='int',
               PM_gridsize3='ptrdiff_t',
               softening2='double',
               machine_ϵ='double',
               two_machine_ϵ='double',
               )
use_PM = True  # Flag specifying wheter the PM method is used or not. THIS SHOULD BE COMPUTED BASED ON PARTICLES CHOSEN IN THE PARAMETER FILE!!!!!!!!!!!
ewald_file = '.ewald_gridsize=' + str(ewald_gridsize) + '.hdf5'  # Name of file storing the Ewald grid
boxsize2 = boxsize**2
two_ewald_gridsize = 2*ewald_gridsize
PM_gridsize3 = PM_gridsize**3
softening2 = softening**2
from numpy import finfo
machine_ϵ = finfo('float64').eps  # Machine epsilon
two_machine_ϵ = 2*machine_ϵ

##############################################################################
# MPI setup                                                                  #
##############################################################################
from mpi4py import MPI
cython.declare(nprocs='int',
               rank='int',
               master='bint',
               )
# Functions for (collective) communication
comm = MPI.COMM_WORLD
Abort = comm.Abort
Allgather = comm.Allgather
Allgatherv = comm.Allgatherv
Allreduce = comm.Allreduce
Bcast = comm.Bcast
Reduce = comm.Reduce
Scatter = comm.Scatter
Sendrecv = comm.Sendrecv
nprocs = comm.size  # Number of processes started with mpiexec
rank = comm.rank    # The unique rank of the running process
master = not rank   # Flag identifying the master/root process (that which have rank 0)
# Function for easily partitioning of multidimensional arrays
from numpy import prod, unravel_index
@cython.locals(# Arguments
               array_shape='tuple',
               # Locals
               problem_size='int',
               local_size='int',
               errmsg='str',
               indices_start='size_t[::1]',
               indices_end='size_t[::1]',
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
    # Partition the local shape based on the rank.
    # size_t should correspond to uint64 un a 64 bit machine. Otherwize a ValueError will be thrown.
    local_size = int(problem_size/nprocs)
    indices_start = array(unravel_index(local_size*rank, array_shape), dtype='uint64')
    indices_end = array(unravel_index(local_size*(rank + 1) - 1, array_shape), dtype='uint64') + 1
    return indices_start, indices_end

