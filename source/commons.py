# This module contains imports, Cython declarations and values
# of parameters common to all other modules. Each module should have
# 'from commons import *' as its first statement.

############################################
# Imports common to pure Python and Cython #
############################################
from __future__ import division  # Needed for Python3 division in Cython
from numpy import (arange, array, asarray, concatenate, cumsum, delete, empty,
                   linspace, max, ones, prod, trapz, sum, unravel_index, zeros)  # FIND OUT WHY min CANNOT BE IMPORTED WITHOUT SCREWING UP EVERYTHING!!!
from numpy.random import random
import numpy as np
import h5py
import os
import sys

# WHILE DEVELOPING
from time import time, sleep

########################
# Cython-related stuff #
########################
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
    # C allocation syntax for memory management
    def sizeof(dtype):
        # C dtype names to Numpy dtype names
        if dtype == 'int':
            dtype == 'int32'
        elif dtype == 'double':
            dtype == 'float64'
        elif dtype == 'size_t':
            dtype == 'uintp'
        else:
            raise TypeError(dtype + ' not implemented as a Numpy dtype in commons.py')
        return array([1], dtype=dtype)
    def malloc(a):
        return empty(a[0], dtype=a.dtype)
    def realloc(p, a):
        new_a = empty(a[0], dtype=a.dtype)
        if new_a.size >= p.size:
            new_a[:p.size] = p
        else:
            new_a[:] = p[:new_a.size]
        return new_a
    def free(a):
        pass
    # Array casting
    def cast(a, dtype):
        return a
else:
    # Lines in triple quotes will be executed in .pyx files.
    """
    # Get full access to all of Cython
    cimport cython
    # GNU Scientific Library
    from cython_gsl cimport *
    # Mathematical functions
    from libc.math cimport round
    # Import the signed integer type ptrdiff_t
    from libc.stddef cimport ptrdiff_t
    # Functions for manual memory management
    from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
    # Function type definitions of the form func_returntype_argumenttypes
    ctypedef double  (*func_d_dd)(double, double)
    ctypedef double  (*func_d_ddd)(double, double, double)
    ctypedef double* (*func_ddd_ddd)(double, double, double)
    """
# Seperate but equivalent imports and definitions in pure Python and Cython
if not cython.compiled:
    # Mathematical functions
    from numpy import pi, sqrt, exp, sin, log
    from scipy.special import erfc
    # Import the units module
    import units
    # Import all user specified constants
    from params import *
else:
    # Lines in triple quotes will be executed in .pyx files.
    """
    # Mathematical functions
    from libc.math cimport M_PI as pi
    from libc.math cimport sqrt, exp, sin, log, erfc
    # Import the units module
    cimport units
    # Import all user specified constants 
    from params cimport *
    """

#####################################
# Global (module level) allocations #
#####################################
# Useful for temporary storage of 3D vector
cython.declare(vector='double*',
               vector_mw='double[::1]',
               )
vector = malloc(3*sizeof('double'))
vector_mw = cast(vector, 'double[:3]')

################
# Pure numbers #
################
cython.declare(minus_4pi='double',
               one_third='double',
               sqrt_pi='double',
               two_pi='double',
               )
minus_4pi = -4*pi
one_third = 1.0/3.0
sqrt_pi = sqrt(pi)
two_pi = 2*pi

############################################
# Derived and internally defined constants #
############################################
cython.declare(G_Newton='double',
               PM_gridsize3='ptrdiff_t',
               boxsize2='double',
               ewald_file='str',
               machine_ϵ='double',
               softening='double',
               softening2='double',
               two_ewald_gridsize='int',
               two_machine_ϵ='double',
               use_PM='bint',
               ϱ='double',
               )
G_Newton = 6.6738e-11*units.m**3/units.kg/units.s**2  # Newtons constant
ϱ = 3*H0**2/(8*pi*G_Newton) # The average, comoing density (the critical comoving density since we only study flat universes)
softening = 300*units.kpc #0.02*boxsize/(8000**one_third) #(boxsize/30)*2000**(-one_third)  # 2000 should be the particle Number. Source: http://popia.ft.uam.es/aknebe/page3/files/ComputationalAstrophysics/PhysicalProcesses.pdf page 85. Or maybe use 2-4% of the mean-interparticle distance (V/N)**(1/3), http://www.ast.cam.ac.uk/~puchwein/NumericalCosmology02.pdf page 13.

PM_gridsize3 = PM_gridsize**3
boxsize2 = boxsize**2
ewald_file = '.ewald_gridsize=' + str(ewald_gridsize) + '.hdf5'  # Name of file storing the Ewald grid
machine_ϵ = np.finfo('float64').eps  # Machine epsilon
softening2 = softening**2
two_ewald_gridsize = 2*ewald_gridsize
two_machine_ϵ = 2*machine_ϵ
use_PM = True  # Flag specifying whether the PM method is used or not. THIS SHOULD BE COMPUTED BASED ON PARTICLES CHOSEN IN THE PARAMETER FILE!!!!!!!!!!!

#############
# MPI setup #
#############
from mpi4py import MPI
cython.declare(master='bint',
               nprocs='int',
               rank='int',
               )
# Functions for communication
comm = MPI.COMM_WORLD
Abort = comm.Abort
Allgather = comm.Allgather
Allgatherv = comm.Allgatherv
Allreduce = comm.Allreduce
Barrier = comm.Barrier
Bcast = comm.Bcast
Gather = comm.Gather
Gatherv = comm.Gatherv
Reduce = comm.Reduce
Scatter = comm.Scatter
Sendrecv = comm.Sendrecv
allreduce = comm.allreduce
reduce = comm.reduce
sendrecv = comm.sendrecv
# Constants
nprocs = comm.size  # Number of processes started with mpiexec
rank = comm.rank    # The unique rank of the running process
master = not rank   # Flag identifying the master/root process (that which have rank 0)
# Function for easily partitioning of multidimensional arrays
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


####################
# Useful functions #
####################
# Function for printing warnings
def warn(msg):
    os.system('printf "\033[1m\033[91mWarning: ' + msg + '\033[0m\n" >&2')


###########################################
# Absolute paths to directories and files #
###########################################
# The paths are stored in the top_dir/.paths file
import imp
cython.declare(paths='dict')
top_dir = '.'
ls_prev = []
possible_root_dir = 0
while True:
    ls = os.listdir(top_dir)
    possible_root_dir = (possible_root_dir + 1) if ls == ls_prev else 0
    if possible_root_dir == 3:  # 3 ../ and still the same files. "Must" be /.
        raise Exception('Cannot find the .paths file!')
    if '.paths' in ls:
        break
    top_dir = '../' +  top_dir
    ls_prev = ls
paths_module = imp.load_source('paths', top_dir + '/.paths')
paths = paths_module.__dict__

