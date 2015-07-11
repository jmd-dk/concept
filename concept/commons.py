# This module contains imports, Cython declarations and values
# of parameters common to all other modules. Each module should have
# 'from commons import *' as its first statement.

############################################
# Imports common to pure Python and Cython #
############################################
from __future__ import division  # Needed for Python3 division in Cython
from numpy import (arange, array, asarray, concatenate, cumsum, ceil, delete,
                   empty, linspace, ones, trapz, unravel_index, zeros)
from numpy.random import random
import numpy as np
import h5py
import os
import sys

# For development purposes only
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
    for directive in ('boundscheck',
                      'cdivision',
                      'initializedcheck',
                      'wraparound',
                      ):
        setattr(cython, directive, dummy_function)
    # Dummy Cython functions
    for func in ('address', ):
        setattr(cython, func, lambda _: _)
    # C allocation syntax for memory management
    def sizeof(dtype):
        # C dtype names to Numpy dtype names
        if dtype == 'int':
            dtype = 'int32'
        elif dtype == 'double':
            dtype = 'float64'
        elif dtype == 'size_t':
            dtype = 'uintp'
        elif dtype in ('func_b_ddd', 'func_d_dd', 'func_d_ddd', 'func_ddd_ddd'):
            dtype='object'
        elif dtype[-1] == '*':
            # Allocate pointer array of pointers (eg. int**).
            # Emulate these as lists of arrays.
            return [empty(1, dtype=sizeof(dtype[:-1]).dtype)]
        else:
            raise TypeError(dtype + ' not implemented as a Numpy dtype in commons.py')
        return array([1], dtype=dtype)
    def malloc(a):
        if isinstance(a, list):
            # Pointer to pointer represented as list of arrays
            return a
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
    # Dummy fused types
    number = number2 = integer = floating = []
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
    ctypedef bint    (*func_b_ddd)  (double, double, double)
    ctypedef double  (*func_d_dd)   (double, double)
    ctypedef double  (*func_d_ddd)  (double, double, double)
    ctypedef double* (*func_ddd_ddd)(double, double, double)
    # Create a fused number type containing all necessary numerical types
    ctypedef fused number:
        cython.int
        cython.size_t
        cython.float
        cython.double
    # Create another fused number type, so that function arguments can have
    # different specializations.
    ctypedef fused number2:
        cython.int
        cython.size_t
        cython.float
        cython.double
    # Create integer and floating fused types
    ctypedef fused integer:
        cython.int
        cython.size_t
    ctypedef fused floating:
        cython.float
        cython.double
    # Custom classes
    from species cimport Particles
    from IO cimport Gadget_snapshot
    """
# Seperate but equivalent imports and definitions in pure Python and Cython
if not cython.compiled:
    # Mathematical constants and functions
    from numpy import pi as π
    from numpy import sqrt, exp, sin, log
    from scipy.special import erfc
    # Import the units module
    import units
    # Import all user specified constants
    from params import *
else:
    # Lines in triple quotes will be executed in .pyx files.
    """
    # Mathematical constants and functions
    from libc.math cimport M_PI as π
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
               vector_mv='double[::1]',
               )
vector = malloc(3*sizeof('double'))
vector_mv = cast(vector, 'double[:3]')

################
# Pure numbers #
################
cython.declare(minus_4π='double',
               one_third='double',
               one_twelfth='double',
               sqrt_π='double',
               two_π='double',
               two_thirds='double',
               )
minus_4π = -4*π
one_third = 1.0/3.0
one_twelfth = 1.0/12.0
sqrt_π = sqrt(π)
two_thirds = 2.0/3.0
two_π = 2*π

############################################
# Derived and internally defined constants #
############################################
cython.declare(a_max='double',
               G_Newton='double',
               PM_gridsize3='ptrdiff_t',
               PM_gridsize_padding='ptrdiff_t',
               boxsize2='double',
               boxsize3='double',
               ewald_file='str',
               half_PM_gridsize='ptrdiff_t',
               half_PM_gridsize_padding='ptrdiff_t',
               half_boxsize='double',
               machine_ϵ='double',
               minus_half_boxsize='double',
               two_ewald_gridsize='int',
               two_machine_ϵ='double',
               two_recp_boxsize='double',
               use_Ewald ='bint',
               use_PM='bint',
               recp_boxsize2='double',
               ϱ='double',
               ϱm='double',
               PM_fac_const='double',
               longrange_exponent_fac='double',
               P3M_cutoff_phys='double',
               P3M_scale_phys='double',
               π_recp_PM_gridsize='double',
               )
G_Newton = 6.6738e-11*units.m**3/units.kg/units.s**2  # Newtons constant
if len(snapshot_times + powerspec_times) == 0:
    a_max = a_begin
else:
    a_max = np.max(snapshot_times + powerspec_times)  # The scale factor at the last time step
ϱ = 3*H0**2/(8*π*G_Newton) # The average, comoing density (the critical comoving density since we only study flat universes)
ϱm = Ωm*ϱ  # The average, comoving matter density
PM_gridsize3 = PM_gridsize**3
PM_gridsize_padding = 2*(PM_gridsize//2 + 1)
half_PM_gridsize = PM_gridsize//2
half_PM_gridsize_padding = PM_gridsize_padding//2
boxsize2 = boxsize**2
boxsize3 = boxsize**3
recp_boxsize2 = 1/boxsize2
half_boxsize = 0.5*boxsize
minus_half_boxsize = -half_boxsize
two_recp_boxsize = 2/boxsize
π_recp_PM_gridsize = π/PM_gridsize
ewald_file = '.ewald_gridsize=' + str(ewald_gridsize) + '.hdf5'  # Name of file storing the Ewald grid
machine_ϵ = np.finfo('float64').eps  # Machine epsilon
two_ewald_gridsize = 2*ewald_gridsize
two_machine_ϵ = 2*machine_ϵ
use_PM = False  # Flag specifying whether the PM method is used or not
if len(powerspec_times) > 0:
    use_PM = True
else:
    for kick_algorithm in kick_algorithms.values():
        if kick_algorithm in ('PM', 'P3M'):
            use_PM = True  
            break
# All constant factors across the PM scheme is gathered in the PM_fac
# variable. It's contributions are:
# For CIC interpolating particle masses/volume to the grid points:
#     particles.mass/(boxsize/PM_gridsize)**3
# Factor in the Greens function:
#     -4*π*G_Newton/((2*π/((boxsize/PM_gridsize)*PM_gridsize))**2)   
# From finite differencing to get the forces:
#     -PM_gridsize/boxsize
# For converting acceleration to momentum
#     particles.mass*Δt
# Everything except the mass and the time are constant, and is condensed
# into the PM_fac_const variable.
PM_fac_const = G_Newton*PM_gridsize**4/(π*boxsize**2)
# The exponential cutoff for the long-range force looks like exp(-k2*rs2).
# In the code, the wave vector is in grid units in stead of radians. The
# conversion is this 2*π/PM_gridsize. The total factor on k2 in the
# exponential is then
longrange_exponent_fac = -(2*π/PM_gridsize*P3M_scale)**2
# The short-range/long-range force scale
P3M_scale_phys = P3M_scale*boxsize/PM_gridsize
# Particles within this distance to the surface of the domain should
# interact with particles in the neighboring domain via the shortrange
# force, when the P3M algorithm is used.
P3M_cutoff_phys = P3M_scale_phys*P3M_cutoff



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
Isend = comm.Isend
Reduce = comm.Reduce
Recv = comm.Recv
Scatter = comm.Scatter
Send = comm.Send
Sendrecv = comm.Sendrecv
allreduce = comm.allreduce
reduce = comm.reduce
sendrecv = comm.sendrecv
# Constants
nprocs = comm.size  # Number of processes started with mpiexec
rank = comm.rank    # The unique rank of the running process
master = not rank   # Flag identifying the master/root process (that which have rank 0)


####################
# Useful functions #
####################
# Max function for 1D memory views of numbers
if not cython.compiled:
    # Pure Python already have a generic max function
    pass
else:
    """
    @cython.cfunc
    @cython.inline
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.returns(number)
    def max(number[::1] a):
        cdef:
            number m
            size_t N
            size_t i
        N = a.shape[0]
        m = a[0]
        for i in range(1, N):
            if a[i] > m:
                m = a[i]
        return m
    """

# Min function for 1D memory views of numbers
if not cython.compiled:
    # Pure Python already have a generic min function
    pass
else:
    """
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.returns(number)
    def min(number[::1] a):
        cdef:
            number m
            size_t N
            size_t i
        N = a.shape[0]
        m = a[0]
        for i in range(1, N):
            if a[i] < m:
                m = a[i]
        return m
    """

# Modulo function for numbers
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               x=number,
               length=number2,
               )
@cython.returns(number)
def mod(x, length):
    # mod(integer, floating) not possible. Note that no error will occur if
    # called with illegal types!
    if not (number in integer and number2 in floating):
        # Note that -length < x < 2*length must be true for this to work
        if x < 0:
            x += length
        elif x >= length:
            x -= length
        # A general prescription would be x = (x % length) + (x < 0)*length
        return x

# Sum function for 1D memory views of numbers
if not cython.compiled:
    # Pure Python already have a generic sum function
    pass
else:
    """
    @cython.cfunc
    @cython.inline
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.returns(number)
    def sum(number[::1] a):
        cdef:
            number Σ
            size_t N
            size_t i
        N = a.shape[0]
        if N == 0:
            return 0
        Σ = a[0]
        for i in range(1, N):
            Σ += a[i]
        return Σ
    """

# Prod function for 1D memory views of numbers
if not cython.compiled:
    # Utilize the prod function from numpy for pure Python
    prod = np.prod
else:
    """
    @cython.cfunc
    @cython.inline
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.returns(number)
    def prod(number[::1] a):
        cdef:
            number Π
            size_t N
            size_t i
        N = a.shape[0]
        if N == 0:
            return 1
        Π = a[0]
        for i in range(1, N):
            Π *= a[i]
        return Π
    """

# Unnormalized sinc function (faster than gsl_sf_sinc)
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(x='double',
               y='double')
@cython.returns('double')
def sinc(x):
    y = sin(x)
    if y != x:
        y /= x
    else:
        y = 1
    return y

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
    top_dir = '../' + top_dir
    ls_prev = ls
paths_module = imp.load_source('paths', top_dir + '/.paths')
paths = paths_module.__dict__
