# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from ewald import ewald
    from mesh import CIC_coordinates2grid
    # FFT functionality via Numpy
    #from scipy.fftpack import fftn, ifftn
    from numpy.fft import rfftn, irfftn
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from ewald cimport ewald
    from mesh cimport CIC_coordinates2grid
    # FFT functionality via FFTW from fft.c
    cdef extern from "fft.c":
        # The fftw_plan type
        ctypedef struct fftw_plan_struct:
            pass
        ctypedef fftw_plan_struct *fftw_plan
        # The returned struct of fftw_setup
        struct fftw_return_struct:
            ptrdiff_t gridsize_local_x
            ptrdiff_t gridsize_local_y
            ptrdiff_t gridstart_local_x
            ptrdiff_t gridstart_local_y
            double* grid
            fftw_plan plan_forward
            fftw_plan plan_backward
        # Functions
        fftw_return_struct fftw_setup(ptrdiff_t gridsize_x, ptrdiff_t gridsize_y, ptrdiff_t gridsize_z)
        void fftw_execute(fftw_plan plan)
        void fftw_clean(double* grid, fftw_plan plan_forward, fftw_plan plan_backward)
    """

import numpy as np

# Function for computing the gravitational force by direct summation on all particles
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               # Locals
               force_factor='double',
               posx='double*',
               posy='double*',
               posz='double*',
               velx='double*',
               vely='double*',
               velz='double*',
               N='size_t',
               i='size_t',
               xi='double',
               yi='double',
               zi='double',
               j='size_t',
               x='double',
               y='double',
               z='double',
               force='double*',
               r3='double',
               dim='size_t',
               )
def PP(particles):
    """ This function updates the velocities of all particles via the
    particle-particle (PP) method. 
    """

    # Extract variables
    force_factor = G_Newton*particles.mass*dt
    posx = particles.posx
    posy = particles.posy
    posz = particles.posz
    velx = particles.velx
    vely = particles.vely
    velz = particles.velz
    N_local = particles.N_local
    # Direct gravity solver
    for i in range(0, N_local - 1):
        xi = posx[i]
        yi = posy[i]
        zi = posz[i]
        for j in range(i + 1, N_local):
            x = posx[j] - xi
            y = posy[j] - yi
            z = posz[j] - zi
            # The Ewald correction force
            force = ewald(x, y, z)
            # Add in the force from the actual particle
            r3 = (x**2 + y**2 + z**2 + softening2)**1.5
            force[0] -= x/r3
            force[1] -= y/r3
            force[2] -= z/r3
            # Multiply by G*M*m*dt to get the velocity change
            for dim in range(3):
                force[dim] *= force_factor
            # Update velocities
            velx[i] -= force[0]
            vely[i] -= force[1]
            velz[i] -= force[2]
            velx[j] += force[0]
            vely[j] += force[1]
            velz[j] += force[2]


# Function for computing the gravitational force by direct summation on all particles
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               # Locals
               i='ptrdiff_t',
               j='ptrdiff_t',
               k='ptrdiff_t',
               )
def PM(particles):
    """ This function updates the velocities of all particles via the
    particle-mesh (PM) method. 
    """
    global PM_grid

    # Reset the mesh
    PM_grid[...] = 0
             
    # Interpolate particle masses to meshpoints
    #CIC_coordinates2grid(PM_grid, particles)
 
    # Fourier transform the grid forwards to Fourier space
    fftw_execute(plan_forward)


    #for i in range(PM_gridsize):
    #    for j in range(PM_gridsize_local_y):
    #        for k in range(PM_gridsize):
    #            PM_grid[j, i, k] *= i**2 + (j + PM_gridstart_local_y)**2 + k**2

    # Fourier transform the grid back to real space
    fftw_execute(plan_backward)

    for i in range(PM_gridsize_local_x):
        for j in range(PM_gridsize):
            for k in range(PM_gridsize):
                PM_grid[i, j, k] /= PM_gridsize3
    


    # multiply by the Greens function and the
    # short-range cutoff factor and do a double deconvolution (one for the
    # mesh mass assignment and one for the upcoming particle force assignment)

    # Fourier transform the grid back to coordinate space.
    # Now the grid stores potential values


    # Get the forces at the meshpoints via finite differences

    # Interpolate forces from the mesh points to the particles

    # Add in short range force contributions via the PP method


    # Done performing FFT's. Cleanup
    #fftw_clean(cython.address(PM_grid[0, 0, 0]), plan_forward, plan_backward)


# Initializes the PM mesh at import time, if the PM method is to be used
if use_PM:
    if not cython.compiled:
        # Initialization of the PM mesh in pure Python.
        PM_gridsize_local_x = PM_gridsize_local_y = int(PM_gridsize/nprocs)
        if PM_gridsize_local_x != PM_gridsize/nprocs:
            raise ValueError('The PM method in pure Python mode only works when\n' + 
                            'PM_gridsize is divisible by the number of processes!')
        PM_gridstart_local_x = PM_gridstart_local_y = PM_gridsize_local_x*rank
        PM_gridsize_padding = 2*(PM_gridsize//2 + 1)
        PM_grid = empty((PM_gridsize_local_x, PM_gridsize, PM_gridsize_padding))
        # The output of the following function is formatted just
        # like that of the MPI implementation of FFTW.
        plan_backward = 'plan_backward'
        plan_forward = 'plan_forward'
        def fftw_execute(plan):
            global PM_grid
            # The pure Python FFT implementation is serial. Every process
            # computes the entire FFT, of the temporary varaible PM_grid_global.
            PM_grid_global = empty((PM_gridsize, PM_gridsize, PM_gridsize_padding))
            Allgatherv(PM_grid, PM_grid_global)
            if plan == plan_forward:
                # Delete the padding on last dimension
                for i in range(PM_gridsize_padding - PM_gridsize):
                    PM_grid_global = delete(PM_grid_global, -1, axis=2)
                # Do real transform
                PM_grid_global = rfftn(PM_grid_global)
                # FFTW transposes the first two dimensions
                PM_grid_global = PM_grid_global.transpose([1, 0, 2])
                # FFTW represents the complex array by doubles only
                tmp = empty((PM_gridsize, PM_gridsize, PM_gridsize_padding))
                for i in range(PM_gridsize_padding):
                    if i % 2:
                        tmp[:, :, i] = PM_grid_global.imag[:, :, i//2]
                    else:
                        tmp[:, :, i] = PM_grid_global.real[:, :, i//2]
                PM_grid_global = tmp
                # As in FFTW, distribute the slabs along the y-dimension
                # (which is the first dimension now, due to transposing)
                PM_grid = PM_grid_global[PM_gridstart_local_y:(PM_gridstart_local_y + PM_gridsize_local_y), :, :]
            elif plan == plan_backward:
                # FFTW represents the complex array by doubles only.
                # Go back to using complex entries
                tmp = zeros((PM_gridsize, PM_gridsize, PM_gridsize_padding/2),
                            dtype='complex128')
                for i in range(PM_gridsize_padding):
                    if i % 2:
                        tmp[:, :, i//2] += 1j*PM_grid_global[:, :, i]
                    else:
                        tmp[:, :, i//2] += PM_grid_global[:, :, i]
                PM_grid_global = tmp
                # FFTW transposes the first two dimensions back to normal
                PM_grid_global = PM_grid_global.transpose([1, 0, 2])
                # Do real inverse transform
                PM_grid_global = irfftn(PM_grid_global, s=[PM_gridsize]*3)
                # Remove the autoscaling provided by Numpy
                PM_grid_global[...] *= PM_gridsize3
                # Add padding on last dimension, as in FFTW
                padding = empty((PM_gridsize,
                                 PM_gridsize,
                                 PM_gridsize_padding - PM_gridsize,
                                 ))
                PM_grid_global = concatenate((PM_grid_global, padding), axis=2)
                # As in FFTW, distribute the slabs along the x-dimension
                PM_grid = PM_grid_global[PM_gridstart_local_x:(PM_gridstart_local_x + PM_gridsize_local_x), :, :]
    else:
        """
        # Initialization of the PM mesh in Cython       
        cython.declare(fftw_struct='fftw_return_struct',
                       PM_gridsize_padding='ptrdiff_t',
                       PM_gridsize_local_x='ptrdiff_t',
                       PM_gridsize_local_y='ptrdiff_t',
                       PM_gridstart_local_x='ptrdiff_t',
                       PM_gridstart_local_y='ptrdiff_t',
                       PM_grid='double[:, :, ::1]',
                       plan_forward='fftw_plan',
                       plan_backward='fftw_plan',
                       )
        # Initialize fftw_mpi, allocate the grid, initialize the
        # local grid sizes and start indices and do FFTW planning.
        fftw_struct = fftw_setup(PM_gridsize, PM_gridsize, PM_gridsize)
        # Unpack fftw_struct
        PM_gridsize_padding = 2*(PM_gridsize//2 + 1)
        PM_gridsize_local_x = fftw_struct.gridsize_local_x
        PM_gridsize_local_y = fftw_struct.gridsize_local_y
        PM_gridstart_local_x = fftw_struct.gridstart_local_x
        PM_gridstart_local_y = fftw_struct.gridstart_local_y
        # Wrap a memryview around the grid. Loop as noted in fft.c,
        # but use PM_grid[i, j, k] when in real space and PM_grid[j, i, k] when in Fourier space
        if PM_gridsize_local_x > 0:
            PM_grid = <double[:PM_gridsize_local_x, :PM_gridsize, :PM_gridsize_padding]> fftw_struct.grid
        else:
            # The process do not participate in the FFT computations
            PM_grid = empty((0, PM_gridsize, PM_gridsize_padding))
        plan_forward  = fftw_struct.plan_forward
        plan_backward = fftw_struct.plan_backward
        """
