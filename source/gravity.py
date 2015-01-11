# Import everything from the _params_active module (including cython).
# In the .pyx file, this line will be replaced by the content of _params_active.py itself 
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from ewald import ewald
    from mesh import CIC_coordinates2grid
    from species import Particles
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from ewald cimport ewald
    from mesh cimport CIC_coordinates2grid
    # Particles cimported from species in the .pxd
    # FFTW functionality from fft.c
    cdef extern from "fft.c":
        # The fftw_plan type
        ctypedef struct fftw_plan_struct:
            pass
        ctypedef fftw_plan_struct *fftw_plan
        # The returned struct of fftw_setup
        struct fftw_return_struct:
            double* grid
            ptrdiff_t gridsize_local_x
            ptrdiff_t gridsize_local_y
            ptrdiff_t gridstart_local_x
            ptrdiff_t gridstart_local_y
            fftw_plan plan_forward
            fftw_plan plan_backward
        # Functions
        fftw_return_struct fftw_setup(ptrdiff_t gridsize_x, ptrdiff_t gridsize_y, ptrdiff_t gridsize_z)
        void fftw_execute(fftw_plan plan)
        void fftw_clean(double* grid, fftw_plan plan_forward, fftw_plan plan_backward)
    """


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
    N = particles.N
    # Direct gravity solver
    for i in range(0, N - 1):
        xi = posx[i]
        yi = posy[i]
        zi = posz[i]
        for j in range(i + 1, N):
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


from numpy.fft import rfftn

# Function for computing the gravitational force by direct summation on all particles
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               )
def PM(particles):
    """ This function updates the velocities of all particles via the
    particle-mesh (PM) method. 
    """

    # Reset the mesh
    PM_grid[...] = 0
    # Interpolate particle masses to meshpoints
    CIC_coordinates2grid(PM_grid, particles)
    # Fourier transform the mesh
    #PM_grid = rfftn(PM_grid)
    # Operations in Fourier space
    


    # multiply by the Greens function and the
    # short-range cutoff factor and do a double deconvolution (one for the
    # mesh mass assignment and one for the upcoming particle force assignment)

    # Fourier transform back to coordinate space. Now the mesh stores potential values

    # Get the forces at the meshpoints via finite differences

    # Interpolate forces from the mesh points to the particles

    # Add in short range force contributions via the PP method




# Initialize the PM mesh at import time
cython.declare(PM_grid='double[:, :, ::1]',
               PM_gridsize_padding='ptrdiff_t',
               fftw_struct='fftw_return_struct',
               PM_gridsize_local_x='ptrdiff_t',
               PM_gridsize_local_y='ptrdiff_t',
               PM_gridstart_local_x='ptrdiff_t',
               PM_gridstart_local_y='ptrdiff_t',
               plan_forward='fftw_plan',
               plan_backward='fftw_plan',
               )
PM_gridsize_padding = 2*(PM_gridsize//2 + 1)
# Initialize fftw_mpi, allocate the grid, initialize the
# local grid sizes and start indices and do FFTW planning.
fftw_struct = fftw_setup(PM_gridsize, PM_gridsize, PM_gridsize)
# Unpack fftw_struct
PM_grid = <double[:PM_gridsize, :PM_gridsize, :PM_gridsize_padding]> fftw_struct.grid
PM_gridsize_local_x = fftw_struct.gridsize_local_x
PM_gridsize_local_y = fftw_struct.gridsize_local_y
PM_gridstart_local_x = fftw_struct.gridstart_local_x
PM_gridstart_local_y = fftw_struct.gridstart_local_y
plan_forward  = fftw_struct.plan_forward
plan_backward = fftw_struct.plan_backward
#from time import time
# Do forward FFT
#t0 = time()
#fftw_execute(plan_forward)
#print('fft time:', time() - t0)
# Done performing FFT's. Cleanup
#fftw_clean(cython.address(PM_grid[0, 0, 0]), plan_forward, plan_backward)













