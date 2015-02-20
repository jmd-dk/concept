# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from ewald import ewald
    from communication import find_N_recv
    from mesh import CIC_coordinates2grid
    # FFT functionality via Numpy
    #from scipy.fftpack import fftn, ifftn
    from numpy.fft import rfftn, irfftn
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from ewald cimport ewald
    from communication cimport find_N_recv
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
        fftw_return_struct fftw_setup(ptrdiff_t gridsize_x,
                                      ptrdiff_t gridsize_y,
                                      ptrdiff_t gridsize_z)
        void fftw_execute(fftw_plan plan)
        void fftw_clean(double* grid, fftw_plan plan_forward,
                                      fftw_plan plan_backward)
    """


# Function for direct summation of gravitational forces between particles
# in two domains.
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_i='double[::1]',
               posy_i='double[::1]',
               posz_i='double[::1]',
               momx_i='double[::1]',
               momy_i='double[::1]',
               momz_i='double[::1]',
               mass_i='double',
               N_local_i='size_t',
               posx_j='double[::1]',
               posy_j='double[::1]',
               posz_j='double[::1]',
               mass_j='double',
               N_local_j='size_t',
               Δt='double',
               flag_input='int',
               # Locals
               dim='int',
               eom_factor='double',
               force='double*',
               i='size_t',
               j='size_t',
               r3='double',
               x='double',
               xi='double',
               y='double',
               yi='double',
               z='double',
               zi='double',
               Δmomx_j='double[::1]',
               Δmomy_j='double[::1]',
               Δmomz_j='double[::1]',
               )
def direct_summation(posx_i, posy_i, posz_i, momx_i, momy_i, momz_i,
                     mass_i, N_local_i,
                     posx_j, posy_j, posz_j, Δmomx_j, Δmomy_j, Δmomz_j,
                     mass_j, N_local_j,
                     Δt, flag_input=0):
    """This function takes in positions and momenta of particles located in
    the domain designated the calling process, as well as positions and
    preallocated nullified momentum changes for particles located in another
    domian.
    The two sets of particles are denoted i and j. The function computes the
    momentum changes due to gravity via direct summation. The two sets of
    particles can be the same, which is signalled by flag_input=0. That is,
    this function can also be used to compute interactions within a single
    domain. Use flag_input=1 when using two different domains. Here, set i
    should be the particles belonging to the caller process. For these,
    momentum changes are added to the momenta. For set j, the momentum
    changes are computed but not added to the momenta, as these reside on a
    different process. Use flag_input=2 to skip the computation of the
    momentum changes of set j.
    Note that the time step size Δt is really ∫_t^(t + Δt) dt/a.
    """
  
    # The factor (G*m_i*m_j*∫_t^(t + Δt) dt/a) in the
    # comoving equations of motion
    # p_i --> p_i + ∫_t^(t + Δt) F/a*dt = p_i + m_i*F*∫_t^(t + Δt) dt/a
    #       = p_i + (-G*m_i*m_j/r**2)*∫_t^(t + Δt) dt/a
    #       = p_i - 1/r**2*(G*m_i*m_j*∫_t^(t + Δt) dt/a)
    eom_factor = G_Newton*mass_i*mass_j*Δt
    # Direct summation
    for i in range(0, N_local_i if (flag_input > 0) else (N_local_i - 1)):
        xi = posx_i[i]
        yi = posy_i[i]
        zi = posz_i[i]
        for j in range(0 if (flag_input > 0) else (i + 1), N_local_j):
            x = posx_j[j] - xi
            y = posy_j[j] - yi
            z = posz_j[j] - zi
            # The Ewald correction force
            force = ewald(x, y, z)
            # Add in the force from the actual particle
            r3 = (x**2 + y**2 + z**2 + softening2)**1.5
            force[0] -= x/r3
            force[1] -= y/r3
            force[2] -= z/r3
            # Multiply the force by (G*m_i*m_j*∫_t^(t + Δt) dt/a).
            # Note that "force" is now really the momentum change
            force[0] *= eom_factor
            force[1] *= eom_factor
            force[2] *= eom_factor
            # Update momenta for group i and momentum changes for group j
            if flag_input == 0:
                # Group i and j are the same.
                # Update momenta of both particles in the pair
                momx_i[i] -= force[0]
                momy_i[i] -= force[1]
                momz_i[i] -= force[2]
                momx_i[j] += force[0]
                momy_i[j] += force[1]
                momz_i[j] += force[2]
            else:
                # Group i and j are different.
                # Update local momenta
                momx_i[i] -= force[0]
                momy_i[i] -= force[1]
                momz_i[i] -= force[2]
                if flag_input == 1:
                   # Also update external momentum changes
                   Δmomx_j[j] += force[0]
                   Δmomy_j[j] += force[1]
                   Δmomz_j[j] += force[2]

# Function for computing the gravitational force
# by direct summation on all particles
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               Δt='double',
               # Locals
               ID_recv='int',
               ID_send='int',
               N_extrn='size_t',
               N_extrn_max='size_t',
               N_extrns='size_t[::1]',
               N_local='size_t',
               N_partnerproc_pairs='int',
               even_nprocs='bint',
               factor='double',
               i='size_t',
               j='int',
               mass='double',
               posx_extrn='double[::1]',
               posx_local='double[::1]',
               posy_extrn='double[::1]',
               posy_local='double[::1]',
               posz_extrn='double[::1]',
               posz_local='double[::1]',
               momx_local='double[::1]',
               momy_local='double[::1]',
               momz_local='double[::1]',
               Δmomx_extrn='double[::1]',
               Δmomx_local='double[::1]',
               Δmomy_extrn='double[::1]',
               Δmomy_local='double[::1]',
               Δmomz_extrn='double[::1]',
               Δmomz_local='double[::1]',
               )
def PP(particles, Δt):
    """ This function updates the momenta of all particles via the
    particle-particle (PP) method.
    Note that the time step size Δt is really ∫_t^(t + Δt) dt/a.
    """

    # Extract variables from particles
    posx_local = particles.posx_mw
    posy_local = particles.posy_mw
    posz_local = particles.posz_mw
    momx_local = particles.momx_mw
    momy_local = particles.momy_mw
    momz_local = particles.momz_mw
    mass = particles.mass
    N_local = particles.N_local


    # DET MÆRKELIGE LED PROPORTIONALT MED x
    factor = 0.5*H0**2*mass*Δt
    for i in range(N_local):
        momx_local[i] += factor*posx_local[i]
        momy_local[i] += factor*posy_local[i]
        momz_local[i] += factor*posz_local[i]






    # Update local momenta due to forces between local particles.
    # Note that vector_mw is not actually used due to flag_input=0
    direct_summation(posx_local, posy_local, posz_local,
                     momx_local, momy_local, momz_local,
                     mass, N_local,
                     posx_local, posy_local, posz_local,
                     vector_mw, vector_mw, vector_mw,
                     mass, N_local,
                     Δt)
    # All work done if only one domain exists
    if nprocs == 1:
        return
    # Update local momenta and compute and send external momentum changes
    # due to forces between local and external particles.
    # Find out how many particles will be recieved from each process
    N_extrns = find_N_recv(array([N_local], dtype='uintp'))
    N_extrn_max = N_extrns[rank]
    # Allocate position recieve buffers and momentum change buffers
    posx_extrn = empty(N_extrn_max)
    posy_extrn = empty(N_extrn_max)
    posz_extrn = empty(N_extrn_max)
    Δmomx_local = empty(N_local)
    Δmomy_local = empty(N_local)
    Δmomz_local = empty(N_local)
    Δmomx_extrn = zeros(N_extrn_max)
    Δmomy_extrn = zeros(N_extrn_max)
    Δmomz_extrn = zeros(N_extrn_max)
    # Number of pairs of process partners to send/recieve data to/from
    even_nprocs = not (nprocs%2)
    flag_input = 1
    N_partnerproc_pairs = 1 + nprocs//2
    N_partnerproc_pairs_minus_1 = N_partnerproc_pairs - 1
    for j in range(1, N_partnerproc_pairs):
        # Process ranks to send/recieve to/from
        ID_send = ((rank + j)%nprocs)
        ID_recv = ((rank - j)%nprocs)
        N_extrn = N_extrns[ID_recv]
        # Send and recieve positions
        Sendrecv(posx_local[:N_local], dest=ID_send,
                 recvbuf=posx_extrn, source=ID_recv)
        Sendrecv(posy_local[:N_local], dest=ID_send,
                 recvbuf=posy_extrn, source=ID_recv)
        Sendrecv(posz_local[:N_local], dest=ID_send,
                 recvbuf=posz_extrn, source=ID_recv)
        # In the end in the case of even nprocs, a single (not a pair)
        # process remains. Flag to compute momenta for local particles
        # only, as the results will not be sent afterwards.
        if even_nprocs and j == N_partnerproc_pairs_minus_1:
            flag_input = 2
        # Do direct summation between local and external particles
        direct_summation(posx_local, posy_local, posz_local,
                         momx_local, momy_local, momz_local,
                         mass, N_local,
                         posx_extrn, posy_extrn, posz_extrn,
                         Δmomx_extrn, Δmomy_extrn, Δmomz_extrn,
                         mass, N_extrn,
                         Δt, flag_input=flag_input)
        # When flag_input == 2, no momentum updates has been computed.
        # Do not sent or recieve these noncomputed updates.
        if flag_input == 2:
            return
        # Send momentum updates back to the process from which positions
        # were recieved. Recieve momentum updates from the process which
        # the local positions were send to.
        Sendrecv(Δmomx_extrn[:N_extrn], dest=ID_recv,
                 recvbuf=Δmomx_local, source=ID_send)
        Sendrecv(Δmomy_extrn[:N_extrn], dest=ID_recv,
                 recvbuf=Δmomy_local, source=ID_send)
        Sendrecv(Δmomz_extrn[:N_extrn], dest=ID_recv,
                 recvbuf=Δmomz_local, source=ID_send)
        # Apply local momentum updates recieved from other process
        for i in range(N_local):
            momx_local[i] += Δmomx_local[i]
            momy_local[i] += Δmomy_local[i]
            momz_local[i] += Δmomz_local[i]
        # Reset external momentum change buffers
        if j != N_partnerproc_pairs_minus_1:
            for i in range(N_extrns[((rank - j - 1)%nprocs)]):
                Δmomx_extrn[i] = 0
                Δmomy_extrn[i] = 0
                Δmomz_extrn[i] = 0

# Function for computing the gravitational force by the particle mesh method
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               Δt='double',
               # Locals
               i='ptrdiff_t',
               j='ptrdiff_t',
               k='ptrdiff_t',
               )
def PM(particles, Δt):
    """ This function updates the momenta of all particles via the
    particle-mesh (PM) method.
    Note that the time step size Δt is really ∫_t^(t + Δt) dt/a.
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
