# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from ewald import ewald
    from communication import cutout_domains, find_N_recv, neighboring_ranks
    from mesh import CIC_grid2coordinates_scalar, CIC_grid2coordinates_vector, CIC_particles2grid, communicate_boundaries, communicate_ghosts, domain2PM, PM2domain
    # FFT functionality via Numpy
    from numpy.fft import rfftn, irfftn
else:
    # Lines in triple quotes will be executed in the .pyx file
    """
    from ewald cimport ewald
    from communication cimport cutout_domains, find_N_recv, neighboring_ranks
    from mesh cimport CIC_grid2coordinates_scalar, CIC_grid2coordinates_vector, CIC_particles2grid, communicate_boundaries, communicate_ghosts, domain2PM, PM2domain
    # FFT functionality via FFTW from fft.c
    cdef extern from "fft.c":
        # The fftw_plan type
        ctypedef struct fftw_plan_struct:
            pass
        ctypedef fftw_plan_struct *fftw_plan
        # The returned struct of fftw_setup
        struct fftw_return_struct:
            ptrdiff_t gridsize_local_i
            ptrdiff_t gridsize_local_j
            ptrdiff_t gridstart_local_i
            ptrdiff_t gridstart_local_j
            double* grid
            fftw_plan plan_forward
            fftw_plan plan_backward
        # Functions
        fftw_return_struct fftw_setup(ptrdiff_t gridsize_i,
                                      ptrdiff_t gridsize_j,
                                      ptrdiff_t gridsize_k)
        void fftw_execute(fftw_plan plan)
        void fftw_clean(double* grid, fftw_plan plan_forward,
                                      fftw_plan plan_backward)
    """



# Function for direct summation of gravitational forces between particles
# in two domains.
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_i='double*',
               posy_i='double*',
               posz_i='double*',
               momx_i='double*',
               momy_i='double*',
               momz_i='double*',
               mass_i='double',
               N_local_i='size_t',
               posx_j='double*',
               posy_j='double*',
               posz_j='double*',
               Δmomx_j='double*',
               Δmomy_j='double*',
               Δmomz_j='double*',
               mass_j='double',
               N_local_j='size_t',
               Δt='double',
               softening2='double',
               only_short_range='bint',
               flag_input='int',
               # Locals
               dim='int',
               eom_factor='double',
               force='double*',
               i='size_t',
               i_end='size_t',
               j='size_t',
               j_start='size_t',
               r_scaled='double',
               r3='double',
               shortrange_fac='double',
               x='double',
               xi='double',
               y='double',
               yi='double',
               z='double',
               zi='double',
               )
def direct_summation(posx_i, posy_i, posz_i, momx_i, momy_i, momz_i,
                     mass_i, N_local_i,
                     posx_j, posy_j, posz_j, Δmomx_j, Δmomy_j, Δmomz_j,
                     mass_j, N_local_j,
                     Δt, softening2, flag_input, only_short_range=False):
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
    # No interactions if either of the two sets of particles is empty
    if N_local_i == 0 or N_local_j == 0:
        return
    # If either of the two sets of particles is empty, no interactions occur
    # The factor (G*m_i*m_j*∫_t^(t + Δt) dt/a) in the
    # comoving equations of motion
    # p_i --> p_i + ∫_t^(t + Δt) F/a*dt = p_i + m_i*F*∫_t^(t + Δt) dt/a
    #       = p_i + (-G*m_i*m_j/r**2)*∫_t^(t + Δt) dt/a
    #       = p_i - 1/r**2*(G*m_i*m_j*∫_t^(t + Δt) dt/a)
    eom_factor = G_Newton*mass_i*mass_j*Δt
    # Direct summation
    force = vector
    i_end = N_local_i if flag_input > 0 else N_local_i - 1
    for i in range(i_end):
        xi = posx_i[i]
        yi = posy_i[i]
        zi = posz_i[i]
        j_start = 0 if flag_input > 0 else i + 1
        for j in range(j_start, N_local_j):
            x = posx_j[j] - xi
            y = posy_j[j] - yi
            z = posz_j[j] - zi
            # Evaluate the gravitational force in one of three ways: Just the
            # short range force, the total force with Ewald corrections or
            # the total force without Ewald corrections.
            if only_short_range:
                # Translate coordinates so they correspond to the nearest image
                if x > half_boxsize:
                    x -= boxsize
                elif x < minus_half_boxsize:
                    x += boxsize
                if y > half_boxsize:
                    y -= boxsize
                elif y < minus_half_boxsize:
                    y += boxsize
                if z > half_boxsize:
                    z -= boxsize
                elif z < minus_half_boxsize:
                    z += boxsize
                r = sqrt(x**2 + y**2 + z**2 + softening2)
                r3 = r**3
                r_scaled = r/P3M_scale_phys
                shortrange_fac = r_scaled/sqrt_π*exp(-0.25*r_scaled**2) + erfc(0.5*r_scaled)
                force[0] = -x/r3*shortrange_fac
                force[1] = -y/r3*shortrange_fac
                force[2] = -z/r3*shortrange_fac
            else:
                # Compute the gravitational force (corresponding to 1/r**2)
                if use_Ewald:
                    # Translate coordinates so they correspond to
                    # the nearest image
                    if x > half_boxsize:
                        x -= boxsize
                    elif x < minus_half_boxsize:
                        x += boxsize
                    if y > half_boxsize:
                        y -= boxsize
                    elif y < minus_half_boxsize:
                        y += boxsize
                    if z > half_boxsize:
                        z -= boxsize
                    elif z < minus_half_boxsize:
                        z += boxsize
                    # The Ewald correction force for all images except the
                    # nearest one, which may not be the actual particle.
                    force = ewald(x, y, z)
                    # Add in the force from the particle's nearest image
                    r3 = (x**2 + y**2 + z**2 + softening2)**1.5
                    force[0] -= x/r3
                    force[1] -= y/r3
                    force[2] -= z/r3
                else:
                    # The force from the actual particle,
                    # without periodic images
                    r3 = (x**2 + y**2 + z**2 + softening2)**1.5
                    force[0] = -x/r3
                    force[1] = -y/r3
                    force[2] = -z/r3
            # Multiply the force by (G*m_i*m_j*∫_t^(t + Δt) dt/a).
            # Note that "force" is now really the momentum change.
            force[0] *= eom_factor
            force[1] *= eom_factor
            force[2] *= eom_factor
            # Update momenta and momentum changes
            if flag_input == 0:
                # Group i and j are the same (and belongs to the local domain).
                # Update momenta of both particles in the pair.
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
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
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
               flag_input='int',
               i='size_t',
               j='int',
               mass='double',
               momx_local='double*',
               momy_local='double*',
               momz_local='double*',
               posx_local='double*',
               posx_local_mv='double[::1]',
               posy_local='double*',
               posy_local_mv='double[::1]',
               posz_local='double*',
               posz_local_mv='double[::1]',
               softening2='double',
               )
def PP(particles, Δt):
    """ This function updates the momenta of all particles via the
    particle-particle (PP) method.
    Note that the time step size Δt is really ∫_t^(t + Δt) dt/a.
    """
    global posx_extrn, posy_extrn, posz_extrn
    global Δmomx_local, Δmomy_local, Δmomz_local
    global Δmomx_extrn, Δmomy_extrn, Δmomz_extrn
    global posx_extrn_mv, posy_extrn_mv, posz_extrn_mv
    global Δmomx_local_mv, Δmomy_local_mv, Δmomz_local_mv
    global Δmomx_extrn_mv, Δmomy_extrn_mv, Δmomz_extrn_mv
    # Extract variables from particles
    N_local = particles.N_local
    mass = particles.mass
    momx_local = particles.momx
    momy_local = particles.momy
    momz_local = particles.momz
    posx_local = particles.posx
    posy_local = particles.posy
    posz_local = particles.posz
    posx_local_mv = particles.posx_mv
    posy_local_mv = particles.posy_mv
    posz_local_mv = particles.posz_mv
    softening2 = particles.softening**2
    # Update local momenta due to forces between local particles.
    # Note that "vector" is not actually used due to flag_input=0.
    direct_summation(posx_local, posy_local, posz_local,
                     momx_local, momy_local, momz_local,
                     mass, N_local,
                     posx_local, posy_local, posz_local,
                     vector, vector, vector,
                     mass, N_local,
                     Δt, softening2,
                     0)
    # All work done if only one domain exists (if run on a single process)
    if nprocs == 1:
        return
    # Update local momenta and compute and send external momentum changes
    # due to forces between local and external particles.
    # Find out how many particles will be recieved from each process
    N_extrns = find_N_recv(array([N_local], dtype='uintp'))
    N_extrn_max = N_extrns[rank]
    # Enlarges the buffers if necessary
    if posx_extrn_mv.shape[0] < N_extrn_max:
        posx_extrn = realloc(posx_extrn, N_extrn_max*sizeof('double'))
        posy_extrn = realloc(posy_extrn, N_extrn_max*sizeof('double'))
        posz_extrn = realloc(posz_extrn, N_extrn_max*sizeof('double'))
        posx_extrn_mv = cast(posx_extrn, 'double[:N_extrn_max]')
        posy_extrn_mv = cast(posy_extrn, 'double[:N_extrn_max]')
        posz_extrn_mv = cast(posz_extrn, 'double[:N_extrn_max]')
        Δmomx_extrn = realloc(Δmomx_extrn, N_extrn_max*sizeof('double'))
        Δmomy_extrn = realloc(Δmomy_extrn, N_extrn_max*sizeof('double'))
        Δmomz_extrn = realloc(Δmomz_extrn, N_extrn_max*sizeof('double'))
        Δmomx_extrn_mv = cast(Δmomx_extrn, 'double[:N_extrn_max]')
        Δmomy_extrn_mv = cast(Δmomy_extrn, 'double[:N_extrn_max]')
        Δmomz_extrn_mv = cast(Δmomz_extrn, 'double[:N_extrn_max]')
    if Δmomx_local_mv.shape[0] < N_local:
        Δmomx_local = realloc(Δmomx_local, N_local*sizeof('double'))
        Δmomy_local = realloc(Δmomy_local, N_local*sizeof('double'))
        Δmomz_local = realloc(Δmomz_local, N_local*sizeof('double'))
        Δmomx_local_mv = cast(Δmomx_local, 'double[:N_local]')
        Δmomy_local_mv = cast(Δmomy_local, 'double[:N_local]')
        Δmomz_local_mv = cast(Δmomz_local, 'double[:N_local]')
    # Nullifies the external momentum changes
    for i in range(N_extrn_max):
        Δmomx_extrn[i] = 0
        Δmomy_extrn[i] = 0
        Δmomz_extrn[i] = 0
    # Number of pairs of process partners to send/recieve data to/from
    even_nprocs = not (nprocs % 2)
    flag_input = 1
    N_partnerproc_pairs = 1 + nprocs//2
    N_partnerproc_pairs_minus_1 = N_partnerproc_pairs - 1
    for j in range(1, N_partnerproc_pairs):
        # Process ranks to send/recieve to/from
        ID_send = mod(rank + j, nprocs)
        ID_recv = mod(rank - j, nprocs)
        N_extrn = N_extrns[ID_recv]
        # Send and recieve positions
        Sendrecv(posx_local_mv[:N_local], dest=ID_send,
                 recvbuf=posx_extrn_mv, source=ID_recv)
        Sendrecv(posy_local_mv[:N_local], dest=ID_send,
                 recvbuf=posy_extrn_mv, source=ID_recv)
        Sendrecv(posz_local_mv[:N_local], dest=ID_send,
                 recvbuf=posz_extrn_mv, source=ID_recv)
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
                         Δt, softening2,
                         flag_input)
        # When flag_input == 2, no momentum updates has been computed.
        # Do not sent or recieve these noncomputed updates.
        if flag_input == 2:
            continue
        # Send momentum updates back to the process from which
        # positions were recieved. Recieve momentum updates from the
        # process which the local positions were send to.
        Sendrecv(Δmomx_extrn_mv[:N_extrn], dest=ID_recv,
                 recvbuf=Δmomx_local_mv, source=ID_send)
        Sendrecv(Δmomy_extrn_mv[:N_extrn], dest=ID_recv,
                 recvbuf=Δmomy_local_mv, source=ID_send)
        Sendrecv(Δmomz_extrn_mv[:N_extrn], dest=ID_recv,
                 recvbuf=Δmomz_local_mv, source=ID_send)
        # Apply local momentum updates recieved from other process
        for i in range(N_local):
            momx_local[i] += Δmomx_local[i]
            momy_local[i] += Δmomy_local[i]
            momz_local[i] += Δmomz_local[i]
        # Reset external momentum change buffers
        if j != N_partnerproc_pairs_minus_1:
            for i in range(N_extrns[mod(rank - j - 1, nprocs)]):
                Δmomx_extrn[i] = 0
                Δmomy_extrn[i] = 0
                Δmomz_extrn[i] = 0

# Function for updating all particle momenta in a particular direction,
# used in the PM algorithm.
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               N_local='size_t',
               PM_fac='double',
               force_grid='double[:, :, ::1]',
               posx='double*',
               posy='double*',
               posz='double*',
               mom='double*',
               # Locals
               i='size_t',
               x='double',
               y='double',
               z='double',
               force='double'
               )
def PM_update_mom(N_local, PM_fac, force_grid, posx, posy, posz, mom):
    for i in range(N_local):
        # The coordinates of the i'th particle,
        # transformed so that 0 <= x, y, z < 1.
        x = (posx[i] - domain_start_x)/domain_size_x
        y = (posy[i] - domain_start_y)/domain_size_y
        z = (posz[i] - domain_start_z)/domain_size_z
        # Look up the force via a CIC interpolation in the force grid
        force = CIC_grid2coordinates_scalar(force_grid, x, y, z)
        # Update the i'th momentum
        mom[i] += force*PM_fac


# Function for CIC interpolating the particles to the PM mesh,
# followed by a Fourier transformation.
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               )
def PM_CIC_FFT(particles):
    global PM_grid, domain_grid, domain_grid_noghosts
    # Nullify the PM mesh and the domain grid
    PM_grid[...] = 0
    domain_grid[...] = 0
    # Interpolate particle coordinates to the domain grid
    # (without the ghost layers).
    CIC_particles2grid(particles, domain_grid_noghosts)
    # External particles will contribute to the upper boundaries (not the ghost
    # layers) of domain_grid on other processes. Do the needed communication.
    communicate_boundaries(domain_grid_noghosts)
    # Communicate the interpolated data in the domain grid into the PM grid
    domain2PM(domain_grid_noghosts, PM_grid)
    # Fourier transform the grid forwards to Fourier space
    fftw_execute(plan_forward)


# Function for computing the gravitational force by the particle mesh method
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               Δt='double',
               only_long_range='bint',
               # Locals
               Greens_deconvolution='double',
               deconvolution_ij='double',
               deconvolution_ijk='double',
               deconvolution_j='double',
               PM_fac='double',
               force='double',
               posx='double*',
               posy='double*',
               posz='double*',
               momx='double*',
               momy='double*',
               momz='double*',
               i='int',
               j='int',
               j_global='int',
               k='int',
               ki='int',
               kj='int',
               kk='int',
               k2='unsigned long int',
               x='double',
               y='double',
               z='double',
               )
def PM(particles, Δt, only_long_range=False):
    """This function updates the momenta of all particles via the
    particle-mesh (PM) method.
    Note that the time step size Δt is really ∫_t^(t + Δt) dt/a.
    """
    global PM_grid, domain_grid, domain_grid_noghosts, force_grid
    # Extract variables from particles
    N_local = particles.N_local
    mass = particles.mass
    posx = particles.posx
    posy = particles.posy
    posz = particles.posz
    momx = particles.momx
    momy = particles.momy
    momz = particles.momz
    # CIC interpolate the particles and do forward Fourier transformation
    PM_CIC_FFT(particles)
    # Loop through the local j-dimension
    for j in range(PM_gridsize_local_j):
        # The j-component of the wave vector. Since PM_grid is distributed
        # along the j-dimension, an offset must be used.
        j_global = j + PM_gridstart_local_j
        if j_global > half_PM_gridsize:
            kj = j_global - PM_gridsize
        else:
            kj = j_global
        # (Square root of) the j-component of the deconvolution
        deconvolution_j = sinc(kj*π_recp_PM_gridsize)
        # Loop through the complete i-dimension
        for i in range(PM_gridsize):
            # The i-component of the wave vector
            if i > half_PM_gridsize:
                ki = i - PM_gridsize
            else:
                ki = i
            # (Square root of) the product of the i- and the j-component of
            # the deconvolution.
            deconvolution_ij = sinc(ki*π_recp_PM_gridsize)*deconvolution_j
            # Loop through the complete, padded k-dimension in steps of 2
            # (one complex number at a time).
            for k in range(0, PM_gridsize_padding, 2):
                # The k-component of the wave vector
                kk = k//2
                # Zero-division is illegal in pure Python. The [0, 0, 0]
                # element of the PM grid will be set later.
                if not cython.compiled:
                    if ki == kj == kk == 0:
                        continue
                # (Square root of) the product of all components of
                # the deconvolution.
                deconvolution_ijk = deconvolution_ij*sinc(kk*π_recp_PM_gridsize)
                # Multiply by the Greens function 1/k2 to get the the
                # potential. Deconvolve twice for the two CIC interpolations
                # (the mass assignment and the upcomming force interpolation).
                # Remember that PM_grid is transposed in the first two
                # dimensions due to the forward FFT.
                k2 = ki**2 + kj**2 + kk**2
                Greens_deconvolution = 1/(k2*deconvolution_ijk**4)
                if only_long_range:
                    Greens_deconvolution *= exp(k2*longrange_exponent_fac)
                PM_grid[j, i, k] *= Greens_deconvolution      # Real part
                PM_grid[j, i, k + 1] *= Greens_deconvolution  # Imaginary part
    # The global [0, 0, 0] element of the PM grid should be zero
    if PM_gridstart_local_j == 0:
        PM_grid[0, 0, 0] = 0  # Real part
        PM_grid[0, 0, 1] = 0  # Imaginary part
    # Fourier transform the grid back to coordinate space.
    # Now the grid stores potential values.
    fftw_execute(plan_backward)
    for i in range(PM_grid.shape[0]):
        for j in range(PM_grid.shape[1]):
            for k in range(PM_grid.shape[2]):
                PM_grid[i, j, k] /= PM_gridsize3
    # Communicate the potential stored in the PM mesh to the domain grid
    PM2domain(domain_grid_noghosts, PM_grid)
    # The upper boundaries (not the ghost layers) of the domain grid should be
    # a copy of the lower boundaries of the next domain. Do the needed
    # communication.
    communicate_boundaries(domain_grid_noghosts, mode=1)
    # Communicate the ghost layers of the domain grid
    communicate_ghosts(domain_grid)
    # The factor which shold be multiplied on the PM grid to get actual units
    PM_fac = PM_fac_const*mass**2*Δt
    # Compute the local forces in the x-direction via the four point rule
    for i in range(2, domain_grid.shape[0] - 2):
        for j in range(2, domain_grid.shape[1] - 2):
            for k in range(2, domain_grid.shape[2] - 2):
                force_grid[i - 2,
                           j - 2,
                           k - 2] = (two_thirds*(domain_grid[i + 1, j, k]
                                               - domain_grid[i - 1, j, k])
                                  - one_twelfth*(domain_grid[i + 2, j, k]
                                               - domain_grid[i - 2, j, k]))
    # Update local x-momenta
    PM_update_mom(N_local, PM_fac, force_grid, posx, posy, posz, momx)
    # Compute the forces in the y-direction via the four point rule
    for i in range(2, domain_grid.shape[0] - 2):
        for j in range(2, domain_grid.shape[1] - 2):
            for k in range(2, domain_grid.shape[2] - 2):
                force_grid[i - 2,
                           j - 2,
                           k - 2] = (two_thirds*(domain_grid[i, j + 1, k]
                                               - domain_grid[i, j - 1, k])
                                  - one_twelfth*(domain_grid[i, j + 2, k]
                                               - domain_grid[i, j - 2, k]))
    # Update local y-momenta
    PM_update_mom(N_local, PM_fac, force_grid, posx, posy, posz, momy)
    # Compute the forces in the z-direction via the four point rule
    for i in range(2, domain_grid.shape[0] - 2):
        for j in range(2, domain_grid.shape[1] - 2):
            for k in range(2, domain_grid.shape[2] - 2):
                force_grid[i - 2,
                           j - 2,
                           k - 2] = (two_thirds*(domain_grid[i, j, k + 1]
                                               - domain_grid[i, j, k - 1])
                                  - one_twelfth*(domain_grid[i, j, k + 2]
                                               - domain_grid[i, j, k - 2]))
    # Update local z-momenta
    PM_update_mom(N_local, PM_fac, force_grid, posx, posy, posz, momz)

    # Done performing FFT's. Cleanup
    #fftw_clean(cython.address(PM_grid[0, 0, 0]), plan_forward, plan_backward)


# This collection of functions simply test whether the passed coordinates
# lie within a certain domain boundary or not.
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_right(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i > boundary_x_max

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_left(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i < boundary_x_min

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_forward(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i > boundary_y_max

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_backward(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i < boundary_y_min

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_up(posx_local_i, posy_local_i, posz_local_i):
    return posz_local_i > boundary_z_max

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_down(posx_local_i, posy_local_i, posz_local_i):
    return posz_local_i < boundary_z_min

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_rightforward(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i > boundary_x_max and posy_local_i > boundary_y_max

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_rightbackward(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i > boundary_x_max and posy_local_i < boundary_y_min

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_rightup(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i > boundary_x_max and posz_local_i > boundary_z_max

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_rightdown(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i > boundary_x_max and posz_local_i < boundary_z_min

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_leftforward(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i < boundary_x_min and posy_local_i > boundary_y_max

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_leftbackward(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i < boundary_x_min and posy_local_i < boundary_y_min

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_leftup(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i < boundary_x_min and posz_local_i > boundary_z_max

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_leftdown(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i < boundary_x_min and posz_local_i < boundary_z_min

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_forwardup(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i > boundary_y_max and posz_local_i > boundary_z_max

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_forwarddown(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i > boundary_y_max and posz_local_i < boundary_z_min

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_backwardup(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i < boundary_y_min and posz_local_i > boundary_z_max

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_backwarddown(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i < boundary_y_min and posz_local_i < boundary_z_min

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_rightforwardup(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i > boundary_x_max and posy_local_i > boundary_y_max
                                          and posz_local_i > boundary_z_max)

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_rightforwarddown(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i > boundary_x_max and posy_local_i > boundary_y_max
                                          and posz_local_i < boundary_z_min)

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_rightbackwardup(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i > boundary_x_max and posy_local_i < boundary_y_min
                                          and posz_local_i > boundary_z_max)

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_rightbackwarddown(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i > boundary_x_max and posy_local_i < boundary_y_min
                                          and posz_local_i < boundary_z_min)

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_leftforwardup(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i < boundary_x_min and posy_local_i > boundary_y_max
                                          and posz_local_i > boundary_z_max)

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_leftforwarddown(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i < boundary_x_min and posy_local_i > boundary_y_max
                                          and posz_local_i < boundary_z_min)

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_leftbackwardup(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i < boundary_x_min and posy_local_i < boundary_y_min
                                          and posz_local_i > boundary_z_max)

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               )
@cython.returns('bint')
def in_boundary_leftbackwarddown(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i < boundary_x_min and posy_local_i < boundary_y_min
                                          and posz_local_i < boundary_z_min)

# Function for computing the gravitational force by the particle mesh method
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               Δt='double',
               # Locals
               N_extrn='size_t',
               N_local='size_t',
               N_boundary1='size_t',
               N_boundary2='size_t',
               i='size_t',
               in_boundary1='func_b_ddd',
               in_boundary2='func_b_ddd',
               j='int',
               mass='double',
               momx_local='double*',
               momy_local='double*',
               momz_local='double*',
               posx_local='double*',
               posx_local_i='double',
               posy_local='double*',
               posy_local_i='double',
               posz_local='double*',
               posz_local_i='double',
               rank_send='int',
               rank_recv='int',
               softening2='double',
               Δmemory='size_t',
               )
def P3M(particles, Δt):
    """The long-range part is computed via the PM function. Local particles
    also interact via short-range direct summation. Finally, each process
    send particles near its boundary to the corresponding processor, which
    computes the short-range forces via direct summation between the
    received boundary particles and its own boundary particles at the
    opposite face/edge/point. This is done by iterating over pairs of
    neighboring processes. In the first iteration, particles in the right
    boundary are sent to the right process, while particles in the left
    process' right boundary are received. The boundary containing particles
    which are not sent is refered to as boundary1, while the other is refered
    to as boundary2.
    """
    global posx_local_boundary, posy_local_boundary, posz_local_boundary
    global posx_local_boundary_mv, posy_local_boundary_mv, posz_local_boundary_mv
    global Δmomx_local_boundary, Δmomy_local_boundary, Δmomz_local_boundary
    global Δmomx_local_boundary_mv, Δmomy_local_boundary_mv, Δmomz_local_boundary_mv
    global posx_extrn, posy_extrn, posz_extrn
    global posx_extrn_mv, posy_extrn_mv, posz_extrn_mv
    global Δmomx_extrn, Δmomy_extrn, Δmomz_extrn
    global Δmomx_extrn_mv, Δmomy_extrn_mv, Δmomz_extrn_mv
    global Δmomx_local, Δmomy_local, Δmomz_local
    global Δmomx_local_mv, Δmomy_local_mv, Δmomz_local_mv
    global indices_send, indices_send_mv
    global indices_boundary, indices_boundary_mv
    # Compute the long-range force via the PM method
    PM(particles, Δt, True)
    # Extract variables from particles
    N_local = particles.N_local
    mass = particles.mass
    momx_local = particles.momx
    momy_local = particles.momy
    momz_local = particles.momz
    posx_local = particles.posx
    posy_local = particles.posy
    posz_local = particles.posz
    softening2 = particles.softening**2
    # Compute the short-range interactions within the local domain.
    # Note that "vector" is not actually used due to flag_input=0.
    direct_summation(posx_local, posy_local, posz_local,
                     momx_local, momy_local, momz_local,
                     mass, N_local,
                     posx_local, posy_local, posz_local,
                     vector, vector, vector,
                     mass, N_local,
                     Δt, softening2, 0,
                     only_short_range=True)
    # All work done if only one domain exists (if run on a single process)
    if nprocs == 1:
        return
    # Now only short-range interactions between neighboring
    # domain boundaries remain.
    # The buffers below may increase their size by this amount at a time.
    Δmemory = int(1 + ceil(0.05*N_local*np.max([P3M_cutoff_phys/domain_size_x,
                                                P3M_cutoff_phys/domain_size_y,
                                                P3M_cutoff_phys/domain_size_z,
                                                ])))
    # Loop over all 26 neighbors (two at a time)
    for j in range(13):
        # It is important that the processes iterate synchronously, so that
        # the received data really is what the local process think it is.
        Barrier()
        # The ranks of the processes to communicate with
        rank_send = boundary_ranks_send[j]
        rank_recv = boundary_ranks_recv[j]
        # The functions for in-boundary tests
        in_boundary1 = in_boundary1_funcs[j]
        in_boundary2 = in_boundary2_funcs[j]
        # Find out which particles participate in the local and the external
        # short-range interaction.
        N_boundary1 = 0
        N_boundary2 = 0
        for i in range(N_local):
            posx_local_i = posx_local[i]
            posy_local_i = posy_local[i]
            posz_local_i = posz_local[i]
            # Check if particle should interact to the left
            if in_boundary1(posx_local_i, posy_local_i, posz_local_i):
                indices_boundary[N_boundary1] = i
                posx_local_boundary[N_boundary1] = posx_local_i
                posy_local_boundary[N_boundary1] = posy_local_i
                posz_local_boundary[N_boundary1] = posz_local_i
                N_boundary1 += 1
                # Enlarge buffers if needed
                if posx_local_boundary_mv.shape[0] == N_boundary1:
                    indices_boundary = realloc(indices_boundary, (N_boundary1 + Δmemory)*sizeof('size_t'))
                    indices_boundary_mv = cast(indices_boundary, 'size_t[:(N_boundary1 + Δmemory)]')
                    posx_local_boundary = realloc(posx_local_boundary, (N_boundary1 + Δmemory)*sizeof('double'))
                    posx_local_boundary_mv = cast(posx_local_boundary, 'double[:(N_boundary1 + Δmemory)]')
                    posy_local_boundary = realloc(posy_local_boundary, (N_boundary1 + Δmemory)*sizeof('double'))
                    posy_local_boundary_mv = cast(posy_local_boundary, 'double[:(N_boundary1 + Δmemory)]')
                    posz_local_boundary = realloc(posz_local_boundary, (N_boundary1 + Δmemory)*sizeof('double'))
                    posz_local_boundary_mv = cast(posz_local_boundary, 'double[:(N_boundary1 + Δmemory)]')
                    Δmomx_local_boundary = realloc(Δmomx_local_boundary, (N_boundary1 + Δmemory)*sizeof('double'))
                    Δmomx_local_boundary_mv = cast(Δmomx_local_boundary, 'double[:(N_boundary1 + Δmemory)]')
                    Δmomy_local_boundary = realloc(Δmomy_local_boundary, (N_boundary1 + Δmemory)*sizeof('double'))
                    Δmomy_local_boundary_mv = cast(Δmomy_local_boundary, 'double[:(N_boundary1 + Δmemory)]')
                    Δmomz_local_boundary = realloc(Δmomz_local_boundary, (N_boundary1 + Δmemory)*sizeof('double'))
                    Δmomz_local_boundary_mv = cast(Δmomz_local_boundary, 'double[:(N_boundary1 + Δmemory)]')
            # Check if particle should be sent to the right
            if in_boundary2(posx_local_i, posy_local_i, posz_local_i):
                # Particle i should be send
                indices_send[N_boundary2] = i
                # Fill buffers with its coordinates. The Δmom(x/y/z)_local
                # variables are used for this to save memory, as these
                # oterwise first used after the particles have
                # been comminucated.
                Δmomx_local[N_boundary2] = posx_local_i
                Δmomy_local[N_boundary2] = posy_local_i
                Δmomz_local[N_boundary2] = posz_local_i
                N_boundary2 += 1
                # Enlarge buffers if needed
                if indices_send_mv.shape[0] == N_boundary2:
                    indices_send = realloc(indices_send, (N_boundary2 + Δmemory)*sizeof('size_t'))
                    indices_send_mv = cast(indices_send, 'size_t[:(N_boundary2 + Δmemory)]')
                if Δmomx_local_mv.shape[0] == N_boundary2:
                    Δmomx_local = realloc(Δmomx_local, (N_boundary2 + Δmemory)*sizeof('double'))
                    Δmomx_local_mv = cast(Δmomx_local, 'double[:(N_boundary2 + Δmemory)]')
                    Δmomy_local = realloc(Δmomy_local, (N_boundary2 + Δmemory)*sizeof('double'))
                    Δmomy_local_mv = cast(Δmomy_local, 'double[:(N_boundary2 + Δmemory)]')
                    Δmomz_local = realloc(Δmomz_local, (N_boundary2 + Δmemory)*sizeof('double'))
                    Δmomz_local_mv = cast(Δmomz_local, 'double[:(N_boundary2 + Δmemory)]')
        # Communicate the number of particles to be communicated
        N_extrn = sendrecv(N_boundary2, dest=rank_send, source=rank_recv)
        # Enlarge the receive buffers if needed
        if posx_extrn_mv.shape[0] < N_extrn:
            posx_extrn = realloc(posx_extrn, N_extrn*sizeof('double'))
            posy_extrn = realloc(posy_extrn, N_extrn*sizeof('double'))
            posz_extrn = realloc(posz_extrn, N_extrn*sizeof('double'))
            posx_extrn_mv = cast(posx_extrn, 'double[:N_extrn]')
            posy_extrn_mv = cast(posy_extrn, 'double[:N_extrn]')
            posz_extrn_mv = cast(posz_extrn, 'double[:N_extrn]')
            Δmomx_extrn = realloc(Δmomx_extrn, N_extrn*sizeof('double'))
            Δmomy_extrn = realloc(Δmomy_extrn, N_extrn*sizeof('double'))
            Δmomz_extrn = realloc(Δmomz_extrn, N_extrn*sizeof('double'))
            Δmomx_extrn_mv = cast(Δmomx_extrn, 'double[:N_extrn]')
            Δmomy_extrn_mv = cast(Δmomy_extrn, 'double[:N_extrn]')
            Δmomz_extrn_mv = cast(Δmomz_extrn, 'double[:N_extrn]')
        # Communicate the particles. Remember that at this point,
        # Δmom(x/y/z)_local actually store coordinates of particles in
        # the second boundary.
        Sendrecv(Δmomx_local_mv[:N_boundary2], dest=rank_send,
                 recvbuf=posx_extrn_mv, source=rank_recv)
        Sendrecv(Δmomy_local_mv[:N_boundary2], dest=rank_send,
                 recvbuf=posy_extrn_mv, source=rank_recv)
        Sendrecv(Δmomz_local_mv[:N_boundary2], dest=rank_send,
                 recvbuf=posz_extrn_mv, source=rank_recv)
        # Do direct summation between local and external particles
        for i in range(N_extrn):
            Δmomx_extrn[i] = 0
            Δmomy_extrn[i] = 0
            Δmomz_extrn[i] = 0
        for i in range(N_boundary1):
            Δmomx_local_boundary[i] = 0
            Δmomy_local_boundary[i] = 0
            Δmomz_local_boundary[i] = 0
        direct_summation(posx_local_boundary,
                         posy_local_boundary,
                         posz_local_boundary,
                         Δmomx_local_boundary,
                         Δmomy_local_boundary,
                         Δmomz_local_boundary,
                         mass, N_boundary1,
                         posx_extrn, posy_extrn, posz_extrn,
                         Δmomx_extrn, Δmomy_extrn, Δmomz_extrn,
                         mass, N_extrn,
                         Δt, softening2, 1,
                         only_short_range=True)
        # Apply the momentum changes to the local particle momentum data
        for i in range(N_boundary1):
            momx_local[indices_boundary[i]] += Δmomx_local_boundary[i]
            momy_local[indices_boundary[i]] += Δmomy_local_boundary[i]
            momz_local[indices_boundary[i]] += Δmomz_local_boundary[i]
        # Send momentum updates back to the process from which
        # positions were recieved. Recieve momentum updates from the
        # process which the local positions were send to.
        Sendrecv(Δmomx_extrn_mv[:N_extrn], dest=rank_recv,
                 recvbuf=Δmomx_local_mv, source=rank_send)
        Sendrecv(Δmomy_extrn_mv[:N_extrn], dest=rank_recv,
                 recvbuf=Δmomy_local_mv, source=rank_send)
        Sendrecv(Δmomz_extrn_mv[:N_extrn], dest=rank_recv,
                 recvbuf=Δmomz_local_mv, source=rank_send)
        # Apply local momentum updates recieved from other process
        for i in range(N_boundary2):
            momx_local[indices_send[i]] += Δmomx_local[i]
            momy_local[indices_send[i]] += Δmomy_local[i]
            momz_local[indices_send[i]] += Δmomz_local[i]


# Initializes stuff for the PM algorithm at import time,
# if the PM method is to be used.
cython.declare(fftw_struct='fftw_return_struct',
               PM_gridsize_local_i='ptrdiff_t',
               PM_gridsize_local_j='ptrdiff_t',
               PM_gridstart_local_i='ptrdiff_t',
               PM_gridstart_local_j='ptrdiff_t',
               PM_grid='double[:, :, ::1]',
               plan_forward='fftw_plan',
               plan_backward='fftw_plan',
               )
if use_PM:
    # The PM mesh and functions on it
    if not cython.compiled:
        # Initialization of the PM mesh in pure Python.
        PM_gridsize_local_i = PM_gridsize_local_j = int(PM_gridsize/nprocs)
        if PM_gridsize_local_i != PM_gridsize/nprocs:
            # If PM_gridsize is not divisible by nprocs, the code cannot
            # figure out exactly how FFTW distribute the grid among the
            # processes. In stead of guessing, do not even try to emulate
            # the behaviour of FFTW.
            raise ValueError('The PM method in pure Python mode only works '
                     + 'when\nPM_gridsize is divisible by the number'
                     + 'of processes!')
        PM_gridstart_local_i = PM_gridstart_local_j = PM_gridsize_local_i*rank
        PM_grid = empty((PM_gridsize_local_i, PM_gridsize,
                         PM_gridsize_padding), dtype='float64')
        # The output of the following function is formatted just
        # like that of the MPI implementation of FFTW.
        plan_backward = 'plan_backward'
        plan_forward = 'plan_forward'
        def fftw_execute(plan):
            global PM_grid
            # The pure Python FFT implementation is serial. Every process
            # computes the entire FFT of the temporary varaible PM_grid_global.
            PM_grid_global = empty((PM_gridsize, PM_gridsize,
                                    PM_gridsize_padding))
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
                PM_grid = PM_grid_global[PM_gridstart_local_j:
                                         (PM_gridstart_local_j
                                          + PM_gridsize_local_j),
                                         :, :]
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
                PM_grid = PM_grid_global[PM_gridstart_local_i:
                                         (PM_gridstart_local_i
                                          + PM_gridsize_local_i),
                                         :, :]
    else:
        """
        # Initialize fftw_mpi, allocate the grid, initialize the
        # local grid sizes and start indices and do FFTW planning.
        fftw_struct = fftw_setup(PM_gridsize, PM_gridsize, PM_gridsize)
        # Unpack fftw_struct
        PM_gridsize_local_i = fftw_struct.gridsize_local_i
        PM_gridsize_local_j = fftw_struct.gridsize_local_j
        PM_gridstart_local_i = fftw_struct.gridstart_local_i
        PM_gridstart_local_j = fftw_struct.gridstart_local_j
        # Wrap a memoryview around the grid. Loop as noted in fft.c, but use
        # PM_grid[i, j, k] when in real space and PM_grid[j, i, k] when in
        # Fourier space
        if PM_gridsize_local_i > 0:
            PM_grid = <double[:PM_gridsize_local_i, :PM_gridsize, :PM_gridsize_padding]> fftw_struct.grid
        else:
            # The process do not participate in the FFT computations
            PM_grid = empty((0, PM_gridsize, PM_gridsize_padding))
        plan_forward  = fftw_struct.plan_forward
        plan_backward = fftw_struct.plan_backward
        """
else:
    # As these should be importable, they need to be assigned even if not used
    PM_gridsize_local_j = 0
    PM_gridstart_local_j = 0
    PM_grid = empty((1, 1, 1), dtype='float64')
# Cut out the domains at import time
cython.declare(domain_cuts='list',
               domain_local='int[::1]',
               domain_size_i='int',
               domain_size_j='int',
               domain_size_k='int',
               domain_size_x='double',
               domain_size_y='double',
               domain_size_z='double',
               domain_start_x='double',
               domain_start_y='double',
               domain_start_z='double',
               )
# Number of domains in all three dimensions
domain_cuts = cutout_domains(nprocs)
# The indices in domain_layout of the local domain
domain_local = array(np.unravel_index(rank, domain_cuts), dtype='int32')
# The linear size of the domains, which are the same for all of them
domain_size_x = boxsize/domain_cuts[0]
domain_size_y = boxsize/domain_cuts[1]
domain_size_z = boxsize/domain_cuts[2]
# The start positions of the local domain
domain_start_x = domain_local[0]*domain_size_x
domain_start_y = domain_local[1]*domain_size_y
domain_start_z = domain_local[2]*domain_size_z
# The size of the domain grid in PM grid units (actually needed for
# the P3M, not the PM, algorithm).
domain_size_i = PM_gridsize//domain_cuts[0]
domain_size_j = PM_gridsize//domain_cuts[1]
domain_size_k = PM_gridsize//domain_cuts[2]
# Initialize the domain grid and the force grid, if the PM method
# should be used.
if use_PM:
    cython.declare(i='int',
                   domain_grid='double[:, :, ::1]',
                   domain_grid_noghosts='double[:, :, ::1]',
                   force_grid='double[:, :, ::1]',
                   )
    # A grid over the local domain. An additional layer of thickness 1 is given
    # to the domain grid, so that these outer points corresponds to the same
    # physical coordinates as the first points in the next domain.
    # Also, an additional layer of thickness 2 is given on top of the previous
    # layer. This shall be used as a ghost layer for finite differencing.
    domain_grid = zeros([PM_gridsize//domain_cuts[i] + 1 + 2*2 for i in range(3)],
                        dtype='float64')
    # Memoryview of the domain grid without the ghost layers
    domain_grid_noghosts = domain_grid[2:(domain_grid.shape[0] - 2),
                                       2:(domain_grid.shape[1] - 2),
                                       2:(domain_grid.shape[2] - 2)]
    # The grid containing the forces in the PM algorithm,
    # one component at a time.
    force_grid = zeros((domain_grid_noghosts.shape[0],
                        domain_grid_noghosts.shape[1],
                        domain_grid_noghosts.shape[2]), dtype='float64')
    # Test if the grid has been constructed correctly. If not it is because
    # nprocs and PM_gridsize are incompatible.
    for i in range(3):
        if PM_gridsize != domain_cuts[i]*(domain_grid.shape[i] - 1 - 2*2):
            msg = ('A PM_gridsize of ' + str(PM_gridsize) + ' cannot be'
                   + ' equally shared among ' + str(nprocs) + ' processes')
            raise ValueError(msg)
        if np.min([domain_grid.shape[i] for i in range(3)]) < 2 + 1 + 2*2:
            msg = ('A PM_gridsize of ' + str(PM_gridsize) + ' is too small'
                   + ' for ' + str(nprocs) + ' processes')
            raise ValueError(msg)
# Check if PM_grid is large enough for P3M to work, if the P3M
# algorithm is to be used.
for kick_algorithm in kick_algorithms.values():
    if kick_algorithm == 'P3M':
        if (   domain_size_i < P3M_scale*P3M_cutoff
            or domain_size_j < P3M_scale*P3M_cutoff
            or domain_size_k < P3M_scale*P3M_cutoff):
            msg = ('A PM_gridsize of ' + str(PM_gridsize) + ' and '
                   + str(nprocs) + ' processes results in following domain'
                   + ' partition: ' + str(list(domain_cuts))
                   + '.\nThe smallest domain width is '
                   + str(np.min([domain_size_i, domain_size_j,
                                 domain_size_k]))
                   + ' grid cells, while the choice of P3M_scale ('
                   + str(P3M_scale) + ') and P3M_cutoff ('
                   + str(P3M_cutoff) + ')\nmeans that the domains must be '
                   + 'at least '
                   + str(P3M_scale*P3M_cutoff) + ' grid cells for the '
                   + 'P3M algorithm to work.'
                )
            raise ValueError(msg)
        if (  (domain_size_i < 2*P3M_scale*P3M_cutoff
            or domain_size_j < 2*P3M_scale*P3M_cutoff
            or domain_size_k < 2*P3M_scale*P3M_cutoff)
            and np.min(domain_cuts) < 3):
            # This is only allowed if domain_cuts are at least 3 in each
            # direction. Otherwise the left and the right (say) process
            # is the same, and the boundaries will be send to it twize,
            # and these will overlap with each other in the left/right
            # domain and gravity will be applied twize.
            msg = ('A PM_gridsize of ' + str(PM_gridsize) + ' and '
                   + str(nprocs) + ' processes results in following domain'
                   + ' partition: ' + str(list(domain_cuts))
                   + '.\nThe smallest domain width is '
                   + str(np.min([domain_size_i, domain_size_j,
                                 domain_size_k]))
                   + ' grid cells, while the choice of P3M_scale ('
                   + str(P3M_scale) + ') and P3M_cutoff ('
                   + str(P3M_cutoff) + ')\nmeans that the domains must be '
                   + 'at least '
                   + str(2*P3M_scale*P3M_cutoff) + ' grid cells for the '
                   + 'P3M algorithm to work.'
                )
            raise ValueError(msg)
        break
# Initialize stuff for the PP and P3M algorithms at import time
cython.declare(boundary_ranks_recv='int[::1]',
               boundary_ranks_send='int[::1]',  
               boundary_x_max='double',
               boundary_x_min='double',
               boundary_y_max='double',
               boundary_y_min='double',
               boundary_z_max='double',
               boundary_z_min='double',  
               in_boundary1_funcs='func_b_ddd*',
               in_boundary2_funcs='func_b_ddd*',
               indices_boundary='size_t*',
               indices_boundary_mv='size_t[::1]',
               indices_send='size_t*',
               indices_send_mv='size_t[::1]',
               neighbors='dict',
               posx_extrn='double*',
               posx_extrn_mv='double[::1]',
               posx_local_boundary='double*',
               posx_local_boundary_mv='double[::1]',
               posy_extrn='double*',
               posy_extrn_mv='double[::1]',
               posy_local_boundary='double*',
               posy_local_boundary_mv='double[::1]',
               posz_extrn='double*',              
               posz_extrn_mv='double[::1]',
               posz_local_boundary='double*',
               posz_local_boundary_mv='double[::1]',
               rank_right='int',
               rank_left='int',
               rank_forward='int',
               rank_backward='int',
               rank_up='int',
               rank_down='int',
               rank_rightforward='int',
               rank_rightbackward='int',
               rank_rightup='int',
               rank_rightdown='int',
               rank_leftforward='int',
               rank_leftbackward='int',
               rank_leftup='int',
               rank_leftdown='int',
               rank_forwardup='int',
               rank_forwarddown='int',
               rank_backwardup='int',
               rank_backwarddown='int',
               rank_rightforwardup='int',
               rank_rightforwarddown='int',
               rank_rightbackwardup='int',
               rank_rightbackwarddown='int',
               rank_leftforwardup='int',
               rank_leftforwarddown='int',
               rank_leftbackwardup='int',
               rank_leftbackwarddown='int',
               Δmomx_extrn='double*',
               Δmomx_extrn_mv='double[::1]',
               Δmomx_local='double*',
               Δmomx_local_boundary='double*',
               Δmomx_local_boundary_mv='double[::1]',
               Δmomx_local_mv='double[::1]',
               Δmomy_extrn='double*',
               Δmomy_extrn_mv='double[::1]',
               Δmomy_local='double*',
               Δmomy_local_boundary='double*',
               Δmomy_local_boundary_mv='double[::1]',
               Δmomy_local_mv='double[::1]',
               Δmomz_extrn='double*',
               Δmomz_extrn_mv='double[::1]',               
               Δmomz_local='double*',
               Δmomz_local_boundary='double*',
               Δmomz_local_boundary_mv='double[::1]',
               Δmomz_local_mv='double[::1]',
               )
# For storing positions of particles received from external domains
posx_extrn = malloc(1*sizeof('double'))
posy_extrn = malloc(1*sizeof('double'))
posz_extrn = malloc(1*sizeof('double'))
posx_extrn_mv = cast(posx_extrn, 'double[:1]')
posy_extrn_mv = cast(posy_extrn, 'double[:1]')
posz_extrn_mv = cast(posz_extrn, 'double[:1]')
# For storing momentum changes
Δmomx_local = malloc(1*sizeof('double'))
Δmomy_local = malloc(1*sizeof('double'))
Δmomz_local = malloc(1*sizeof('double'))
Δmomx_local_mv = cast(Δmomx_local, 'double[:1]')
Δmomy_local_mv = cast(Δmomy_local, 'double[:1]')
Δmomz_local_mv = cast(Δmomz_local, 'double[:1]')
Δmomx_extrn = malloc(1*sizeof('double'))
Δmomy_extrn = malloc(1*sizeof('double'))
Δmomz_extrn = malloc(1*sizeof('double'))
Δmomx_extrn_mv = cast(Δmomx_extrn, 'double[:1]')
Δmomy_extrn_mv = cast(Δmomy_extrn, 'double[:1]')
Δmomz_extrn_mv = cast(Δmomz_extrn, 'double[:1]')
# For storing the indices of particles to be send
indices_send = malloc(1*sizeof('size_t'))
indices_send_mv = cast(indices_send, 'size_t[:1]')
# For storing the indices of local particles which should interact with
# boundaries of other domains in the P3M method.
indices_boundary = malloc(1*sizeof('size_t'))
indices_boundary_mv = cast(indices_send, 'size_t[:1]')
# For storing a copy of those local particles that consitutes the short-range
# domain boundaries.
posx_local_boundary = malloc(1*sizeof('double'))
posy_local_boundary = malloc(1*sizeof('double'))
posz_local_boundary = malloc(1*sizeof('double'))
posx_local_boundary_mv = cast(posx_local_boundary, 'double[:1]')
posy_local_boundary_mv = cast(posy_local_boundary, 'double[:1]')
posz_local_boundary_mv = cast(posz_local_boundary, 'double[:1]')
Δmomx_local_boundary = malloc(1*sizeof('double'))
Δmomy_local_boundary = malloc(1*sizeof('double'))
Δmomz_local_boundary = malloc(1*sizeof('double'))
Δmomx_local_boundary_mv = cast(Δmomx_local_boundary, 'double[:1]')
Δmomy_local_boundary_mv = cast(Δmomy_local_boundary, 'double[:1]')
Δmomz_local_boundary_mv = cast(Δmomz_local_boundary, 'double[:1]')
# Find the ranks of all neighboring domains
neighbors = neighboring_ranks()
rank_right = neighbors['right']
rank_left = neighbors['left']
rank_forward = neighbors['forward']
rank_backward = neighbors['backward']
rank_up = neighbors['up']
rank_down = neighbors['down']
rank_rightforward = neighbors['rightforward']
rank_rightbackward = neighbors['rightbackward']
rank_rightup = neighbors['rightup']
rank_rightdown = neighbors['rightdown']
rank_leftforward = neighbors['leftforward']
rank_leftbackward = neighbors['leftbackward']
rank_leftup = neighbors['leftup']
rank_leftdown = neighbors['leftdown']
rank_forwardup = neighbors['forwardup']
rank_forwarddown = neighbors['forwarddown']
rank_backwardup = neighbors['backwardup']
rank_backwarddown = neighbors['backwarddown']
rank_rightforwardup = neighbors['rightforwardup']
rank_rightforwarddown = neighbors['rightforwarddown']
rank_rightbackwardup = neighbors['rightbackwardup']
rank_rightbackwarddown = neighbors['rightbackwarddown']
rank_leftforwardup = neighbors['leftforwardup']
rank_leftforwarddown = neighbors['leftforwarddown']
rank_leftbackwardup = neighbors['leftbackwardup']
rank_leftbackwarddown = neighbors['leftbackwarddown']
# Save the neighboring ranks in a particular order,
# for use in the P3M algorithm
boundary_ranks_send = array([rank_right, rank_forward, rank_up,
                             rank_rightforward, rank_rightbackward,
                             rank_rightup, rank_rightdown,
                             rank_forwardup, rank_forwarddown,
                             rank_rightforwardup, rank_rightforwarddown,
                             rank_rightbackwardup, rank_rightbackwarddown, 
                             ], dtype='int32')
boundary_ranks_recv = array([rank_left,  rank_backward, rank_down,
                             rank_leftbackward, rank_leftforward,
                             rank_leftdown, rank_leftup,
                             rank_backwarddown, rank_backwardup,
                             rank_leftbackwarddown, rank_leftbackwardup,
                             rank_leftforwarddown, rank_leftforwardup,
                             ], dtype='int32')
# Function pointer arrays to the in-boundary test functions
in_boundary1_funcs = malloc(13*sizeof('func_b_ddd'))
in_boundary2_funcs = malloc(13*sizeof('func_b_ddd'))
in_boundary1_funcs[0] = in_boundary_left
in_boundary1_funcs[1] = in_boundary_backward
in_boundary1_funcs[2] = in_boundary_down
in_boundary1_funcs[3] = in_boundary_leftbackward
in_boundary1_funcs[4] = in_boundary_leftforward
in_boundary1_funcs[5] = in_boundary_leftdown
in_boundary1_funcs[6] = in_boundary_leftup
in_boundary1_funcs[7] = in_boundary_backwarddown
in_boundary1_funcs[8] = in_boundary_backwardup
in_boundary1_funcs[9] = in_boundary_leftbackwarddown
in_boundary1_funcs[10] = in_boundary_leftbackwardup
in_boundary1_funcs[11] = in_boundary_leftforwarddown
in_boundary1_funcs[12] = in_boundary_leftforwardup
in_boundary2_funcs[0] = in_boundary_right
in_boundary2_funcs[1] = in_boundary_forward
in_boundary2_funcs[2] = in_boundary_up
in_boundary2_funcs[3] = in_boundary_rightforward
in_boundary2_funcs[4] = in_boundary_rightbackward
in_boundary2_funcs[5] = in_boundary_rightup
in_boundary2_funcs[6] = in_boundary_rightdown
in_boundary2_funcs[7] = in_boundary_forwardup
in_boundary2_funcs[8] = in_boundary_forwarddown
in_boundary2_funcs[9] = in_boundary_rightforwardup
in_boundary2_funcs[10] = in_boundary_rightforwarddown
in_boundary2_funcs[11] = in_boundary_rightbackwardup
in_boundary2_funcs[12] = in_boundary_rightbackwarddown
# These coordinates define the boundaries of the domain. Particles
# within the boundaries shall interact with particles in the neighbor
# domain via the short-range interaction, in the P3M method.
boundary_x_max = domain_start_x + domain_size_x - P3M_cutoff_phys
boundary_x_min = domain_start_x + P3M_cutoff_phys
boundary_y_max = domain_start_y + domain_size_y - P3M_cutoff_phys
boundary_y_min = domain_start_y + P3M_cutoff_phys
boundary_z_max = domain_start_z + domain_size_z - P3M_cutoff_phys
boundary_z_min = domain_start_z + P3M_cutoff_phys


pxd = """
double[:, :, ::1] PM_grid
ptrdiff_t PM_gridsize_local_j, PM_gridstart_local_j
"""
