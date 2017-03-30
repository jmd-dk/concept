# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2017 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CO𝘕CEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CO𝘕CEPT. If not, see http://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
from mesh import diff_domain
cimport('from ewald import ewald')
cimport('from communication import communicate_domain, find_N_recv, rank_neighboring_domain')
cimport('from communication import domain_size_x,  domain_size_y,  domain_size_z')
cimport('from communication import domain_start_x, domain_start_y, domain_start_z')
cimport('from mesh import CIC_components2slabs, CIC_grid2grid, CIC_scalargrid2coordinates')
cimport('from mesh import slab, slab_size_j, slab_start_j, slabs_FFT, slabs_IFFT, slabs2φ')
cimport('from mesh import φ, φ_noghosts')



# Function for direct summation of gravitational forces between
# particles in two domains (possibly the same).
@cython.header(# Arguments
               posx_i='double*',
               posy_i='double*',
               posz_i='double*',
               momx_i='double*',
               momy_i='double*',
               momz_i='double*',
               mass_i='double',
               N_local_i='Py_ssize_t',
               posx_j='double*',
               posy_j='double*',
               posz_j='double*',
               Δmomx_j='double*',
               Δmomy_j='double*',
               Δmomz_j='double*',
               mass_j='double',
               N_local_j='Py_ssize_t',
               ᔑdt='dict',
               softening2='double',
               only_short_range='bint',
               flag_input='int',
               # Locals
               dim='int',
               eom_factor='double',
               force='double*',
               i='Py_ssize_t',
               i_end='Py_ssize_t',
               j='Py_ssize_t',
               j_start='Py_ssize_t',
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
                     ᔑdt, softening2, flag_input, only_short_range=False):
    """This function takes in positions and momenta of particles located
    in the domain designated the calling process, as well as positions
    and preallocated nullified momentum changes for particles located in
    another domian. The two sets of particles are denoted i and j. The
    function computes the momentum changes due to gravity via direct
    summation. The two sets of particles can be the same, which is
    signalled by flag_input=0. That is, this function can also be used
    to compute interactions within a single domain. Use flag_input=1
    when using two different domains. Here, set i should be the
    particles belonging to the caller process. For these, momentum
    changes are added to the momenta. For set j, the momentum changes
    are computed but not added to the momenta, as these reside on a
    different process. Use flag_input=2 to skip the computation of the
    momentum changes of set j.
    """
    # No interactions if either of the two sets of particles are empty
    if N_local_i == 0 or N_local_j == 0:
        return
    # The factor (G*m_i*m_j*∫_t^(t + Δt) dt/a) in the
    # comoving equations of motion
    # p_i --> p_i + ∫_t^(t + Δt) F/a*dt = p_i + m_i*F*∫_t^(t + Δt) dt/a
    #       = p_i + (-G*m_i*m_j/r**2)*∫_t^(t + Δt) dt/a
    #       = p_i - 1/r**2*(G*m_i*m_j*∫_t^(t + Δt) dt/a)
    eom_factor = G_Newton*mass_i*mass_j*ᔑdt['a⁻¹']
    # Direct summation over all pairs of particles (i, j).
    # If both i and j are on the same process (flag_input == 0),
    # i should run from 0 to the second last particle index,
    # then j should run from i + 1 to the last particle index.
    force = vector
    i_end = N_local_i if flag_input > 0 else N_local_i - 1
    for i in range(i_end):
        xi = posx_i[i]
        yi = posy_i[i]
        zi = posz_i[i]
        # If both i and j are on the same process (flag_input == 0),
        # then j should run from i + 1 and up.
        # Otherwise, j should start from zero.
        with unswitch:
            if flag_input > 0:
                j_start = 0
            else:
                j_start = i + 1
        for j in range(j_start, N_local_j):
            x = posx_j[j] - xi
            y = posy_j[j] - yi
            z = posz_j[j] - zi
            # Evaluate the gravitational force in one of three ways:
            # Just the short range force, the total force with Ewald
            # corrections or the total force without Ewald corrections.
            with unswitch:
                if only_short_range:
                    # Translate coordinates so they
                    # correspond to the nearest image.
                    if x > ℝ[0.5*boxsize]:
                        x -= boxsize
                    elif x < ℝ[-0.5*boxsize]:
                        x += boxsize
                    if y > ℝ[0.5*boxsize]:
                        y -= boxsize
                    elif y < ℝ[-0.5*boxsize]:
                        y += boxsize
                    if z > ℝ[0.5*boxsize]:
                        z -= boxsize
                    elif z < ℝ[-0.5*boxsize]:
                        z += boxsize
                    r = sqrt(x**2 + y**2 + z**2 + softening2)
                    r_scaled = r*ℝ[1/p3m_scale_phys]
                    shortrange_fac = (  r_scaled*ℝ[1/sqrt(π)]*exp(-0.25*r_scaled**2)
                                      + erfc(0.5*r_scaled))
                    force[0] = -x*ℝ[1/r**3]*shortrange_fac
                    force[1] = -y*ℝ[1/r**3]*shortrange_fac
                    force[2] = -z*ℝ[1/r**3]*shortrange_fac
                elif enable_Ewald:
                    # Compute the gravitational force
                    # (corresponding to 1/r**2).
                    # Translate coordinates so they
                    # correspond to the nearest image.
                    if x > ℝ[0.5*boxsize]:
                        x -= boxsize
                    elif x < ℝ[-0.5*boxsize]:
                        x += boxsize
                    if y > ℝ[0.5*boxsize]:
                        y -= boxsize
                    elif y < ℝ[-0.5*boxsize]:
                        y += boxsize
                    if z > ℝ[0.5*boxsize]:
                        z -= boxsize
                    elif z < ℝ[-0.5*boxsize]:
                        z += boxsize
                    # The Ewald correction force for all 
                    # images except the nearest one,
                    # which may not be the actual particle.
                    force = ewald(x, y, z)
                    # Add in the force from the particle's nearest image
                    r3 = (x**2 + y**2 + z**2 + softening2)**1.5
                    force[0] -= x*ℝ[1/r3]
                    force[1] -= y*ℝ[1/r3]
                    force[2] -= z*ℝ[1/r3]
                else:
                    # The force from the actual particle,
                    # without periodic images.
                    r3 = (x**2 + y**2 + z**2 + softening2)**1.5
                    force[0] = -x*ℝ[1/r3]
                    force[1] = -y*ℝ[1/r3]
                    force[2] = -z*ℝ[1/r3]
            # Multiply the force by (G*m_i*m_j*∫_t^(t + Δt) dt/a).
            # Note that "force" is now really the momentum change.
            force[0] *= eom_factor
            force[1] *= eom_factor
            force[2] *= eom_factor
            # Update momenta and momentum changes.
            # Always update the momenta of particles
            # in the local domain (group i)
            momx_i[i] -= force[0]
            momy_i[i] -= force[1]
            momz_i[i] -= force[2]
            with unswitch:
                if flag_input == 0:
                    # Group i and j are the same (and belongs to the
                    # local domain).
                    # Also update momenta of particles of group j.
                    momx_i[j] += force[0]
                    momy_i[j] += force[1]
                    momz_i[j] += force[2]
                elif flag_input == 1:
                    # Group i and j are different (and j belongs to a
                    # different domain).
                    # Update the external momentum changes.
                    Δmomx_j[j] += force[0]
                    Δmomy_j[j] += force[1]
                    Δmomz_j[j] += force[2]

# Function for computing the gravitational force
# by direct summation on all particles
# (the particle-particle or PP method).
@cython.pheader(# Arguments
                component='Component',
                ᔑdt='dict',
                # Locals
                ID_recv='int',
                ID_send='int',
                N_extrn='Py_ssize_t',
                N_extrn_max='Py_ssize_t',
                N_extrns='Py_ssize_t[::1]',
                N_local='Py_ssize_t',
                N_partnerproc_pairs='int',
                even_nprocs='bint',
                factor='double',
                flag_input='int',
                i='Py_ssize_t',
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
def pp(component, ᔑdt):
    """ This function updates the momenta of all particles in the
    passed component via the particle-particle (PP) method.
    """
    global posx_extrn, posy_extrn, posz_extrn
    global Δmomx_local, Δmomy_local, Δmomz_local
    global Δmomx_extrn, Δmomy_extrn, Δmomz_extrn
    global posx_extrn_mv, posy_extrn_mv, posz_extrn_mv
    global Δmomx_local_mv, Δmomy_local_mv, Δmomz_local_mv
    global Δmomx_extrn_mv, Δmomy_extrn_mv, Δmomz_extrn_mv
    # Extract variables from the component
    N_local = component.N_local
    mass = component.mass
    momx_local    = component.momx
    momy_local    = component.momy
    momz_local    = component.momz
    posx_local    = component.posx
    posy_local    = component.posy
    posz_local    = component.posz
    posx_local_mv = component.posx_mv
    posy_local_mv = component.posy_mv
    posz_local_mv = component.posz_mv
    softening2    = component.softening**2
    # Update local momenta due to forces between local particles.
    # Note that "vector" is not actually used due to flag_input=0.
    direct_summation(posx_local, posy_local, posz_local,
                     momx_local, momy_local, momz_local,
                     mass, N_local,
                     posx_local, posy_local, posz_local,
                     vector, vector, vector,
                     mass, N_local,
                     ᔑdt, softening2,
                     0)
    # All work done if only one domain exists
    # (if run on a single process).
    if nprocs == 1:
        return
    # Update local momenta and compute and send external momentum
    # changes due to forces between local and external particles.
    # Find out how many particles will be received from each process.
    N_extrns = find_N_recv(np.array([N_local], dtype=C2np['Py_ssize_t']))
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
    # Number of pairs of process partners to send/receive data to/from
    even_nprocs = not (nprocs % 2)
    flag_input = 1
    N_partnerproc_pairs = 1 + nprocs//2
    N_partnerproc_pairs_minus_1 = N_partnerproc_pairs - 1
    for j in range(1, N_partnerproc_pairs):
        # Process ranks to send/receive to/from
        ID_send = mod(rank + j, nprocs)
        ID_recv = mod(rank - j, nprocs)
        N_extrn = N_extrns[ID_recv]
        # Send and receive positions
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
                         ᔑdt, softening2,
                         flag_input)
        # When flag_input == 2, no momentum updates has been computed.
        # Do not sent or receive these noncomputed updates.
        if flag_input == 2:
            continue
        # Send momentum updates back to the process from which
        # positions were received. Recieve momentum updates from the
        # process which the local positions were send to.
        Sendrecv(Δmomx_extrn_mv[:N_extrn], dest=ID_recv,
                 recvbuf=Δmomx_local_mv, source=ID_send)
        Sendrecv(Δmomy_extrn_mv[:N_extrn], dest=ID_recv,
                 recvbuf=Δmomy_local_mv, source=ID_send)
        Sendrecv(Δmomz_extrn_mv[:N_extrn], dest=ID_recv,
                 recvbuf=Δmomz_local_mv, source=ID_send)
        # Apply local momentum updates received from other process
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

# Function for computing the gravitational force
# by the particle mesh (PM) method.
@cython.header(# Arguments
               component='Component',
               ᔑdt='dict',
               gradφ_dim='double[:, :, ::1]',
               dim='int',
               # Locals
               J_dim='FluidScalar',
               i='Py_ssize_t',
               mom_dim='double*',
               pm_fac='double',
               posx='double*',
               posy='double*',
               posz='double*',
               x='double',
               y='double',
               z='double',
               )
def pm(component, ᔑdt, gradφ_dim, dim):
    """This function updates the momenta of all particles/fluid elements
    of a component via the particle-mesh (PM) method.
    """
    if component.representation == 'particles':
        # Extract variables from component
        posx    = component.posx
        posy    = component.posy
        posz    = component.posz
        mom_dim = component.mom[dim]
        # The constant factors with which to multiply the values
        # in gradφ_dim to actually get the negative differentiated
        # potential -[∇φ]_dim is gathered in pm_fac_const. The total
        # factor to multiply gradφ_dim by to get momentum updates is
        # then pm_fac*mass*Δt, where Δt = ᔑdt['1'].
        pm_fac = pm_fac_const*component.mass*ᔑdt['1']
        # Update the dim momentum component of particle i
        for i in range(component.N_local):
            # The coordinates of the i'th particle,
            # transformed so that 0 <= x, y, z < 1.
            x = (posx[i] - domain_start_x)/domain_size_x
            y = (posy[i] - domain_start_y)/domain_size_y
            z = (posz[i] - domain_start_z)/domain_size_z
            # Look up the force via a CIC interpolation,
            # convert it to momentum units and add it to the
            # momentum of particle i.
            mom_dim[i] += pm_fac*CIC_scalargrid2coordinates(gradφ_dim, x, y, z)
    elif component.representation == 'fluid':
        # Simply scale and extrapolate the values in gradφ_dim
        # to the grid points of the dim'th component of the
        # fluid variable J.
        # First extract this fluid scalar.
        J_dim = component.J[dim]
        # The constant factors with which to multiply the values in
        # gradφ_dim in order to get the negative differentiated
        # potential -[∇φ]_dim is gathered in pm_fac_const.
        # As the gravitational source term is -a⁻³ʷ*ϱ*∇φ, we also
        # need to multiply each grid point [i, j, k] by ϱ[i, j, k]
        # and all grid points by the same factor a⁻³ʷ. Actually,
        # since what we are after are the updates to the momentum
        # density, we should also multiply by Δt. Since a⁻³ʷ is time
        # dependent, we should then really exchange a⁻³ʷ*Δt for ᔑa⁻³ʷdt.
        CIC_grid2grid(J_dim.grid_noghosts,
                      gradφ_dim,
                      fac=pm_fac_const*ᔑdt['a⁻³ʷ', component],
                      fac_grid=component.ϱ.grid_noghosts,
                      )
        # Communicate the pseudo and ghost points of J_dim
        communicate_domain(J_dim.grid_mv, mode='populate')

# Function which constructs the total gravitational potential φ due
# to all components.
@cython.header(# Arguments
               components='list',
               ᔑdt='dict',
               only_long_range='bint',
               # Locals
               Greens_deconv='double',
               factors='list',
               i='Py_ssize_t',
               j='Py_ssize_t',
               j_global='Py_ssize_t',
               k='Py_ssize_t',
               ki='Py_ssize_t',
               ki2_plus_kj2='Py_ssize_t',
               kj='Py_ssize_t',
               kj2='Py_ssize_t',
               kk='Py_ssize_t',
               k2='Py_ssize_t',
               slab_jik='double*',
               sqrt_deconv_ij='double',
               sqrt_deconv_ijk='double',
               sqrt_deconv_j='double',
               Δt='double',
               returns='double[:, :, ::1]',
               )
def build_φ(components, ᔑdt, only_long_range=False):
    """This function computes the gravitational potential φ due to
    all components given in the components argument.
    Pseudo points and ghost layers will be communicated.
    The Poisson equation which is solved by this function is
    ∇²φ = 4πG Σᵢ(1/Δt*∫_t^(t + Δt)a⁻³ʷ⁻¹dt ϱᵢ),
    where the sum is over all species. For a single w = 0 component
    this reduces to
    ∇²φ = 4πG 1/Δt*∫_t^(t + Δt) a⁻¹dt ϱ,
    where 1/Δt*∫_t^(t + Δt) a⁻¹dt is just the average value of a⁻¹ over
    the time step where φ should be applied. All of these integrals
    should be supplied as the ᔑdt argument.
    The P3M long-range potential can be computed instead of the regular
    potential by specifying only_long_range = True.
    """
    if not use_φ:
        masterwarn('The φ mesh is not initialized. '
                   'Have you specified φ_gridsize in the parameter file?')
    if only_long_range:
        masterprint('Computing the long-range gravitational potential ...')
    else:
        masterprint('Computing the gravitational potential ...')
    # Pull out the needed integrals for each component
    Δt = ᔑdt['1']
    factors = [ℝ[1/Δt]*ᔑdt['a⁻³ʷ⁻¹', component] for component in components]
    # CIC interpolate the particles/fluid elements onto the slabs
    CIC_components2slabs(components, factors)
    # Do forward Fourier transform on the slabs
    # containing the density field.
    slabs_FFT()
    # Loop through the local j-dimension
    for j in range(slab_size_j):
        # The j-component of the wave vector. Since the slabs are
        # distributed along the j-dimension, an offset must be used.
        j_global = slab_start_j + j
        if j_global > φ_gridsize_half:
            kj = j_global - φ_gridsize
        else:
            kj = j_global
        kj2 = kj**2
        # Square root of the j-component of the deconvolution
        sqrt_deconv_j = sinc(kj*ℝ[π/φ_gridsize])
        # Loop through the complete i-dimension
        for i in range(φ_gridsize):
            # The i-component of the wave vector
            if i > φ_gridsize_half:
                ki = i - φ_gridsize
            else:
                ki = i
            ki2_plus_kj2 = ki**2 + kj2
            # Square root of the product of the i-
            # and the j-component of the deconvolution.
            sqrt_deconv_ij = sinc(ki*ℝ[π/φ_gridsize])*sqrt_deconv_j
            # Loop through the complete, padded k-dimension
            # in steps of 2 (one complex number at a time).
            for k in range(0, slab_size_padding, 2):
                # The k-component of the wave vector
                kk = k//2
                # The squared magnitude of the wave vector
                k2 = ki2_plus_kj2 + kk**2
                # Zero-division is illegal in pure Python.
                # The global [0, 0, 0] element of the slabs will be set
                # later anyway.
                if not cython.compiled:
                    if k2 == 0:
                        continue
                # Square root of the product of
                # all components of the deconvolution.
                sqrt_deconv_ijk = sqrt_deconv_ij*sinc(kk*ℝ[π/φ_gridsize])
                # Pointer to the [j, i, k]'th element of the slab.
                # The complex number is then given as
                # Re = slab_jik[0], Im = slab_jik[1].
                slab_jik = cython.address(slab[j, i, k:])
                # Multiply by the Greens function 1/k2 to get the
                # potential. Deconvolve twice for the two CIC
                # interpolations (the mass assignment and the upcomming
                # force interpolation). Remember that the slab is
                # transposed in the first two dimensions due to the
                # forward FFT.
                Greens_deconv = 1/(k2*sqrt_deconv_ijk**4)
                if only_long_range:
                    Greens_deconv *= exp(k2*longrange_exponent_fac)
                slab_jik[0] *= Greens_deconv  # Real part
                slab_jik[1] *= Greens_deconv  # Imag part
    # The global [0, 0, 0] element of the slabs should be zero
    if slab_start_j == 0:
        slab[0, 0, 0] = 0  # Real part
        slab[0, 0, 1] = 0  # Imag part
    # Fourier transform the slabs back to coordinate space.
    # Now the slabs stores potential values.
    slabs_IFFT()
    # Communicate the potential stored in the slabs to φ
    slabs2φ()  # This also populates pseudo and ghost points
    # Finalize progress message
    masterprint('done')
    # Return the potential grid (though this is a global and is often
    # imported directly into other modules).
    return φ

# This collection of functions simply test whether or not the passed
# coordinates lie within a certain domain boundary.
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_right(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i > boundary_x_max
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_left(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i < boundary_x_min
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_forward(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i > boundary_y_max
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_backward(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i < boundary_y_min
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_up(posx_local_i, posy_local_i, posz_local_i):
    return posz_local_i > boundary_z_max
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_down(posx_local_i, posy_local_i, posz_local_i):
    return posz_local_i < boundary_z_min
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_rightforward(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i > boundary_x_max and posy_local_i > boundary_y_max
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_rightbackward(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i > boundary_x_max and posy_local_i < boundary_y_min
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_rightup(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i > boundary_x_max and posz_local_i > boundary_z_max
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_rightdown(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i > boundary_x_max and posz_local_i < boundary_z_min
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_leftforward(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i < boundary_x_min and posy_local_i > boundary_y_max
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_leftbackward(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i < boundary_x_min and posy_local_i < boundary_y_min
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_leftup(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i < boundary_x_min and posz_local_i > boundary_z_max
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_leftdown(posx_local_i, posy_local_i, posz_local_i):
    return posx_local_i < boundary_x_min and posz_local_i < boundary_z_min
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_forwardup(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i > boundary_y_max and posz_local_i > boundary_z_max
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_forwarddown(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i > boundary_y_max and posz_local_i < boundary_z_min
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_backwardup(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i < boundary_y_min and posz_local_i > boundary_z_max
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_backwarddown(posx_local_i, posy_local_i, posz_local_i):
    return posy_local_i < boundary_y_min and posz_local_i < boundary_z_min
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_rightforwardup(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i > boundary_x_max and posy_local_i > boundary_y_max
                                          and posz_local_i > boundary_z_max)
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_rightforwarddown(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i > boundary_x_max and posy_local_i > boundary_y_max
                                          and posz_local_i < boundary_z_min)
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_rightbackwardup(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i > boundary_x_max and posy_local_i < boundary_y_min
                                          and posz_local_i > boundary_z_max)
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_rightbackwarddown(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i > boundary_x_max and posy_local_i < boundary_y_min
                                          and posz_local_i < boundary_z_min)
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_leftforwardup(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i < boundary_x_min and posy_local_i > boundary_y_max
                                          and posz_local_i > boundary_z_max)
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_leftforwarddown(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i < boundary_x_min and posy_local_i > boundary_y_max
                                          and posz_local_i < boundary_z_min)
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_leftbackwardup(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i < boundary_x_min and posy_local_i < boundary_y_min
                                          and posz_local_i > boundary_z_max)
@cython.header(# Arguments
               posx_local_i='double',
               posy_local_i='double',
               posz_local_i='double',
               returns='bint',
               )
def in_boundary_leftbackwarddown(posx_local_i, posy_local_i, posz_local_i):
    return (posx_local_i < boundary_x_min and posy_local_i < boundary_y_min
                                          and posz_local_i < boundary_z_min)

# Function for computing the gravitational force
# by the particle-particle-particle mesh (P³M) method.
@cython.pheader(# Arguments
               component='Component',
               ᔑdt='dict',
               # Locals
               N_extrn='Py_ssize_t',
               N_local='Py_ssize_t',
               N_boundary1='Py_ssize_t',
               N_boundary2='Py_ssize_t',
               dim='int',
               h='double',
               i='Py_ssize_t',
               in_boundary1='func_b_ddd',
               in_boundary2='func_b_ddd',
               j='Py_ssize_t',
               mass='double',
               meshbuf_mv='double[:, :, ::1]',
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
               Δmemory='Py_ssize_t',
               )
def p3m(component, ᔑdt):
    """The long-range part is computed via the pm function. Local
    particles also interact via short-range direct summation. Finally,
    each process send particles near its boundary to the corresponding
    processor, which computes the short-range forces via direct
    summation between the received boundary particles and its own
    boundary particles at the opposite face/edge/point. This is done by
    iterating over pairs of neighboring processes. In the first
    iteration, particles in the right boundary are sent to the right
    process, while particles in the left process' right boundary are
    received. The boundary containing particles which are not sent is
    refered to as boundary1, while the other is refered to as boundary2.
    """
    global posx_local_boundary, posy_local_boundary, posz_local_boundary
    global posx_local_boundary_mv, posy_local_boundary_mv
    global posz_local_boundary_mv
    global Δmomx_local_boundary, Δmomy_local_boundary
    global Δmomz_local_boundary
    global Δmomx_local_boundary_mv, Δmomy_local_boundary_mv
    global Δmomz_local_boundary_mv
    global posx_extrn, posy_extrn, posz_extrn
    global posx_extrn_mv, posy_extrn_mv, posz_extrn_mv
    global Δmomx_extrn, Δmomy_extrn, Δmomz_extrn
    global Δmomx_extrn_mv, Δmomy_extrn_mv, Δmomz_extrn_mv
    global Δmomx_local, Δmomy_local, Δmomz_local
    global Δmomx_local_mv, Δmomy_local_mv, Δmomz_local_mv
    global indices_send, indices_send_mv
    global indices_boundary, indices_boundary_mv
    
    # NOTE: This function now only supply the short-range force
    # and should therefore be renamed !!!
    #

    # Extract variables from component
    N_local    = component.N_local
    mass       = component.mass
    momx_local = component.momx
    momy_local = component.momy
    momz_local = component.momz
    posx_local = component.posx
    posy_local = component.posy
    posz_local = component.posz
    softening2 = component.softening**2
    # Compute the short-range interactions within the local domain.
    # Note that "vector" is not actually used due to flag_input=0.
    direct_summation(posx_local, posy_local, posz_local,
                     momx_local, momy_local, momz_local,
                     mass, N_local,
                     posx_local, posy_local, posz_local,
                     vector, vector, vector,
                     mass, N_local,
                     ᔑdt, softening2, 0,
                     only_short_range=True)
    # All work done if only one domain
    # exists (if run on a single process)
    if nprocs == 1:
        return
    # Now only short-range interactions between neighboring domain
    # boundaries remain.
    # The buffers below may increase their size by this amount at a
    # time. Here we cast to a Python int (which is then implicitly
    # casted to a Py_ssize_t) since np.float64 cannot be explicitly
    # casted to Py_ssize_t.
    Δmemory = 2 + int(0.05*N_local*np.max([p3m_cutoff_phys/domain_size_x,
                                           p3m_cutoff_phys/domain_size_y,
                                           p3m_cutoff_phys/domain_size_z,
                                           ]))
    # Loop over all 26 neighbors (two at a time)
    for j in range(13):
        # It is important that the processes iterate synchronously,
        # so that the received data really is what the local process
        # thinks it is.
        Barrier()
        # The ranks of the processes to communicate with
        rank_send = boundary_ranks_send[j]
        rank_recv = boundary_ranks_recv[j]
        # The functions for in-boundary tests
        in_boundary1 = in_boundary1_funcs[j]
        in_boundary2 = in_boundary2_funcs[j]
        # Find out which particles participate in the
        # local and the external short-range interaction.
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
                    indices_boundary = realloc(indices_boundary,
                                               (N_boundary1 + Δmemory)*sizeof('Py_ssize_t'))
                    indices_boundary_mv = cast(indices_boundary,
                                               'Py_ssize_t[:(N_boundary1 + Δmemory)]')
                    posx_local_boundary = realloc(posx_local_boundary,
                                                  (N_boundary1 + Δmemory)*sizeof('double'))
                    posx_local_boundary_mv = cast(posx_local_boundary,
                                                  'double[:(N_boundary1 + Δmemory)]')
                    posy_local_boundary = realloc(posy_local_boundary,
                                                  (N_boundary1 + Δmemory)*sizeof('double'))
                    posy_local_boundary_mv = cast(posy_local_boundary,
                                                  'double[:(N_boundary1 + Δmemory)]')
                    posz_local_boundary = realloc(posz_local_boundary,
                                                  (N_boundary1 + Δmemory)*sizeof('double'))
                    posz_local_boundary_mv = cast(posz_local_boundary,
                                                  'double[:(N_boundary1 + Δmemory)]')
                    Δmomx_local_boundary = realloc(Δmomx_local_boundary,
                                                   (N_boundary1 + Δmemory)*sizeof('double'))
                    Δmomx_local_boundary_mv = cast(Δmomx_local_boundary,
                                                   'double[:(N_boundary1 + Δmemory)]')
                    Δmomy_local_boundary = realloc(Δmomy_local_boundary,
                                                   (N_boundary1 + Δmemory)*sizeof('double'))
                    Δmomy_local_boundary_mv = cast(Δmomy_local_boundary,
                                                   'double[:(N_boundary1 + Δmemory)]')
                    Δmomz_local_boundary = realloc(Δmomz_local_boundary,
                                                   (N_boundary1 + Δmemory)*sizeof('double'))
                    Δmomz_local_boundary_mv = cast(Δmomz_local_boundary,
                                                   'double[:(N_boundary1 + Δmemory)]')
            # Check if particle should be sent to the right
            if in_boundary2(posx_local_i, posy_local_i, posz_local_i):
                # Particle i should be send
                indices_send[N_boundary2] = i
                # Fill buffers with their coordinate.
                # The Δmom(x/y/z)_local variables are used for this to
                # save memory, as these otherwise first used after the
                # particles have been communicated.
                Δmomx_local[N_boundary2] = posx_local_i
                Δmomy_local[N_boundary2] = posy_local_i
                Δmomz_local[N_boundary2] = posz_local_i
                N_boundary2 += 1
                # Enlarge buffers if needed
                if indices_send_mv.shape[0] == N_boundary2:
                    indices_send = realloc(indices_send,
                                           (N_boundary2 + Δmemory)*sizeof('Py_ssize_t'))
                    indices_send_mv = cast(indices_send, 'Py_ssize_t[:(N_boundary2 + Δmemory)]')
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
                         ᔑdt, softening2, 1,
                         only_short_range=True)
        # Apply the momentum changes to the local particle momentum data
        for i in range(N_boundary1):
            momx_local[indices_boundary[i]] += Δmomx_local_boundary[i]
            momy_local[indices_boundary[i]] += Δmomy_local_boundary[i]
            momz_local[indices_boundary[i]] += Δmomz_local_boundary[i]
        # Send momentum updates back to the process from which
        # positions were received. Recieve momentum updates from the
        # process which the local positions were send to.
        Sendrecv(Δmomx_extrn_mv[:N_extrn], dest=rank_recv,
                 recvbuf=Δmomx_local_mv, source=rank_send)
        Sendrecv(Δmomy_extrn_mv[:N_extrn], dest=rank_recv,
                 recvbuf=Δmomy_local_mv, source=rank_send)
        Sendrecv(Δmomz_extrn_mv[:N_extrn], dest=rank_recv,
                 recvbuf=Δmomz_local_mv, source=rank_send)
        # Apply local momentum updates received from other process
        for i in range(N_boundary2):
            momx_local[indices_send[i]] += Δmomx_local[i]
            momy_local[indices_send[i]] += Δmomy_local[i]
            momz_local[indices_send[i]] += Δmomz_local[i]



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
               indices_boundary='Py_ssize_t*',
               indices_boundary_mv='Py_ssize_t[::1]',
               indices_send='Py_ssize_t*',
               indices_send_mv='Py_ssize_t[::1]',
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
indices_send = malloc(1*sizeof('Py_ssize_t'))
indices_send_mv = cast(indices_send, 'Py_ssize_t[:1]')
# For storing the indices of local particles which should interact with
# boundaries of other domains in the P3M method.
indices_boundary = malloc(1*sizeof('Py_ssize_t'))
indices_boundary_mv = cast(indices_send, 'Py_ssize_t[:1]')
# For storing a copy of those local particles
# that consitutes the short-range domain boundaries.
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
# Save the neighboring ranks in a particular order,
# for use in the P3M algorithm
boundary_ranks_send = np.array([rank_neighboring_domain(+1,  0,  0),
                                rank_neighboring_domain( 0, +1,  0),
                                rank_neighboring_domain( 0,  0, +1),
                                rank_neighboring_domain(+1, +1,  0),
                                rank_neighboring_domain(+1, -1,  0),
                                rank_neighboring_domain(+1,  0, +1),
                                rank_neighboring_domain(+1,  0, -1),
                                rank_neighboring_domain( 0, +1, +1),
                                rank_neighboring_domain( 0, +1, -1),
                                rank_neighboring_domain(+1, +1, +1),
                                rank_neighboring_domain(+1, +1, -1),
                                rank_neighboring_domain(+1, -1, +1),
                                rank_neighboring_domain(+1, -1, -1),
                                ], dtype=C2np['int'])
boundary_ranks_recv = np.array([rank_neighboring_domain(-1,  0,  0),
                                rank_neighboring_domain( 0, -1,  0),
                                rank_neighboring_domain( 0,  0, -1),
                                rank_neighboring_domain(-1, -1,  0),
                                rank_neighboring_domain(-1, +1,  0),
                                rank_neighboring_domain(-1,  0, -1),
                                rank_neighboring_domain(-1,  0, +1),
                                rank_neighboring_domain( 0, -1, -1),
                                rank_neighboring_domain( 0, -1, +1),
                                rank_neighboring_domain(-1, -1, -1),
                                rank_neighboring_domain(-1, -1, +1),
                                rank_neighboring_domain(-1, +1, -1),
                                rank_neighboring_domain(-1, +1, +1),
                                ], dtype=C2np['int'])
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
boundary_x_max = domain_start_x + domain_size_x - p3m_cutoff_phys
boundary_x_min = domain_start_x + p3m_cutoff_phys
boundary_y_max = domain_start_y + domain_size_y - p3m_cutoff_phys
boundary_y_min = domain_start_y + p3m_cutoff_phys
boundary_z_max = domain_start_z + domain_size_z - p3m_cutoff_phys
boundary_z_min = domain_start_z + p3m_cutoff_phys
