# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2019 Jeppe Mosgaard Dakin.
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
cimport('from ewald import ewald')
cimport('from communication import '
    'domain_subdivisions, domain_start_x, domain_start_y, domain_start_z, '
    'domain_size_x, domain_size_y, domain_size_z, '
)



# Function implementing pairwise gravity
@cython.header(
    # Arguments
    receiver='Component',
    supplier='Component',
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier='Py_ssize_t[::1]',
    rank_supplier='int',
    only_supply='bint',
    ᔑdt=dict,
    extra_args=dict,
    # Locals
    N_r='Py_ssize_t',
    N_s='Py_ssize_t',
    force_ij='double*',
    forcex_ij='double',
    forcey_ij='double',
    forcez_ij='double',
    i='Py_ssize_t',
    i_end='Py_ssize_t',
    j='Py_ssize_t',
    j_end='Py_ssize_t',
    j_start='Py_ssize_t',
    mass_r='double',
    mass_s='double',
    momx_r='double*',
    momx_s='double*',
    momy_r='double*',
    momy_s='double*',
    momz_r='double*',
    momz_s='double*',
    only_short_range='bint',
    pairing_level=str,
    particle_indices_r='Py_ssize_t*',
    particle_indices_s='Py_ssize_t*',
    periodic='bint',
    posx_r='double*',
    posx_s='double*',
    posy_r='double*',
    posy_s='double*',
    posz_r='double*',
    posz_s='double*',
    r='double',
    r_scaled='double',
    r2='double',
    r2_max='double',
    r3='double',
    shortrange_fac='double',
    softening_r='double',
    softening_s='double',
    tile_index_r='Py_ssize_t',
    tile_index_s='Py_ssize_t',
    tiles_N_r='Py_ssize_t[::1]',
    tiles_N_s='Py_ssize_t[::1]',
    tiles_particle_indices_r='Py_ssize_t**',
    tiles_particle_indices_s='Py_ssize_t**',
    x_ji='double',
    xi='double',
    y_ji='double',
    yi='double',
    z_ji='double',
    zi='double',
    Δmomx_s='double*',
    Δmomx_ij='double',
    Δmomy_s='double*',
    Δmomy_ij='double',
    Δmomz_s='double*',
    Δmomz_ij='double',
    returns='void',
)
def gravity_pairwise(
    receiver, supplier, tile_indices_receiver, tile_indices_supplier,
    rank_supplier, only_supply, ᔑdt, extra_args,
):
    # Extract extra arguments
    only_short_range = extra_args.get('only_short_range', False)
    periodic         = extra_args.get('periodic',         True)
    # Extract variables from the receiver component
    N_r = receiver.N_local
    mass_r = receiver.mass
    softening_r = receiver.softening_length
    posx_r = receiver.posx
    posy_r = receiver.posy
    posz_r = receiver.posz
    momx_r = receiver.momx
    momy_r = receiver.momy
    momz_r = receiver.momz
    tiles_N_r = receiver.tiles_N_linear
    tiles_particle_indices_r = receiver.tiles_particle_indices_linear
    # Extract variables from the supplier (the external) component
    N_s = supplier.N_local
    mass_s = supplier.mass
    softening_s = supplier.softening_length
    posx_s = supplier.posx
    posy_s = supplier.posy
    posz_s = supplier.posz
    momx_s = supplier.momx
    momy_s = supplier.momy
    momz_s = supplier.momz
    Δmomx_s = supplier.Δmomx
    Δmomy_s = supplier.Δmomy
    Δmomz_s = supplier.Δmomz
    tiles_N_s = supplier.tiles_N_linear
    tiles_particle_indices_s = supplier.tiles_particle_indices_linear
    # The squared maximum distance the short-range force can reach.
    # The force between particle pairs with larger squared separation
    # that this will be ignored, if we are computing the short-range
    # force only.
    if only_short_range:
        r2_max = ℝ[(p3m_cutoff*(p3m_scale*boxsize/φ_gridsize))**2]
    # Infer pairing level ('domain' or 'tile') from passed tile indices
    if tile_indices_receiver is None:
        pairing_level = 'domain'
        # This is just to create one-iteration loops.
        # The tiling will not be used.
        tile_indices_receiver = tile_indices_supplier = np.zeros(1, dtype=C2np['Py_ssize_t'])
    else:
        pairing_level = 'tile'
    # To satisfy the compiler
    particle_indices_r = tiles_particle_indices_r[0]
    particle_indices_s = tiles_particle_indices_s[0]
    # Loop over the requested tiles in the receiver
    for tile_index_r in range(tile_indices_receiver.shape[0]):
        tile_index_r = tile_indices_receiver[tile_index_r]
        # Prepare for the loop over receiver particles
        with unswitch:
            if 𝔹[pairing_level == 'tile']:
                i_end = tiles_N_r[tile_index_r]
                particle_indices_r = tiles_particle_indices_r[tile_index_r]
            else:  # pairing_level == 'domain'
                i_end = N_r
        # Loop over the requested tiles in the supplier
        for tile_index_s in range(tile_indices_supplier.shape[0]):
            tile_index_s = tile_indices_supplier[tile_index_s]
            # Prepare for the loop over supplier particle
            with unswitch:
                if 𝔹[pairing_level == 'tile']:
                    j_end = tiles_N_s[tile_index_s]
                    particle_indices_s = tiles_particle_indices_s[tile_index_s]
                else:  # pairing_level == 'domain'
                    j_end = N_s
            # Loop over all receiver particles in the domain
            # (pairing_level == 'domain') or just this tile
            # (pairing_level == 'tile').
            for i in range(i_end):
                # If the receiver and supplier component are one and the
                # same and the two paired domains are one and the same,
                # we need to make sure not to
                # double count the particles.
                with unswitch:
                    if 𝔹[receiver.name == supplier.name and rank == rank_supplier]:
                        with unswitch:
                            if 𝔹[pairing_level == 'tile']:
                                # When using tiles, double counting is
                                # only possible when the current
                                # receiver and supplier tile are
                                # one and the same.
                                with unswitch(1):
                                    if tile_index_r == tile_index_s:
                                        j_start = i + 1
                                    else:
                                        j_start = 0
                            else:  # pairing_level == 'domain'
                                j_start = i + 1
                    else:
                        j_start = 0
                # Prepare for the loop over supplier particle
                with unswitch:
                    if 𝔹[pairing_level == 'tile']:
                        i = particle_indices_r[i]
                # Coordinates of receiver particle
                xi = posx_r[i]
                yi = posy_r[i]
                zi = posz_r[i]
                # Loop over all supplier particles in the domain
                # (pairing_level == 'domain') or just this tile
                # (pairing_level == 'tile').
                for j in range(j_start, j_end):
                    with unswitch:
                        if 𝔹[pairing_level == 'tile']:
                            j = particle_indices_s[j]
                    # "Vector" from particle j to particle i
                    x_ji = xi - posx_s[j]
                    y_ji = yi - posy_s[j]
                    z_ji = zi - posz_s[j]
                    # Evaluate the gravitational force in one of
                    # three ways:
                    # - Just the short range force
                    # - The total force with Ewald corrections
                    # - The total force without Ewald corrections
                    with unswitch:
                        if only_short_range:
                            # Translate coordinates so they
                            # correspond to the nearest image.
                            if x_ji > ℝ[0.5*boxsize]:
                                x_ji -= boxsize
                            elif x_ji < ℝ[-0.5*boxsize]:
                                x_ji += boxsize
                            if y_ji > ℝ[0.5*boxsize]:
                                y_ji -= boxsize
                            elif y_ji < ℝ[-0.5*boxsize]:
                                y_ji += boxsize
                            if z_ji > ℝ[0.5*boxsize]:
                                z_ji -= boxsize
                            elif z_ji < ℝ[-0.5*boxsize]:
                                z_ji += boxsize
                            r2 = x_ji**2 + y_ji**2 + z_ji**2
                            # If the particle pair is separated by a
                            # distance larger than the range of the
                            # short-range force, ignore this
                            # interaction completely.
                            if r2 > r2_max:
                                continue
                            # Compute the short-range force
                            r = sqrt(r2 + ℝ[(0.5*(softening_r + softening_s))**2])
                            r_scaled = r*ℝ[1/(p3m_scale*boxsize/φ_gridsize)]
                            shortrange_fac = (
                                r_scaled*ℝ[1/sqrt(π)]*exp(-0.25*r_scaled**2) + erfc(0.5*r_scaled)
                            )
                            forcex_ij = x_ji*ℝ[-shortrange_fac/r**3]
                            forcey_ij = y_ji*ℝ[-shortrange_fac/r**3]
                            forcez_ij = z_ji*ℝ[-shortrange_fac/r**3]
                        elif periodic:
                            # Translate coordinates so they
                            # correspond to the nearest image.
                            if x_ji > ℝ[0.5*boxsize]:
                                x_ji -= boxsize
                            elif x_ji < ℝ[-0.5*boxsize]:
                                x_ji += boxsize
                            if y_ji > ℝ[0.5*boxsize]:
                                y_ji -= boxsize
                            elif y_ji < ℝ[-0.5*boxsize]:
                                y_ji += boxsize
                            if z_ji > ℝ[0.5*boxsize]:
                                z_ji -= boxsize
                            elif z_ji < ℝ[-0.5*boxsize]:
                                z_ji += boxsize
                            # The Ewald correction force for all
                            # images except the nearest one,
                            # which might not be the actual particle.
                            force_ij = ewald(x_ji, y_ji, z_ji)
                            # Add in the force from the particle's
                            # nearest image.
                            r3 = (x_ji**2 + y_ji**2 + z_ji**2
                                + ℝ[(0.5*(softening_r + softening_s))**2])**1.5
                            forcex_ij = force_ij[0] - x_ji*ℝ[1/r3]
                            forcey_ij = force_ij[1] - y_ji*ℝ[1/r3]
                            forcez_ij = force_ij[2] - z_ji*ℝ[1/r3]
                        else:
                            # The force from the actual particle,
                            # without periodic images.
                            r3 = (x_ji**2 + y_ji**2 + z_ji**2
                                + ℝ[(0.5*(softening_r + softening_s))**2])**1.5
                            forcex_ij = x_ji*ℝ[-1/r3]
                            forcey_ij = y_ji*ℝ[-1/r3]
                            forcez_ij = z_ji*ℝ[-1/r3]
                    # Convert force on particle i from particle j to
                    # momentum change of partcicle i due to particle j.
                    # This looks like
                    #   Δmom = force*G*mass_r*mass_s/a.
                    # In the general case of decaying particles,
                    # the mass of each receiver particle is
                    #   mass_r(a) = receiver.mass*a**(-3*receiver.w_eff(a)),
                    # and likewise for the supplier particles. Below we
                    # integrate over the entire time dependence.
                    Δmomx_ij = forcex_ij*ℝ[G_Newton*mass_r*mass_s
                        *ᔑdt['a**(-3*w_eff₀-3*w_eff₁-1)', receiver.name, supplier.name]]
                    Δmomy_ij = forcey_ij*ℝ[G_Newton*mass_r*mass_s
                        *ᔑdt['a**(-3*w_eff₀-3*w_eff₁-1)', receiver.name, supplier.name]]
                    Δmomz_ij = forcez_ij*ℝ[G_Newton*mass_r*mass_s
                        *ᔑdt['a**(-3*w_eff₀-3*w_eff₁-1)', receiver.name, supplier.name]]
                    # Apply momentum change to particle i
                    # of the receiver.
                    momx_r[i] += Δmomx_ij
                    momy_r[i] += Δmomy_ij
                    momz_r[i] += Δmomz_ij
                    # Apply or save the momentum change of particle j
                    # of the supplier (the external component).
                    with unswitch:
                        if 𝔹[not only_supply and rank == rank_supplier]:
                            # This interaction is exlusively within the
                            # local domain. Apply momentum changes
                            # directly to particle j.
                            momx_s[j] -= Δmomx_ij
                            momy_s[j] -= Δmomy_ij
                            momz_s[j] -= Δmomz_ij
                        elif 𝔹[not only_supply]:
                            # Add momentum change to the external
                            # Δmom buffers of the supplier.
                            Δmomx_s[j] -= Δmomx_ij
                            Δmomy_s[j] -= Δmomy_ij
                            Δmomz_s[j] -= Δmomz_ij

# Function implementing the gravitational potential (in Fouier space).
# Here k2 = k² is the squared magnitude of the wave vector,
# in physical units.
@cython.header(
    k2='double',
    returns='double',
)
def gravity_potential(k2):
    return ℝ[-4*π*G_Newton]/k2

# Function implementing just the long-range part
# of the gravitational potential (in Fouier space).
# Here k2 = k² is the squared magnitude of the wave vector,
# in physical units.
@cython.header(
    k2='double',
    returns='double',
)
def gravity_longrange_potential(k2):
    return exp(-k2*ℝ[(p3m_scale*boxsize/φ_gridsize)**2])*gravity_potential(k2)

