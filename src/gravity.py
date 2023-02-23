# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2021 Jeppe Mosgaard Dakin.
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
# along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from ewald import ewald')
cimport(
    'from interactions import       '
    '    combine_softening_lengths, '
    '    get_softened_r3inv,        '
    '    particle_particle,         '
)



# Function for computing the gravitational factor
# G*mass_r*mass_s*Δt/a for all time step integrals.
@cython.header(
    # Arguments
    receiver='Component',
    supplier='Component',
    ᔑdt_rungs=dict,
    # Locals
    rung_index='signed char',
    ᔑdt_arr='double[::1]',
    ᔑdt_ptr='double*',
    returns='const double*',
)
def compute_factors(receiver, supplier, ᔑdt_rungs):
    # This function computes the factors used for momentum updates
    # due to gravitational interaction:
    #   Δmom = -r⃗/r³*(G*mass_r*mass_s*Δt/a).
    # In the general case of decaying particles,
    # the mass of each particle is
    #   mass(a) = component.mass*a**(-3*component.w_eff(a=a)).
    # For each rung we integrate over the time step through lookup
    # in the supplied ᔑdt_rungs dict.
    ᔑdt_arr = ᔑdt_rungs['a**(-3*w_eff₀-3*w_eff₁-1)', receiver.name, supplier.name]
    ᔑdt_ptr = cython.address(ᔑdt_arr[:])
    for rung_index in range(ᔑdt_arr.shape[0]):
        factors[rung_index] = ℝ[G_Newton*receiver.mass*supplier.mass]*ᔑdt_ptr[rung_index]
    return factors
# Global array used by the compute_factors() function
cython.declare(factors='double*')
factors = malloc(3*N_rungs*sizeof('double'))

# Function implementing pairwise gravity (full/periodic)
@cython.nounswitching
@cython.header(
    # Arguments
    interaction_name=str,
    receiver='Component',
    supplier='Component',
    ᔑdt_rungs=dict,
    rank_supplier='int',
    only_supply='bint',
    pairing_level=str,
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier_paired='Py_ssize_t**',
    tile_indices_supplier_paired_N='Py_ssize_t*',
    extra_args=dict,
    # Locals
    apply_to_i='bint',
    apply_to_j='bint',
    factor_i='double',
    factor_j='double',
    factors='const double*',
    force_ij='double*',
    forcex_ij='double',
    forcey_ij='double',
    forcez_ij='double',
    indexᵖ_j='Py_ssize_t',
    indexˣ_i='Py_ssize_t',
    indexˣ_j='Py_ssize_t',
    particle_particle_t_begin='double',
    particle_particle_t_final='double',
    periodic_offset_x='double',
    periodic_offset_y='double',
    periodic_offset_z='double',
    r2='double',
    r3_inv_softened='double',
    rung_index_i='signed char',
    rung_index_j='signed char',
    rung_index_s='signed char',
    rung_indices_jumped_s='signed char*',
    softening='double',
    subtile_contain_jumping_s='bint',
    subtiling_r='Tiling',
    x_ji='double',
    y_ji='double',
    z_ji='double',
    Δmom_r='double*',
    Δmom_s='double*',
    Δmomx='double',
    Δmomy='double',
    Δmomz='double',
    returns='void',
)
def gravity_pairwise(
    interaction_name, receiver, supplier, ᔑdt_rungs, rank_supplier, only_supply, pairing_level,
    tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
    extra_args,
):
    # Extract momentum update buffers
    Δmom_r = receiver.Δmom
    Δmom_s = supplier.Δmom
    # Extract jumped rung indices of the supplier
    # (the receiver is handled by particles_particles() below).
    rung_indices_jumped_s = supplier.rung_indices_jumped
    # Get common softening length
    softening = combine_softening_lengths(
        receiver.softening_length,
        supplier.softening_length,
    )
    # Get array of factors used for momentum updates;
    #   Δmom = -r⃗/r³*(G*mass_r*mass_s*Δt/a).
    # This array is indexed by the jumped rung index
    # of the receiver/supplier particle.
    factors = compute_factors(receiver, supplier, ᔑdt_rungs)
    # Loop over all (receiver, supplier) particle pairs (i, j)
    indexᵖ_j = -1
    for indexˣ_i, indexᵖ_j, indexˣ_j, rung_index_i, rung_index_s, x_ji, y_ji, z_ji, periodic_offset_x, periodic_offset_y, periodic_offset_z, apply_to_i, apply_to_j, factor_i, subtile_contain_jumping_s, particle_particle_t_begin, subtiling_r in particle_particle(
        receiver, supplier, pairing_level,
        tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
        rank_supplier, interaction_name, only_supply, factors,
    ):
        # Translate coordinates so that they
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
        # The Ewald correction force for all images except the
        # nearest one, which might not be the actual particle.
        force_ij = ewald(x_ji, y_ji, z_ji)
        # Add in the softened force from the particle's nearest image
        r2 = x_ji**2 + y_ji**2 + z_ji**2
        r3_inv_softened = get_softened_r3inv(r2, softening)
        forcex_ij = force_ij[0] - x_ji*r3_inv_softened
        forcey_ij = force_ij[1] - y_ji*r3_inv_softened
        forcez_ij = force_ij[2] - z_ji*r3_inv_softened
        # Momentum change of particle i due to particle j
        with unswitch(3):
            if apply_to_i:
                Δmomx = factor_i*forcex_ij
                Δmomy = factor_i*forcey_ij
                Δmomz = factor_i*forcez_ij
                Δmom_r[indexˣ_i + 0] += Δmomx
                Δmom_r[indexˣ_i + 1] += Δmomy
                Δmom_r[indexˣ_i + 2] += Δmomz
        # Momentum change of particle j due to particle i
        with unswitch(8):
            if 𝔹[not only_supply]:
                with unswitch(2):
                    if apply_to_j:
                        with unswitch(4):
                            if subtile_contain_jumping_s:
                                rung_index_j = rung_indices_jumped_s[indexᵖ_j]
                            else:
                                rung_index_j = rung_index_s
                        with unswitch(3):
                            if apply_to_i:
                                if rung_index_i == rung_index_j:
                                    Δmom_s[indexˣ_j + 0] -= Δmomx
                                    Δmom_s[indexˣ_j + 1] -= Δmomy
                                    Δmom_s[indexˣ_j + 2] -= Δmomz
                                    continue
                        factor_j = factors[rung_index_j]
                        Δmom_s[indexˣ_j + 0] -= factor_j*forcex_ij
                        Δmom_s[indexˣ_j + 1] -= factor_j*forcey_ij
                        Δmom_s[indexˣ_j + 2] -= factor_j*forcez_ij
    # Add computation time to the running total,
    # for use with automatic subtiling refinement.
    if indexᵖ_j != -1:
        particle_particle_t_final = time()
        subtiling_r.computation_time += particle_particle_t_final - particle_particle_t_begin

# Function implementing pairwise gravity (short-range only)
@cython.header(
    # Arguments
    interaction_name=str,
    receiver='Component',
    supplier='Component',
    ᔑdt_rungs=dict,
    rank_supplier='int',
    only_supply='bint',
    pairing_level=str,
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier_paired='Py_ssize_t**',
    tile_indices_supplier_paired_N='Py_ssize_t*',
    extra_args=dict,
    # Locals
    apply_to_i='bint',
    apply_to_j='bint',
    factor_i='double',
    factor_j='double',
    factors='const double*',
    forcex_ij='double',
    forcey_ij='double',
    forcez_ij='double',
    indexᵖ_j='Py_ssize_t',
    indexˣ_i='Py_ssize_t',
    indexˣ_j='Py_ssize_t',
    particle_particle_t_begin='double',
    particle_particle_t_final='double',
    periodic_offset_x='double',
    periodic_offset_y='double',
    periodic_offset_z='double',
    r2='double',
    r2_index_scaling='double',
    r2_max='double',
    rung_index_i='signed char',
    rung_index_j='signed char',
    rung_index_s='signed char',
    rung_indices_jumped_s='signed char*',
    shortrange_factor='double',
    shortrange_index='Py_ssize_t',
    softening='double',
    subtile_contain_jumping_s='bint',
    subtiling_r='Tiling',
    table='const double*',
    total_factor='double',
    x_ji='double',
    y_ji='double',
    z_ji='double',
    Δmom_r='double*',
    Δmom_s='double*',
    Δmomx='double',
    Δmomy='double',
    Δmomz='double',
    returns='void',
)
def gravity_pairwise_shortrange(
    interaction_name, receiver, supplier, ᔑdt_rungs, rank_supplier, only_supply, pairing_level,
    tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
    extra_args,
):
    # Extract momentum update buffers
    Δmom_r = receiver.Δmom
    Δmom_s = supplier.Δmom
    # Extract jumped rung indices of the supplier
    # (the receiver is handled by particles_particles() below).
    rung_indices_jumped_s = supplier.rung_indices_jumped
    # Get table of softened gravitational short-range forces
    softening = combine_softening_lengths(
        receiver.softening_length,
        supplier.softening_length,
    )
    table = get_shortrange_table(softening)
    # Get array of factors used for momentum updates;
    #   Δmom = -r⃗/r³*(G*mass_r*mass_s*Δt/a).
    # This array is indexed by the jumped rung index
    # of the receiver/supplier particle.
    factors = compute_factors(receiver, supplier, ᔑdt_rungs)
    # Maximum r² beyond which the interaction is ignored
    r2_max = ℝ[shortrange_range**2]
    # Factor used to scale r² to produce an index into the table
    r2_index_scaling = ℝ[(shortrange_table_size - 1)/shortrange_table_maxr2]
    # Loop over all (receiver, supplier) particle pairs (i, j)
    indexᵖ_j = -1
    for indexˣ_i, indexᵖ_j, indexˣ_j, rung_index_i, rung_index_s, x_ji, y_ji, z_ji, periodic_offset_x, periodic_offset_y, periodic_offset_z, apply_to_i, apply_to_j, factor_i, subtile_contain_jumping_s, particle_particle_t_begin, subtiling_r in particle_particle(
        receiver, supplier, pairing_level,
        tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
        rank_supplier, interaction_name, only_supply, factors, forcerange=shortrange_range,
    ):
        # Translate coordinates so that they
        # correspond to the nearest image.
        with unswitch(6):
            if periodic_offset_x or periodic_offset_y or periodic_offset_z:
                x_ji += periodic_offset_x
                y_ji += periodic_offset_y
                z_ji += periodic_offset_z
        # If the particle pair is separated by a distance larger
        # than the range of the short-range force,
        # ignore this interaction completely.
        r2 = x_ji**2 + y_ji**2 + z_ji**2
        if r2 > r2_max:
            continue
        # Compute the short-range force. Here the "force" is in units
        # of inverse length squared, given by
        #   force = -r⃗/r³ (x/sqrt(π) exp(-x²/4) + erfc(x/2)),
        # where x = r/scale with scale the long/short-range
        # force split scale.
        # We have this whole expression except for r⃗ already tabulated.
        # This tabulation has baked in softening of r⁻³.
        shortrange_index = int(r2*r2_index_scaling)
        shortrange_factor = table[shortrange_index]
        # Momentum change of particle i due to particle j
        with unswitch(3):
            if apply_to_i:
                total_factor = factor_i*shortrange_factor
                Δmomx = x_ji*total_factor
                Δmomy = y_ji*total_factor
                Δmomz = z_ji*total_factor
                Δmom_r[indexˣ_i + 0] += Δmomx
                Δmom_r[indexˣ_i + 1] += Δmomy
                Δmom_r[indexˣ_i + 2] += Δmomz
        # Momentum change of particle j due to particle i
        with unswitch(8):
            if 𝔹[not only_supply]:
                with unswitch(2):
                    if apply_to_j:
                        with unswitch(4):
                            if subtile_contain_jumping_s:
                                rung_index_j = rung_indices_jumped_s[indexᵖ_j]
                            else:
                                rung_index_j = rung_index_s
                        with unswitch(3):
                            if apply_to_i:
                                if rung_index_i == rung_index_j:
                                    Δmom_s[indexˣ_j + 0] -= Δmomx
                                    Δmom_s[indexˣ_j + 1] -= Δmomy
                                    Δmom_s[indexˣ_j + 2] -= Δmomz
                                    continue
                        factor_j = factors[rung_index_j]
                        total_factor = factor_j*shortrange_factor
                        Δmom_s[indexˣ_j + 0] -= x_ji*total_factor
                        Δmom_s[indexˣ_j + 1] -= y_ji*total_factor
                        Δmom_s[indexˣ_j + 2] -= z_ji*total_factor
    # Add computation time to the running total,
    # for use with automatic subtiling refinement.
    if indexᵖ_j != -1:
        particle_particle_t_final = time()
        subtiling_r.computation_time += particle_particle_t_final - particle_particle_t_begin

# Function that tabulates the gravitational short-range force,
# including softening.
@cython.header(
    # Arguments
    softening='double',
    # Locals
    i='Py_ssize_t',
    r='double',
    r2='double',
    r3_inv='double',
    r3_inv_softened='double',
    r_tabulation='double[::1]',
    table='double[::1]',
    table_ptr='double*',
    x='double',
    returns='const double*',
)
def get_shortrange_table(softening):
    # This function tabulates the short-range factor
    #   -r⁻³(x/sqrt(π)exp(-x²/4) + erfc(x/2)),
    # though in a softened version.
    # The tabulation is quadratic in r, which is the distance
    # between two particles, while x = r/scale with scale the
    # long/short-range force split scale.
    # Softening for small r is applied to the Newtonian part only,
    # so that the tabulation becomes
    #     -r⁻³(x/sqrt(π)exp(-x²/4) + erfc(x/2)) - (r⁻³_softened - r⁻³)
    #   = -r⁻³(x/sqrt(π)exp(-x²/4) + erfc(x/2) - 1) - r⁻³_softened
    # We only need the tabulation for 0 <= r <= range, where range
    # is the maximum reach of the short-range force.
    # All tables are cached.
    # Look up table in the cache
    table = shortrange_tables.get(softening)
    if table is not None:
        # Table found
        table_ptr = cython.address(table[:])
        return table_ptr
    # The distances at which the tabulation will be carried out,
    # quadratically spaced.
    r_tabulation = np.sqrt(
        linspace(
            0,
            shortrange_table_maxr2,
            shortrange_table_size,
        )
    )
    # Create the table. The i'th element of table really corresponds
    # to the value at r[i+½]. Nearest grid point lookups
    # can then be performed by cheap floor (int casting) indexing.
    table = empty(shortrange_table_size, dtype=C2np['double'])
    for i in range(shortrange_table_size - 1):
        r2 = 0.5*(r_tabulation[i]**2 + r_tabulation[i+1]**2)
        r = sqrt(r2)
        x = r*ℝ[1/shortrange_scale]
        r3_inv = 1/(r2*r)
        r3_inv_softened = get_softened_r3inv(r2, softening)
        table[i] = (
            - r3_inv*(1/sqrt(π)*x*exp(-ℝ[0.5*x]**2) + (erfc(ℝ[0.5*x]) - 1))
            - r3_inv_softened
        )
    # The last element in table is not populated above.
    # This element is guaranteed to never be accessed as it would
    # require an r > shortrange_range due to the way
    # shortrange_table_maxr2 is constructed. To demonstrate our
    # trust in this, we here assign it NaN.
    table[shortrange_table_size - 1] = NaN
    # Store in cache and return pointer by calling this function anew
    shortrange_tables[softening] = table
    return get_shortrange_table(softening)
# Global variables used by the get_shortrange_table()
# and gravity_pairwise_shortrange() functions.
cython.declare(
    shortrange_scale='double',
    shortrange_range='double',
    shortrange_table_size='Py_ssize_t',
    shortrange_table_maxr2='double',
    shortrange_tables=dict,
)
shortrange_scale      = shortrange_params['gravity']['scale'    ]
shortrange_range      = shortrange_params['gravity']['range'    ]
shortrange_table_size = shortrange_params['gravity']['tablesize']
shortrange_table_maxr2 = (1 + 1/shortrange_table_size)*shortrange_range**2
shortrange_tables = {}

# Function implementing pairwise gravity (non-periodic)
@cython.nounswitching
@cython.header(
    # Arguments
    interaction_name=str,
    receiver='Component',
    supplier='Component',
    ᔑdt_rungs=dict,
    rank_supplier='int',
    only_supply='bint',
    pairing_level=str,
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier_paired='Py_ssize_t**',
    tile_indices_supplier_paired_N='Py_ssize_t*',
    extra_args=dict,
    # Locals
    apply_to_i='bint',
    apply_to_j='bint',
    factor_i='double',
    factor_j='double',
    factors='const double*',
    forcex_ij='double',
    forcey_ij='double',
    forcez_ij='double',
    indexᵖ_j='Py_ssize_t',
    indexˣ_i='Py_ssize_t',
    indexˣ_j='Py_ssize_t',
    particle_particle_t_begin='double',
    particle_particle_t_final='double',
    periodic_offset_x='double',
    periodic_offset_y='double',
    periodic_offset_z='double',
    r2='double',
    r3_inv_softened='double',
    rung_index_i='signed char',
    rung_index_j='signed char',
    rung_index_s='signed char',
    rung_indices_jumped_s='signed char*',
    softening='double',
    subtile_contain_jumping_s='bint',
    subtiling_r='Tiling',
    x_ji='double',
    y_ji='double',
    z_ji='double',
    Δmom_r='double*',
    Δmom_s='double*',
    Δmomx='double',
    Δmomy='double',
    Δmomz='double',
    returns='void',
)
def gravity_pairwise_nonperiodic(
    interaction_name, receiver, supplier, ᔑdt_rungs, rank_supplier, only_supply, pairing_level,
    tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
    extra_args,
):
    # Extract momentum update buffers
    Δmom_r = receiver.Δmom
    Δmom_s = supplier.Δmom
    # Extract jumped rung indices of the supplier
    # (the receiver is handled by particles_particles() below).
    rung_indices_jumped_s = supplier.rung_indices_jumped
    # Get common softening length
    softening = combine_softening_lengths(
        receiver.softening_length,
        supplier.softening_length,
    )
    # Get array of factors used for momentum updates;
    #   Δmom = -r⃗/r³*(G*mass_r*mass_s*Δt/a).
    # This array is indexed by the jumped rung index
    # of the receiver/supplier particle.
    factors = compute_factors(receiver, supplier, ᔑdt_rungs)
    # Loop over all (receiver, supplier) particle pairs (i, j)
    indexᵖ_j = -1
    for indexˣ_i, indexᵖ_j, indexˣ_j, rung_index_i, rung_index_s, x_ji, y_ji, z_ji, periodic_offset_x, periodic_offset_y, periodic_offset_z, apply_to_i, apply_to_j, factor_i, subtile_contain_jumping_s, particle_particle_t_begin, subtiling_r in particle_particle(
        receiver, supplier, pairing_level,
        tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
        rank_supplier, interaction_name, only_supply, factors,
    ):
        # The direct, softened force on particle i from particle j
        r2 = x_ji**2 + y_ji**2 + z_ji**2
        r3_inv_softened = get_softened_r3inv(r2, softening)
        forcex_ij = x_ji*ℝ[-r3_inv_softened]
        forcey_ij = y_ji*ℝ[-r3_inv_softened]
        forcez_ij = z_ji*ℝ[-r3_inv_softened]
        # Momentum change of particle i due to particle j
        with unswitch(3):
            if apply_to_i:
                Δmomx = factor_i*forcex_ij
                Δmomy = factor_i*forcey_ij
                Δmomz = factor_i*forcez_ij
                Δmom_r[indexˣ_i + 0] += Δmomx
                Δmom_r[indexˣ_i + 1] += Δmomy
                Δmom_r[indexˣ_i + 2] += Δmomz
        # Momentum change of particle j due to particle i
        with unswitch(8):
            if 𝔹[not only_supply]:
                with unswitch(2):
                    if apply_to_j:
                        with unswitch(4):
                            if subtile_contain_jumping_s:
                                rung_index_j = rung_indices_jumped_s[indexᵖ_j]
                            else:
                                rung_index_j = rung_index_s
                        with unswitch(3):
                            if apply_to_i:
                                if rung_index_i == rung_index_j:
                                    Δmom_s[indexˣ_j + 0] -= Δmomx
                                    Δmom_s[indexˣ_j + 1] -= Δmomy
                                    Δmom_s[indexˣ_j + 2] -= Δmomz
                                    continue
                        factor_j = factors[rung_index_j]
                        Δmom_s[indexˣ_j + 0] -= factor_j*forcex_ij
                        Δmom_s[indexˣ_j + 1] -= factor_j*forcey_ij
                        Δmom_s[indexˣ_j + 2] -= factor_j*forcez_ij
    # Add computation time to the running total,
    # for use with automatic subtiling refinement.
    if indexᵖ_j != -1:
        particle_particle_t_final = time()
        subtiling_r.computation_time += particle_particle_t_final - particle_particle_t_begin
