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



# Function responsible for constructing pairings between subtiles within
# the supplied subtiling. Subtiles further away than the supplied cutoff
# will not be paired.
@cython.header(
    # Arguments
    subtiling='Tiling',
    cutoff='double',
    # Locals
    all_pairings='Py_ssize_t***',
    all_pairings_N='Py_ssize_t**',
    dim='int',
    pairing_index='Py_ssize_t',
    pairings='Py_ssize_t**',
    pairings_N='Py_ssize_t*',
    pairings_r='Py_ssize_t*',
    r_dim='Py_ssize_t',
    r2='double',
    same_tile='bint',
    subtile_index_r='Py_ssize_t',
    subtile_index_s='Py_ssize_t',
    subtile_index3D='Py_ssize_t[::1]',
    subtile_index3D_r='Py_ssize_t[::1]',
    subtile_index3D_s='Py_ssize_t[::1]',
    subtile_pairings_index='Py_ssize_t',
    tile_pair_index='Py_ssize_t',
    tiles_offset='Py_ssize_t[::1]',
    tiles_offset_i='Py_ssize_t',
    tiles_offset_j='Py_ssize_t',
    tiles_offset_k='Py_ssize_t',
    returns='Py_ssize_t',
)
def get_subtile_pairings(subtiling, cutoff):
    global subtile_pairings_cache, subtile_pairings_N_cache, subtile_pairings_cache_size
    # Get index of the required subtile pairings in the global cache.
    subtile_pairings_index = subtile_pairings_cache_indices.get(
        (subtiling, cutoff), subtile_pairings_cache_size,
    )
    # Return cached results in form of the cache index
    if subtile_pairings_index < subtile_pairings_cache_size:
        return subtile_pairings_index
    # No cached results found.
    # Create subtile pairings for each of the 27 cases
    # of neighbour tiles.
    all_pairings   = malloc(27*sizeof('Py_ssize_t**'))
    all_pairings_N = malloc(27*sizeof('Py_ssize_t*'))
    tiles_offset      = empty(3, dtype=C2np['Py_ssize_t'])
    subtile_index3D_r = empty(3, dtype=C2np['Py_ssize_t'])
    for tiles_offset_i in range(-1, 2):
        tiles_offset[0] = tiles_offset_i
        for tiles_offset_j in range(-1, 2):
            tiles_offset[1] = tiles_offset_j
            for tiles_offset_k in range(-1, 2):
                tiles_offset[2] = tiles_offset_k
                # Does the tile offset correspond to
                # a tile being paired with itself?
                same_tile = (tiles_offset_i == tiles_offset_j == tiles_offset_k == 0)
                # Get 1D tile pair index from the 3D offset
                tile_pair_index = get_neighbourtile_pair_index(tiles_offset)
                # Allocate memory for subtile pairings
                # for this particular tile pair.
                pairings   = malloc(subtiling.size*sizeof('Py_ssize_t*'))
                pairings_N = malloc(subtiling.size*sizeof('Py_ssize_t'))
                all_pairings  [tile_pair_index] = pairings
                all_pairings_N[tile_pair_index] = pairings_N
                # Loop over all receiver subtiles
                for subtile_index_r in range(subtiling.size):
                    # Get 3D subtile index. As the tile_index3D() method
                    # return a view over internal data and we mutate
                    # subtile_index3D_r below, we take a copy of the
                    # returned data.
                    subtile_index3D = subtiling.tile_index3D(subtile_index_r)
                    for dim in range(3):
                        subtile_index3D_r[dim] = subtile_index3D[dim]
                    # The receiver and supplier subtiles belong to
                    # (potentially) diffent tiles, with a relative
                    # offset given by tiles_offset_*, so that the
                    # supplier tile is at the receiver tile location
                    # plus tiles_offset_*. We now subtract this offset
                    # from the receiver 3D subtile index, so that the
                    # difference in subtile indices between the receiver
                    # and supplier subtile is proportional to their
                    # physical separation. Note that subtile_index3D_r
                    # no longer represents the actual index in memory.
                    for dim in range(3):
                        subtile_index3D_r[dim] -= tiles_offset[dim]*subtiling.shape[dim]
                    # Allocate memory for subtile pairings with this
                    # particular receiver subilte.
                    # We give it the maximum possible needed memory.
                    pairings_r = malloc(subtiling.size*sizeof('Py_ssize_t'))
                    pairings[subtile_index_r] = pairings_r
                    # Pair receiver subtile with every supplier subtile,
                    # unless the tile is being paired with itself.
                    # In that case, we need to not double count the
                    # subtile pairing (while still pairing every subtile
                    # with themselves).
                    pairing_index = 0
                    for subtile_index_s in range(
                        subtile_index_r if same_tile else 0,
                        subtiling.size,
                    ):
                        subtile_index3D_s = subtiling.tile_index3D(subtile_index_s)
                        # Measure (squared) distance between the subtile
                        # pair and reject if larger than the passed
                        # cutoff length.
                        r2 = 0
                        for dim in range(3):
                            # Distance between the same point in the two
                            # subtiles along the dim'th dimenson,
                            # in subtile grid units.
                            r_dim = abs(subtile_index3D_r[dim] - subtile_index3D_s[dim])
                            if r_dim > 0:
                                # The two subtiles are offset along the
                                # dim'th dimension. Subtract one unit
                                # from the length, making the length
                                # between the closest two points
                                # in the two subtiles.
                                r_dim -= 1
                            r2 += (r_dim*subtiling.tile_extent[dim])**2
                        if r2 > ℝ[cutoff**2]:
                            continue
                        # Add this supplier subtile to the list of
                        # pairing partners for this receiver subtile.
                        pairings_r[pairing_index] = subtile_index_s
                        pairing_index += 1
                    # All pairs found for this receiver subtile.
                    # Truncate the allocated memory as to only contain
                    # the used chunk.
                    pairings[subtile_index_r] = realloc(
                        pairings_r, pairing_index*sizeof('Py_ssize_t'),
                    )
                    # Record the size of this pairing array
                    pairings_N[subtile_index_r] = pairing_index
    # Store results in global caches
    subtile_pairings_cache_size += 1
    subtile_pairings_cache = realloc(
        subtile_pairings_cache, subtile_pairings_cache_size*sizeof('Py_ssize_t***'),
    )
    subtile_pairings_N_cache = realloc(
        subtile_pairings_N_cache, subtile_pairings_cache_size*sizeof('Py_ssize_t**'),
    )
    subtile_pairings_cache  [subtile_pairings_index] = all_pairings
    subtile_pairings_N_cache[subtile_pairings_index] = all_pairings_N
    subtile_pairings_cache_indices[subtiling, cutoff] = subtile_pairings_index
    # Return cached results in form of the cache index
    return subtile_pairings_index
# Caches used by the get_subtile_pairings function
cython.declare(
    subtile_pairings_cache_size='Py_ssize_t',
    subtile_pairings_cache_indices=dict,
    subtile_pairings_cache='Py_ssize_t****',
    subtile_pairings_N_cache='Py_ssize_t***',
)
subtile_pairings_cache_size = 0
subtile_pairings_cache_indices = {}
subtile_pairings_cache   = malloc(subtile_pairings_cache_size*sizeof('Py_ssize_t***'))
subtile_pairings_N_cache = malloc(subtile_pairings_cache_size*sizeof('Py_ssize_t**'))

# Helper function for the get_subtile_pairings function
@cython.header(
    # Arguments
    tiles_offset='Py_ssize_t[::1]',
    # Locals
    dim='int',
    returns='Py_ssize_t',
)
def get_neighbourtile_pair_index(tiles_offset):
    # The passed tiles_offset is the relative offset between a pair of
    # neighbouring tiles, and so each of its three elements has to be
    # in {-1, 0, +1}. If any element is outside this range, it is due
    # to the periodic boundaries. Fix this now, as we do not care about
    # whether the tile pair is connected through the box boundary.
    for dim in range(3):
        if tiles_offset[dim] > 1:
            tiles_offset[dim] = -1
        elif tiles_offset[dim] < -1:
            tiles_offset[dim] = +1
    # Compute 1D index from a 3×3×3 shape. We add 1 to each element,
    # as they range from -1 to +1.
    return ((tiles_offset[0] + 1)*3 + (tiles_offset[1] + 1))*3 + (tiles_offset[2] + 1)

# Generic function implementing particle-particle pairing.
# Note that this function returns a generator and so should only be
# called within a loop. See gravity_pairwise() for an example.
@cython.iterator
def particle_particle(
    receiver, supplier, tile_indices_receiver, tile_indices_supplier,
    rank_supplier, interaction_name,
):
    # Cython declarations for variables used for the iteration,
    # not including those to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        N_r='Py_ssize_t',
        N_s='Py_ssize_t',
        N_subtiles='Py_ssize_t',
        all_subtile__pairings='Py_ssize_t***',
        all_subtile__pairings_N='Py_ssize_t**',
        dim='int',
        i_end='Py_ssize_t',
        initial_tile_size='Py_ssize_t',
        j_end='Py_ssize_t',
        j_start='Py_ssize_t',
        pairing_level=str,
        posx_r='double*',
        posx_s='double*',
        posy_r='double*',
        posy_s='double*',
        posz_r='double*',
        posz_s='double*',
        subtile_index_r='Py_ssize_t',
        subtile_index_s='Py_ssize_t',
        subtile_pairings='Py_ssize_t**',
        subtile_pairings_r='Py_ssize_t*',
        subtile_pairings_N='Py_ssize_t*',
        subtile_pairings_N_r='Py_ssize_t',
        subtile_pairings_index='Py_ssize_t',
        subtile_r='Py_ssize_t*',
        subtile_s='Py_ssize_t*',
        subtiles_N_r='Py_ssize_t*',
        subtiles_N_s='Py_ssize_t*',
        subtiles_r='Py_ssize_t**',
        subtiles_s='Py_ssize_t**',
        subtiling_name=str,
        subtiling_name_2=str,
        subtiling_r='Tiling',
        subtiling_s='Tiling',
        subtiling_s_2='Tiling',
        tile_extent='double[::1]',
        tile_index_r='Py_ssize_t',
        tile_index_s='Py_ssize_t',
        tile_index3D_r='Py_ssize_t[::1]',
        tile_index3D_s='Py_ssize_t[::1]',
        tile_pair_index='Py_ssize_t',
        tile_r='Py_ssize_t*',
        tile_s='Py_ssize_t*',
        tiles_N_r='Py_ssize_t*',
        tiles_N_s='Py_ssize_t*',
        tiles_r='Py_ssize_t**',
        tiles_s='Py_ssize_t**',
        tiling_location_r='double[::1]',
        tiling_location_s='double[::1]',
        tiling_name=str,
        tiling_r='Tiling',
        tiling_s='Tiling',
        xi='double',
        yi='double',
        zi='double',
    )
    # Extract particle variables from the receiver component
    N_r = receiver.N_local
    posx_r = receiver.posx
    posy_r = receiver.posy
    posz_r = receiver.posz
    # Extract particle variables from the supplier
    # (the external) component.
    N_s = supplier.N_local
    posx_s = supplier.posx
    posy_s = supplier.posy
    posz_s = supplier.posz
    # Infer pairing level ('domain' or 'tile') from passed tile indices
    if tile_indices_receiver is not None:
        pairing_level = 'tile'
        # Extract tiling variables from receiver
        tiling_name       = f'{interaction_name} (tiles)'
        subtiling_name    = f'{interaction_name} (subtiles)'
        tiling_r          = receiver.tilings[tiling_name]
        tiles_r           = tiling_r.tiles
        tiles_N_r         = tiling_r.tiles_N
        tiling_location_r = tiling_r.location
        tile_extent       = tiling_r.tile_extent  # The same for receiver and supplier
        subtiling_r       = receiver.tilings[subtiling_name]
        subtiles_r        = subtiling_r.tiles
        subtiles_N_r      = subtiling_r.tiles_N
        N_subtiles        = subtiling_r.size  # The same for receiver and supplier
        # Extract tiling variables from supplier
        tiling_s          = supplier.tilings[tiling_name]
        tiles_s           = tiling_s.tiles
        tiles_N_s         = tiling_s.tiles_N
        tiling_location_s = tiling_s.location
        subtiling_s       = supplier.tilings[subtiling_name]
        subtiles_s        = subtiling_s.tiles
        subtiles_N_s      = subtiling_s.tiles_N
        # When the receiver and supplier components are the same
        # and the receiver and supplier domains are also the same,
        # we now have a case where (tiling_r is tiling_s) and
        # (subtiling_r is subtiling_s) are both True. This is OK for
        # the coarse tiling, but not for the subtiling, as here we need
        # to re-sort the particles during the iteration below. That is,
        # we need to keep track of the sorting of the receiver tiles
        # into subtiles while also keeping track of the sorting of the
        # supplier tiles into subtiles. We thus always need two separate
        # subtiling_{r/s} instances, which we do not have in the case
        # mentioned. When this is the case, we make use of a second,
        # separate instance.
        if 𝔹[receiver.name == supplier.name and rank == rank_supplier]:
            subtiling_name_2 = f'{interaction_name} (subtiles 2)'
            if subtiling_name_2 not in supplier.tilings:
                initial_tile_size = supplier.N_local//(2*int(np.prod(tiling_s.shape)))
                subtiling_s_2 = supplier.init_tiling(subtiling_name, initial_tile_size)
                supplier.tilings[subtiling_name  ] = subtiling_s
                supplier.tilings[subtiling_name_2] = subtiling_s_2
            subtiling_s = supplier.tilings[subtiling_name_2]
            subtiles_s = subtiling_s.tiles
            subtiles_N_s = subtiling_s.tiles_N
        # Get subtile pairings between each
        # of the 27 possible tile pairings.
        subtile_pairings_index = get_subtile_pairings(
            subtiling_r, shortrange_params[interaction_name]['cutoff'],
        )
        all_subtile_pairings   = subtile_pairings_cache  [subtile_pairings_index]
        all_subtile_pairings_N = subtile_pairings_N_cache[subtile_pairings_index]
    else:
        pairing_level = 'domain'
        # This is just to create one-iteration loops.
        # The tiling will not be used.
        tile_indices_receiver = tile_indices_supplier = empty(1, dtype=C2np['Py_ssize_t'])
        N_subtiles = 1
        # These are not used, but Cython and the compiler
        # likes to have them defined.
        tiling_r = tiling_s = subtiling_r = subtiling_s = None
        tile_r = tile_s = subtile_r = subtile_s = NULL
    # Loop over the requested tiles in the receiver
    for tile_index_r in range(ℤ[tile_indices_receiver.shape[0]]):
        tile_index_r = tile_indices_receiver[tile_index_r]
        # Sort particles within the receiver tile into subtiles
        with unswitch:
            if 𝔹[pairing_level == 'tile']:
                tile_index3D_r = tiling_r.tile_index3D(tile_index_r)
                for dim in range(3):
                    tile_location_r[dim] = (
                        tiling_location_r[dim] + tile_index3D_r[dim]*tile_extent[dim]
                    )
                subtiling_r.relocate(tile_location_r)
                subtiling_r.sort(tiling_r, tile_index_r)
        # Loop over the requested tiles in the supplier
        for tile_index_s in range(ℤ[tile_indices_supplier.shape[0]]):
            tile_index_s = tile_indices_supplier[tile_index_s]
            # Sort particles within the supplier tile into subtiles
            with unswitch:
                if 𝔹[pairing_level == 'tile']:
                    tile_index3D_s = tiling_s.tile_index3D(tile_index_s)
                    for dim in range(3):
                        tile_location_s[dim] = (
                            tiling_location_s[dim] + tile_index3D_s[dim]*tile_extent[dim]
                        )
                    subtiling_s.relocate(tile_location_s)
                    subtiling_s.sort(tiling_s, tile_index_s)
            # Get the needed subtile pairings for the selected receiver
            # and supplier tiles (which should be neighbour tiles).
            with unswitch:
                if 𝔹[pairing_level == 'tile']:
                    for dim in range(3):
                        tiles_offset[dim] = tile_index3D_s[dim] - tile_index3D_r[dim]
                    tile_pair_index = get_neighbourtile_pair_index(tiles_offset)
                    subtile_pairings   = all_subtile_pairings  [tile_pair_index]
                    subtile_pairings_N = all_subtile_pairings_N[tile_pair_index]
            # Loop over all subtiles in the selected receiver tile
            for subtile_index_r in range(N_subtiles):
                # Prepare for the loop over (some of the) supplier tiles
                # and the loop over all particles in the selected
                # receiver subtile.
                with unswitch:
                    if 𝔹[pairing_level == 'tile']:
                        subtile_pairings_r   = subtile_pairings  [subtile_index_r]
                        subtile_pairings_N_r = subtile_pairings_N[subtile_index_r]
                        i_end = subtiles_N_r[subtile_index_r]
                        subtile_r = subtiles_r[subtile_index_r]
                    else:  # pairing_level == 'domain'
                        i_end = N_r
                        subtile_pairings_N_r = 1
                # Loop over the needed supplier tiles
                for subtile_index_s in range(subtile_pairings_N_r):
                    # Prepare for the loop over all particles
                    # in the selected supplier subtile.
                    with unswitch:
                        if 𝔹[pairing_level == 'tile']:
                            subtile_index_s = subtile_pairings_r[subtile_index_s]
                            subtile_s = subtiles_s[subtile_index_s]
                            j_end = subtiles_N_s[subtile_index_s]
                        else:  # pairing_level == 'domain'
                            j_end = N_s
                    # Loop over all receiver particles in the domain
                    # (pairing_level == 'domain') or just this subtile
                    # (pairing_level == 'tile').
                    for i in range(i_end):
                        # If the receiver and supplier component are one
                        # and the same and the two paired domains are
                        # one and the same, we need to make sure not to
                        # double count the particles.
                        with unswitch:
                            if 𝔹[receiver.name == supplier.name and rank == rank_supplier]:
                                with unswitch:
                                    if 𝔹[pairing_level == 'tile']:
                                        # When using (sub)tiles, double
                                        # counting particles is only
                                        # possible when the current
                                        # receiver and supplier tile are
                                        # one and the same and the
                                        # current receiver and supplier
                                        # subtile are also one
                                        # and the same.
                                        if True:  #with unswitch(3):
                                            if 𝔹[tile_index_r == tile_index_s]:
                                                if True: #with unswitch(1):
                                                    if 𝔹[subtile_index_r == subtile_index_s]:
                                                        j_start = i + 1
                                                    else:
                                                        j_start = 0
                                            else:
                                                j_start = 0
                                    else:  # pairing_level == 'domain'
                                        j_start = i + 1
                            else:
                                j_start = 0
                        # Change i from being a subtile-particle index
                        # to being a particle index.
                        with unswitch:
                            if 𝔹[pairing_level == 'tile']:
                                i = subtile_r[i]
                        # Coordinates of receiver particle
                        xi = posx_r[i]
                        yi = posy_r[i]
                        zi = posz_r[i]
                        # Loop over all supplier particles in the domain
                        # (pairing_level == 'domain') or just
                        # this subtile (pairing_level == 'tile').
                        for j in range(j_start, j_end):
                            # Change j from being a subtile-particle
                            # index to being a particle index.
                            with unswitch:
                                if 𝔹[pairing_level == 'tile']:
                                    j = subtile_s[j]
                            # "Vector" from particle j to particle i
                            x_ji = xi - posx_s[j]
                            y_ji = yi - posy_s[j]
                            z_ji = zi - posz_s[j]
                            # Yield the needed variables
                            yield i, j, x_ji, y_ji, z_ji
# Variables used by the particle_particle function
cython.declare(
    tile_location_r='double[::1]',
    tile_location_s='double[::1]',
    tiles_offset='Py_ssize_t[::1]',
)
tile_location_r = empty(3, dtype=C2np['double'])
tile_location_s = empty(3, dtype=C2np['double'])
tiles_offset  = empty(3, dtype=C2np['Py_ssize_t'])

# Function implementing pairwise gravity (full/periodic)
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
    force_ij='double*',
    forcex_ij='double',
    forcey_ij='double',
    forcez_ij='double',
    i='Py_ssize_t',
    interaction_name=str,
    j='Py_ssize_t',
    mass_r='double',
    mass_s='double',
    momx_r='double*',
    momx_s='double*',
    momy_r='double*',
    momy_s='double*',
    momz_r='double*',
    momz_s='double*',
    r3='double',
    softening_r='double',
    softening_s='double',
    x_ji='double',
    y_ji='double',
    z_ji='double',
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
    # Extract particle variables from the receiver component
    mass_r = receiver.mass
    softening_r = receiver.softening_length
    momx_r = receiver.momx
    momy_r = receiver.momy
    momz_r = receiver.momz
    # Extract particle variables from the supplier
    # (the external) component.
    mass_s = supplier.mass
    softening_s = supplier.softening_length
    momx_s = supplier.momx
    momy_s = supplier.momy
    momz_s = supplier.momz
    Δmomx_s = supplier.Δmomx
    Δmomy_s = supplier.Δmomy
    Δmomz_s = supplier.Δmomz
    # Loop over all (receiver, supplier) particle pairs (i, j)
    interaction_name = 'gravity'
    for i, j, x_ji, y_ji, z_ji in particle_particle(
        receiver, supplier, tile_indices_receiver, tile_indices_supplier,
        rank_supplier, interaction_name,
    ):
        # Translate coordinates so they correspond to the nearest image
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
        # Add in the force from the particle's nearest image
        r3 = (x_ji**2 + y_ji**2 + z_ji**2 + ℝ[(0.5*(softening_r + softening_s))**2])**1.5
        forcex_ij = force_ij[0] - x_ji*ℝ[1/r3]
        forcey_ij = force_ij[1] - y_ji*ℝ[1/r3]
        forcez_ij = force_ij[2] - z_ji*ℝ[1/r3]
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

# Function implementing pairwise gravity (short-range only)
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
    forcex_ij='double',
    forcey_ij='double',
    forcez_ij='double',
    i='Py_ssize_t',
    interaction_name=str,
    j='Py_ssize_t',
    mass_r='double',
    mass_s='double',
    momx_r='double*',
    momx_s='double*',
    momy_r='double*',
    momy_s='double*',
    momz_r='double*',
    momz_s='double*',
    r2='double',
    shortrange_factor='double',
    shortrange_index='Py_ssize_t',
    softening_r='double',
    softening_s='double',
    x_ji='double',
    y_ji='double',
    z_ji='double',
    Δmomx_s='double*',
    Δmomx_ij='double',
    Δmomy_s='double*',
    Δmomy_ij='double',
    Δmomz_s='double*',
    Δmomz_ij='double',
    returns='void',
)
def gravity_pairwise_shortrange(
    receiver, supplier, tile_indices_receiver, tile_indices_supplier,
    rank_supplier, only_supply, ᔑdt, extra_args,
):
    # Extract particle variables from the receiver component
    mass_r = receiver.mass
    softening_r = receiver.softening_length
    momx_r = receiver.momx
    momy_r = receiver.momy
    momz_r = receiver.momz
    # Extract particle variables from the supplier
    # (the external) component.
    mass_s = supplier.mass
    softening_s = supplier.softening_length
    momx_s = supplier.momx
    momy_s = supplier.momy
    momz_s = supplier.momz
    Δmomx_s = supplier.Δmomx
    Δmomy_s = supplier.Δmomy
    Δmomz_s = supplier.Δmomz
    # Loop over all (receiver, supplier) particle pairs (i, j)
    interaction_name = 'gravity'
    for i, j, x_ji, y_ji, z_ji in particle_particle(
        receiver, supplier, tile_indices_receiver, tile_indices_supplier,
        rank_supplier, interaction_name,
    ):
        # Translate coordinates so they correspond to the nearest image
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
        # If the particle pair is separated by a distance larger
        # than the range of the short-range force,
        # ignore this interaction completely.
        if r2 > ℝ[shortrange_params['gravity']['cutoff']**2]:
            continue
        # Add softening
        r2 += ℝ[(0.5*(softening_r + softening_s))**2]
        # Compute the short-range force. Here the "force" is in units
        # of inverse length squared, given by
        # force = -r⃗/r³ (x/sqrt(π) exp(-x²/4) + erfc(x/2)),
        # where x = r/scale with scale the long/short-range
        # force split scale.
        # We have this whole expression except for r⃗ already tabulated.
        shortrange_index = int(r2*(shortrange_table.shape[0] - 1)/shortrange_table_maxr2)
        shortrange_factor = shortrange_table_ptr[shortrange_index]
        forcex_ij = x_ji*shortrange_factor
        forcey_ij = y_ji*shortrange_factor
        forcez_ij = z_ji*shortrange_factor
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
# Helper function for the gravity_pairwise_shortrange function,
# which initializes the global shortrange table.
@cython.header(returns='void')
def tabulate_shortrange_gravity():
    """This function tabulates the short-range factor
    y = -r⁻³(x/sqrt(π)exp(-x²/4) + erfc(x/2))
    with x = r/scale where r is the distance between two particles
    and scale is the long/short-range force split scale.
    We only need this for 0 <= r <= cutoff, where cutoff is the maximum
    reach of the short-range force. However, due to softening we
    sometimes need to go a bit beyound cutoff. Just to be safe we
    tabulate out to 2*cutoff.
    """
    global shortrange_table, shortrange_table_ptr, shortrange_table_maxr2
    if shortrange_table is not None:
        return
    scale  = shortrange_params['gravity']['scale']
    cutoff = shortrange_params['gravity']['cutoff']
    size = 2**20
    r2 = np.linspace(0, (2*cutoff)**2, size)
    r = np.sqrt(r2)
    x = r/scale
    # Compute r⁻³. The zeroth element is r == 0.
    # Here we explicitly set r⁻³ to 0, corresponding to zero force in
    # the case of two particles sitting on top of each other.
    one_over_r3 = r.copy()
    one_over_r3[0] = 1
    one_over_r3 **= -3
    one_over_r3[0] = 0
    # Do the tabulation and populate global variables
    shortrange_table = -one_over_r3*(x/np.sqrt(π)*np.exp(-0.25*x**2) + scipy.special.erfc(0.5*x))
    shortrange_table_ptr = cython.address(shortrange_table[:])
    shortrange_table_maxr2 = r2[size - 1]
# Global variables set by the tabulate_shortrange_gravity function
cython.declare(
    shortrange_table='double[::1]',
    shortrange_table_ptr='double*',
    shortrange_table_maxr2='double',
)
shortrange_table = None
shortrange_table_ptr = NULL
shortrange_table_maxr2 = -1

# Function implementing pairwise gravity (non-periodic)
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
    forcex_ij='double',
    forcey_ij='double',
    forcez_ij='double',
    i='Py_ssize_t',
    interaction_name=str,
    j='Py_ssize_t',
    mass_r='double',
    mass_s='double',
    momx_r='double*',
    momx_s='double*',
    momy_r='double*',
    momy_s='double*',
    momz_r='double*',
    momz_s='double*',
    r3='double',
    softening_r='double',
    softening_s='double',
    x_ji='double',
    y_ji='double',
    z_ji='double',
    Δmomx_s='double*',
    Δmomx_ij='double',
    Δmomy_s='double*',
    Δmomy_ij='double',
    Δmomz_s='double*',
    Δmomz_ij='double',
    returns='void',
)
def gravity_pairwise_nonperiodic(
    receiver, supplier, tile_indices_receiver, tile_indices_supplier,
    rank_supplier, only_supply, ᔑdt, extra_args,
):
    # Extract particle variables from the receiver component
    mass_r = receiver.mass
    softening_r = receiver.softening_length
    momx_r = receiver.momx
    momy_r = receiver.momy
    momz_r = receiver.momz
    # Extract particle variables from the supplier
    # (the external) component.
    mass_s = supplier.mass
    softening_s = supplier.softening_length
    momx_s = supplier.momx
    momy_s = supplier.momy
    momz_s = supplier.momz
    Δmomx_s = supplier.Δmomx
    Δmomy_s = supplier.Δmomy
    Δmomz_s = supplier.Δmomz
    # Loop over all (receiver, supplier) particle pairs (i, j)
    interaction_name = 'gravity'
    for i, j, x_ji, y_ji, z_ji in particle_particle(
        receiver, supplier, tile_indices_receiver, tile_indices_supplier,
        rank_supplier, interaction_name,
    ):
        # The direct force on particle i from particle j
        r3 = (x_ji**2 + y_ji**2 + z_ji**2 + ℝ[(0.5*(softening_r + softening_s))**2])**1.5
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
    return exp(-k2*ℝ[shortrange_params['gravity']['scale']**2])*gravity_potential(k2)

