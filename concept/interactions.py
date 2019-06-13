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
cimport('from communication import '
    'communicate_domain, sendrecv_component, rank_neighbouring_domain, domain_subdivisions, '
)
cimport('from communication import domain_size_x , domain_size_y , domain_size_z' )
cimport('from communication import domain_start_x, domain_start_y, domain_start_z')
cimport('from mesh import diff_domain, domain_decompose, fft, slab_decompose')
cimport('from mesh import CIC_components2φ, CIC_grid2grid, CIC_scalargrid2coordinates')
# Import interactions defined in other modules
cimport('from gravity import *')

# Function pointer types used in this module
pxd("""
ctypedef void (*func_interaction)(
    Component,        # receiver
    Component,        # supplier
    str,              # pairing_level
    Py_ssize_t[::1],  # tile_indices_receiver
    Py_ssize_t[::1],  # tile_indices_supplier
    int,              # rank_supplier
    bint,             # only_supply
    dict,             # ᔑdt
    dict,             # interaction_extra_args
)
ctypedef double (*func_potential)(
    double,  # k2
)
""")



# Generic function implementing component-component pairing
@cython.header(
    # Arguments
    receivers=list,
    suppliers=list,
    interaction=func_interaction,
    ᔑdt=dict,
    dependent=list,
    affected=list,
    deterministic='bint',
    pairing_level=str,
    interaction_name=str,
    interaction_extra_args=dict,
    # Locals
    component_pair=set,
    pairings=list,
    receiver='Component',
    subtiling_name=str,
    supplier='Component',
    tile_sorted=set,
    tiling_name=str,
    returns='void',
)
def component_component(
    receivers, suppliers, interaction, ᔑdt, dependent, affected,
    deterministic, pairing_level, interaction_name, interaction_extra_args={},
):
    """This function takes care of pairings between all receiver and
    supplier components. It then calls doman_domain.
    """
    # The names used to refer to the domain and tile level tiling
    # (tiles and subtiles). In the case of pairing_level == 'domain',
    # no actual tiling will take place, but we still need the
    # tile + subtile structure. For this, the trivial tiling,
    # spanning the box, is used.
    if pairing_level == 'tile':
        tiling_name    = f'{interaction_name} (tiles)'
        subtiling_name = f'{interaction_name} (subtiles)'
    else:  # pairing_level == 'domain':
        tiling_name = subtiling_name = 'trivial'
    # Pair each receiver with all suppliers
    pairings = []
    tile_sorted = set()
    for receiver in receivers:
        for supplier in suppliers:
            component_pair = {receiver, supplier}
            if component_pair in pairings:
                continue
            pairings.append(component_pair)
            # Make sure that the tile sorting of particles
            # in the two components are up-to-date.
            with unswitch(1):
                if receiver not in tile_sorted:
                    receiver.tile_sort(tiling_name)
                    tile_sorted.add(receiver)
                    # Also ensure existence of subtiling
                    receiver.init_tiling(subtiling_name)
            if supplier not in tile_sorted:
                supplier.tile_sort(tiling_name)
                tile_sorted.add(supplier)
                # Also ensure existence of subtiling
                supplier.init_tiling(subtiling_name)
            # Flag specifying whether the supplier should only supply
            # forces to the receiver and not receive any force itself.
            only_supply = (supplier not in receivers)
            # Pair up doamins for the current
            # receiver and supplier component.
            domain_domain(
                receiver, supplier, interaction, ᔑdt, dependent, affected, only_supply,
                deterministic, pairing_level, interaction_name, interaction_extra_args,
            )

# Generic function implementing domain-domain pairing
@cython.header(
    # Arguments
    receiver='Component',
    supplier='Component',
    interaction=func_interaction,
    ᔑdt=dict,
    dependent=list,
    affected=list,
    only_supply='bint',
    deterministic='bint',
    pairing_level=str,
    interaction_name=str,
    interaction_extra_args=dict,
    # Locals
    domain_pair_nr='Py_ssize_t',
    interact='bint',
    only_supply_passed='bint',
    rank_recv='int',
    rank_send='int',
    ranks_recv='int[::1]',
    ranks_send='int[::1]',
    supplier_extrl='Component',
    supplier_local='Component',
    tile_indices='Py_ssize_t[:, ::1]',
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier='Py_ssize_t[::1]',
    returns='void',
)
def domain_domain(
    receiver, supplier, interaction, ᔑdt, dependent, affected,
    only_supply, deterministic, pairing_level, interaction_name, interaction_extra_args,
):
    """This function takes care of pairings between the domains
    containing particles/fluid elements of the passed receiver and
    supplier component.
    As the components are distributed at the domain level,
    all communication needed for the interaction will be taken care of
    by this function. The receiver will not be communicated, while the
    supplier will be sent to other processes (domains) and also received
    back from other processes. Thus both local and external versions of
    the supplier exist, called supplier_local and supplier_extrl.
    The dependent and affected arguments specify which attributes of the
    supplier and receiver component are needed to supply and receive
    the force, respectively. Only these attributes will be communicated.
    If affected is an empty list, this is not really an interaction.
    In this case, every domain will both send and receive from every
    other domain.
    If pairing_level == 'domain', the passed interaction will be called
    directly by this function. If pairing_level == 'tile', the tile_tile
    function will be called instead.
    """
    # Just to satisfy the compiler
    tile_indices_receiver = tile_indices_supplier = None
    # Get the process ranks to send to and receive from.
    # When only_supply is True, each domain will be paired with every
    # other domain, either in the entire box (pairing_level == 'domain')
    # or just among the neighbouring domains (pairing_level == 'tile').
    # When only_supply is False, the results of an interaction
    # computed on one process will be send back to the other
    # participating process and applied, cutting the number of domain
    # pairs roughly in half.
    ranks_send, ranks_recv = domain_domain_communication(pairing_level, only_supply)
    # Backup of the passed only_supply boolean
    only_supply_passed = only_supply
    # Pair this process/domain with whichever other
    # processes/domains are needed. This process is paired
    # with two other processes simultaneously. This process/rank sends
    # a copy of the local supplier (from now on referred to
    # as supplier_local) to rank_send, while receiving the external
    # supplier (supplier_extrl) from rank_recv.
    # On each process, the local receiver and the external
    # (received) supplier_extrl then interact.
    supplier_local = supplier
    for domain_pair_nr in range(ranks_send.shape[0]):
        # Process ranks to send to and receive from
        rank_send = ranks_send[domain_pair_nr]
        rank_recv = ranks_recv[domain_pair_nr]
        # The passed interaction function should always update the
        # particles of the receiver component within the local domain,
        # due to the particles of the external supplier component,
        # within whatever domain they happen to be in.
        # Unless the supplier component is truly only a supplier and
        # not also a receiver (only_supply is True), the particles
        # that supply the force also need to be updated by the passed
        # interaction function. If the paired external domain is really
        # the local domain, this update should be done directly. On the
        # other hand, if the external domain is different from the
        # local domain, we have no direct access to the memory of these
        # particles. Instead, the interaction function should store the
        # changes to the affected variables in the corresponding buffer
        # variables (Δmom for gravity). All of this should be figured
        # out by the interaction function on the basis of the passed
        # receiver, supplier, rank_supplier and only_supply.
        # Special cases described below may change whether or not the
        # interaction between this particular domain pair should be
        # carried out on the local process (specified by the
        # interact flag), or whether the only_supply
        # flag should be changed.
        interact = True
        only_supply = only_supply_passed
        with unswitch:
            if 𝔹[pairing_level == 'domain' and not only_supply_passed]:
                if rank_send == rank_recv != rank:
                    # We are dealing with the special case where the
                    # local process and some other (with a rank given by
                    # rank_send == rank_recv) both send all of their
                    # particles to each other, after which the exact
                    # same interaction takes place on both processes.
                    # In such a case, even when only_supply is False,
                    # there is no need to communicate the interaction
                    # results, as these are already known to both
                    # processes. Thus, we always pass in only_supply as
                    # being True in such cases.
                    only_supply = True
                    # In the case of a non-deterministic interaction,
                    # the above logic no longer holds, as the two
                    # versions of the supposedly same interaction
                    # computed on different processes will not be
                    # identical. In such cases, perform the interaction
                    # only on one of the two processes. The process with
                    # the lower rank is chosen for the job.
                    with unswitch:
                        if not deterministic:
                            interact = (rank < rank_send)
                            only_supply = False
        # Communicate the dependent variables (e.g. pos for gravity) of
        # the supplier. For pairing_level == 'domain', communicate all
        # local particles. For pairing_level == 'tile', we only need to
        # communicate particles within the tiles that are going to
        # interact during the current domain-domain pairing.
        with unswitch:
            if 𝔹[pairing_level == 'tile']:
                # Find interacting tiles
                tile_indices = domain_domain_tile_indices(
                    receiver, supplier_local, only_supply_passed, domain_pair_nr, interaction_name)
                tile_indices_receiver = tile_indices[0, :]
                tile_indices_supplier = tile_indices[1, :]
            else:  # pairing_level == 'domain'
                # For domain level pairing we make use of
                # the trivial tiling, containing a single tile.
                tile_indices_receiver = tile_indices_supplier = tile_indices_trivial
        supplier_extrl = sendrecv_component(
            supplier_local, dependent, pairing_level, interaction_name, tile_indices_supplier,
            dest=rank_send, source=rank_recv,
        )
        # Let the local receiver interact with the
        # external supplier_extrl. This will update the affected
        # variables (e.g. mom for gravity) of the local receiver and
        # populate the affected variable buffers of the
        # external supplier, if only_supply is False.
        if interact:
            with unswitch:
                if 𝔹[pairing_level == 'tile']:
                    # Further refer the interaction to the tile level
                    tile_tile(
                        receiver, supplier_extrl, tile_indices_receiver, tile_indices_supplier,
                        interaction, rank_recv, only_supply_passed, only_supply, domain_pair_nr,
                        interaction_name, ᔑdt, interaction_extra_args,
                    )
                else:  # pairing_level == 'domain'
                    # Perform the interaction now, at the domain level
                    interaction(
                        receiver, supplier_extrl, pairing_level,
                        tile_indices_receiver, tile_indices_supplier,
                        rank_recv, only_supply, ᔑdt, interaction_extra_args,
                    )
        # Send the populated buffers back to the process from which the
        # external supplier_extrl came. Add the received values in the
        # buffers to the affected variables (e.g. mom for gravity) of
        # the local supplier_local. Note that we should not do this in
        # the case of a local interaction (rank_send == rank) or in a
        # case where only_supply is True.
        if rank_send != rank and not only_supply:
            sendrecv_component(
                supplier_extrl, affected, pairing_level, interaction_name, tile_indices_supplier,
                dest=rank_recv, source=rank_send, component_recv=supplier_local,
            )
            # Nullify the Δ buffers of the external supplier_extrl,
            # leaving this with no leftover junk.
            supplier_extrl.nullify_Δ(affected)
# Tile indices for the trivial tiling,
# used by the domain_domain function.
cython.declare(tile_indices_trivial='Py_ssize_t[::1]')
tile_indices_trivial = zeros(1, dtype=C2np['Py_ssize_t'])

# Function returning the indices of the tiles of the local receiver and
# supplier which take part in tile-tile interactions under the
# domain-domain pairing with number domain_pair_nr.
@cython.header(
    # Arguments
    receiver='Component',
    supplier='Component',
    only_supply='bint',
    domain_pair_nr='Py_ssize_t',
    interaction_name=str,
    # Locals
    dim='int',
    domain_pair_offsets='Py_ssize_t[:, ::1]',
    domain_pair_offset='Py_ssize_t[::1]',
    sign='int',
    tile_indices='Py_ssize_t[:, ::1]',
    tile_indices_all=list,
    tile_indices_component='Py_ssize_t[::1]',
    tile_indices_list=list,
    tile_layout='Py_ssize_t[:, :, ::1]',
    tile_layout_slice_end='Py_ssize_t[::1]',
    tile_layout_slice_start='Py_ssize_t[::1]',
    tiling='Tiling',
    tiling_name=str,
    returns='Py_ssize_t[:, ::1]',
)
def domain_domain_tile_indices(receiver, supplier, only_supply, domain_pair_nr, interaction_name):
    tile_indices_all = domain_domain_tile_indices_dict.get((receiver, supplier, only_supply))
    if tile_indices_all is None:
        tile_indices_all = [None]*27
        domain_domain_tile_indices_dict[receiver, supplier, only_supply] = tile_indices_all
    else:
        tile_indices = tile_indices_all[domain_pair_nr]
        if tile_indices is not None:
            return tile_indices
    tile_layout_slice_start = empty(3, dtype=C2np['Py_ssize_t'])
    tile_layout_slice_end   = empty(3, dtype=C2np['Py_ssize_t'])
    domain_pair_offsets = domain_domain_communication_dict[
        'tile', only_supply, 'domain_pair_offsets']
    domain_pair_offset = domain_pair_offsets[domain_pair_nr, :]
    tile_indices_list = []
    tiling_name = f'{interaction_name} (tiles)'
    for i, component in enumerate((receiver, supplier)):
        tiling = component.tilings[tiling_name]
        tile_layout = tiling.layout
        sign = {0: -1, 1: +1}[i]
        for dim in range(3):
            if domain_pair_offset[dim] == -sign:
                tile_layout_slice_start[dim] = 0
                tile_layout_slice_end[dim]   = 1
            elif domain_pair_offset[dim] == 0:
                tile_layout_slice_start[dim] = 0
                tile_layout_slice_end[dim]   = tile_layout.shape[dim]
            elif domain_pair_offset[dim] == +sign:
                tile_layout_slice_start[dim] = tile_layout.shape[dim] - 1
                tile_layout_slice_end[dim]   = tile_layout.shape[dim]
        tile_indices_component = asarray(tile_layout[
            tile_layout_slice_start[0]:tile_layout_slice_end[0],
            tile_layout_slice_start[1]:tile_layout_slice_end[1],
            tile_layout_slice_start[2]:tile_layout_slice_end[2],
        ]).flatten()
        tile_indices_list.append(tile_indices_component)
    tile_indices = asarray(tile_indices_list, dtype=C2np['Py_ssize_t'])
    tile_indices_all[domain_pair_nr] = tile_indices
    return tile_indices
# Cached results of the domain_domain_tile_indices function
# are stored in the dict below.
cython.declare(domain_domain_tile_indices_dict=dict)
domain_domain_tile_indices_dict = {}

# Function returning the process ranks with which to pair
# the local process/domain in the domain_domain function,
# depending on the pairing level and supplier only supplies
# or also receives.
@cython.header(
    # Arguments
    pairing_level=str,
    only_supply='bint',
    # Locals
    i='Py_ssize_t',
    returns=tuple,
)
def domain_domain_communication(pairing_level, only_supply):
    ranks = domain_domain_communication_dict.get((pairing_level, only_supply))
    if ranks:
        return ranks
    if pairing_level == 'domain':
        # When only_supply is True, each process should be paired with
        # all processes. When only_supply is False, advantage is taken
        # of the fact that a process is paired with two other processes
        # simultaneously, meaning that the number of pairings is cut
        # (roughly) in half. The particular order implemented below
        # is of no importance.
        N_domain_pairs = nprocs if only_supply else 1 + nprocs//2
        ranks_send = np.empty(N_domain_pairs, dtype=C2np['int'])
        ranks_recv = np.empty(N_domain_pairs, dtype=C2np['int'])
        for i in range(N_domain_pairs):
            ranks_send[i] = mod(rank + i, nprocs)
            ranks_recv[i] = mod(rank - i, nprocs)
        domain_domain_communication_dict[pairing_level, only_supply] = (ranks_send, ranks_recv)
    elif pairing_level == 'tile':
        # When only_supply is True, each domian should be paired with
        # itself and all 26 neighbouring domains. Even though we might
        # have nprocs < 27, meaning that some of the neighbouring
        # domains might be the same, we always include all of them.
        # If only_supply is False, advantage is taken of the fact that a
        # domain is simultaneously paired with two other domains along
        # the same direction (e.g. to the left and to the right),
        # cutting the number of pairings (roughly) in half. The order is
        # as specified below, and stored (as directions, not ranks) in
        # domain_domain_communication_dict[
        #     'tile', only_supply, 'domain_pair_offsets'].
        ranks_send = []
        ranks_recv = []
        offsets_list = []
        # - This domain itself
        offsets = np.array([0, 0, 0], dtype=C2np['int'])
        offsets_list.append(offsets.copy())
        ranks_send.append(rank_neighbouring_domain(*(+offsets)))
        ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
        # - Domains at the 6 faces
        #   (when only_supply is False, send right, forward, upward)
        direction = np.array([+1, 0, 0], dtype=C2np['int'])
        for i in range(3):
            offsets = np.roll(direction, i)
            offsets_list.append(offsets.copy())
            ranks_send.append(rank_neighbouring_domain(*(+offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
            if only_supply:
                offsets_list.append(-offsets)
                ranks_send.append(rank_neighbouring_domain(*(-offsets)))
                ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
        # - Domains at the 12 edges
        #   (when only_supply is False, send
        #     {right  , forward}, {left     , forward },
        #     {forward, upward }, {backward , upward  },
        #     {right  , upward }, {rightward, downward},
        # )
        direction = np.array([+1, +1,  0], dtype=C2np['int'])
        flip      = np.array([-1, +1, +1], dtype=C2np['int'])
        for i in range(3):
            offsets = np.roll(direction, i)
            offsets_list.append(offsets.copy())
            ranks_send.append(rank_neighbouring_domain(*(+offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
            if only_supply:
                offsets_list.append(-offsets)
                ranks_send.append(rank_neighbouring_domain(*(-offsets)))
                ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
            offsets *= np.roll(flip, i)
            offsets_list.append(offsets.copy())
            ranks_send.append(rank_neighbouring_domain(*(+offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
            if only_supply:
                offsets_list.append(-offsets)
                ranks_send.append(rank_neighbouring_domain(*(-offsets)))
                ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
        # - Domains at the 8 corners
        #   (when only_supply is False, send
        #    {right, forward , upward  },
        #    {right, forward , downward},
        #    {left , forward , upward  },
        #    {right, backward, upward  },
        # )
        offsets = np.array([+1, +1, +1], dtype=C2np['int'])
        offsets_list.append(offsets.copy())
        ranks_send.append(rank_neighbouring_domain(*(+offsets)))
        ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
        if only_supply:
            offsets_list.append(-offsets)
            ranks_send.append(rank_neighbouring_domain(*(-offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
        direction = np.array([+1, +1, -1], dtype=C2np['int'])
        for i in range(3):
            offsets = np.roll(direction, i)
            offsets_list.append(offsets.copy())
            ranks_send.append(rank_neighbouring_domain(*(+offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
            if only_supply:
                offsets_list.append(-offsets)
                ranks_send.append(rank_neighbouring_domain(*(-offsets)))
                ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
        domain_domain_communication_dict[pairing_level, only_supply] = (
            (np.array(ranks_send, dtype=C2np['int']), np.array(ranks_recv, dtype=C2np['int']))
        )
        domain_domain_communication_dict[pairing_level, only_supply, 'domain_pair_offsets'] = (
            np.array(offsets_list, dtype=C2np['Py_ssize_t'])
        )
    else:
        abort(
            f'domain_domain_communication() got '
            f'pairing_level = {pairing_level} ∉ {{"domain", "tile"}}'
        )
    return domain_domain_communication_dict[pairing_level, only_supply]
# Cached results of the domain_domain_communication function
# are stored in the dict below.
cython.declare(domain_domain_communication_dict=dict)
domain_domain_communication_dict = {}

# Generic function implementing tile-tile pairing
@cython.header(
    # Arguments
    receiver='Component',
    supplier='Component',
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier='Py_ssize_t[::1]',
    interaction=func_interaction,
    rank_supplier='int',
    only_supply_passed='bint',
    only_supply='bint',
    domain_pair_nr='Py_ssize_t',
    interaction_name=str,
    ᔑdt=dict,
    interaction_extra_args=dict,
    # Locals
    i='Py_ssize_t',
    pairing_level=str,
    tile_indices_receiver_supplier=object,  # np.ndarray of dtype object
    tile_indices_supplier_paired='Py_ssize_t[::1]',
    returns='void',
)
def tile_tile(
    receiver, supplier, tile_indices_receiver, tile_indices_supplier,
    interaction, rank_supplier, only_supply_passed, only_supply, domain_pair_nr,
    interaction_name, ᔑdt, interaction_extra_args,
):
    """This function takes care of pairings between neighbouring tiles
    within a domain and boundary tiles at the interface between two
    domains.
    If the supplier component is external to this rank, it is expected
    that it has been properly communicated. Also, both the receiver and
    the supplier components are expected to already be tile sorted.
    """
    # The strategy for pairing the tiles is as follows.
    # - We always have just a single receiver tile at a time, as no two
    #   tiles can ever need the same set of other tiles with which to
    #   interact (except for the case of just two neighbour tiles,
    #   but that is covered by a single receiver and a single
    #   supplier tile).
    # - We thus loop throug the receiver tiles one by one, and pair them
    #   up with as many supplier tiles as possible. Thee first receiver
    #   tile then gets paired with all of the tiles in
    #   tile_indices_supplier which happen to be a neighbour tile.
    #   Later receiver tiles may not get assigned all neighbouring
    #   supplier tiles, as the {receiver tile, supplier tile} pair may
    #   already have been encountered. This avoidance of double counting
    #   should only be done in the case of rank == rank_supplier.
    # Get the supplier tile indices
    # with which to pair each receiver tile.
    tile_indices_receiver_supplier = get_tile_tile_pairs(
        receiver, supplier, tile_indices_receiver, tile_indices_supplier,
        rank_supplier, only_supply_passed, domain_pair_nr, interaction_name,
    )
    # For each receiver tile, call the interaction with the required
    # neighbouring supplier tiles.
    pairing_level = 'tile'
    for i in range(tile_indices_receiver.shape[0]):
        tile_indices_supplier_paired = tile_indices_receiver_supplier[i]
        interaction(
            receiver, supplier, pairing_level,
            tile_indices_receiver[i:i+1], tile_indices_supplier_paired,
            rank_supplier, only_supply, ᔑdt, interaction_extra_args,
        )

# Function that given arrays of receiver and supplier tiles
# returns them in paired format.
@cython.header(
    # Arguments
    receiver='Component',
    supplier='Component',
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier='Py_ssize_t[::1]',
    rank_supplier='int',
    only_supply_passed='bint',
    domain_pair_nr='Py_ssize_t',
    interaction_name=str,
    # Locals
    dim='int',
    domain_pair_offset='Py_ssize_t[::1]',
    global_tile_layout_shape='Py_ssize_t[::1]',
    i='Py_ssize_t',
    j='Py_ssize_t',
    key=tuple,
    l='Py_ssize_t',
    l_offset='Py_ssize_t',
    l_s='Py_ssize_t',
    m='Py_ssize_t',
    m_offset='Py_ssize_t',
    m_s='Py_ssize_t',
    n='Py_ssize_t',
    n_offset='Py_ssize_t',
    n_s='Py_ssize_t',
    neighbourtile_index_3D_global='Py_ssize_t[::1]',
    suppliertile_indices_3D_global_to_1D_local=dict,
    tile_index_3D_global_s=tuple,
    tile_index_r='Py_ssize_t',
    tile_index_s='Py_ssize_t',
    tile_layout='Py_ssize_t[:, :, ::1]',
    tiling='Tiling',
    tiling_name=str,
    wraparound='bint',
    returns=object,  # np.ndarray of dtype object
)
def get_tile_tile_pairs(
    receiver, supplier, tile_indices_receiver, tile_indices_supplier,
    rank_supplier, only_supply_passed, domain_pair_nr, interaction_name,
):
    # Lookup cached result
    key = (receiver.name, supplier.name, interaction_name, domain_pair_nr)
    tile_indices_receiver_supplier = tile_indices_receiver_supplier_dict.get(key)
    if tile_indices_receiver_supplier is not None:
        return tile_indices_receiver_supplier
    # List of lists storing the supplier tile indices
    # for each receiver tile. The type of this data structure will
    # change during the computation.
    tile_indices_receiver_supplier = [[] for i in range(tile_indices_receiver.shape[0])]
    # Get the shape of the local (domain) tile layout,
    # as well as of the global (box) tile layout.
    tiling_name = f'{interaction_name} (tiles)'
    tiling = receiver.tilings[tiling_name]
    tile_layout = tiling.layout
    tile_layout_shape = asarray(asarray(tile_layout).shape)
    # The general computation below takes a long time when dealing with
    # many tiles. By far the worst case is when all tiles in the local
    # domain should be paired with themselves, which is the case for
    # domain_pair_nr == 0. For this case we perform a much faster,
    # more specialised computation.
    if domain_pair_nr == 0:
        if rank != rank_supplier:
            abort(
                f'get_tile_tile_pairs() got rank_supplier = {rank_supplier} != rank = {rank} '
                f'at domain_pair_nr == 0'
            )
        if not np.all(asarray(tile_indices_receiver) == asarray(tile_indices_supplier)):
            abort(
                f'get_tile_tile_pairs() got tile_indices_receiver != tile_indices_supplier '
                f'at domain_pair_nr == 0'
            )
        i = 0
        for         l in range(tile_layout.shape[0]):
            for     m in range(tile_layout.shape[1]):
                for n in range(tile_layout.shape[2]):
                    if i != tile_layout[l, m, n]:
                        abort(
                            f'It looks as though the tile layout of {receiver.name} is incorrect'
                        )
                    neighbourtile_indices_supplier = tile_indices_receiver_supplier[i]
                    for l_offset in range(-1, 2):
                        l_s = l + l_offset
                        if l_s == -1 or l_s == ℤ[tile_layout.shape[0]]:
                            continue
                        for m_offset in range(-1, 2):
                            m_s = m + m_offset
                            if m_s == -1 or m_s == ℤ[tile_layout.shape[1]]:
                                continue
                            for n_offset in range(-1, 2):
                                n_s = n + n_offset
                                if n_s == -1 or n_s == ℤ[tile_layout.shape[2]]:
                                    continue
                                tile_index_s = tile_layout[l_s, m_s, n_s]
                                if tile_index_s >= i:
                                    neighbourtile_indices_supplier.append(tile_index_s)
                    tile_indices_receiver_supplier[i] = asarray(
                        neighbourtile_indices_supplier, dtype=C2np['Py_ssize_t'],
                    )
                    i += 1
    else:
        # Get relative offsets of the domains currently being paired
        domain_pair_offset = domain_domain_communication_dict[
            'tile', only_supply_passed, 'domain_pair_offsets'][domain_pair_nr, :]
        # Get the indices of the global domain layout matching the
        # receiver (local) domain and supplier domain.
        domain_layout_receiver_indices = asarray(
            np.unravel_index(rank, domain_subdivisions)
        )
        domain_layout_supplier_indices = asarray(
            np.unravel_index(rank_supplier, domain_subdivisions)
        )
        global_tile_layout_shape = asarray(
            asarray(domain_subdivisions)*tile_layout_shape,
            dtype=C2np['Py_ssize_t'],
        )
        tile_index_3D_r_start = domain_layout_receiver_indices*tile_layout_shape
        tile_index_3D_s_start = domain_layout_supplier_indices*tile_layout_shape
        # Construct dict mapping global supplier 3D indices to their
        # local 1D counterparts.
        suppliertile_indices_3D_global_to_1D_local = {}
        for j in range(tile_indices_supplier.shape[0]):
            tile_index_s = tile_indices_supplier[j]
            tile_index_3D_s = asarray(tiling.tile_index3D(tile_index_s))
            tile_index_3D_global_s = tuple(tile_index_3D_s + tile_index_3D_s_start)
            suppliertile_indices_3D_global_to_1D_local[tile_index_3D_global_s] = tile_index_s
        # Pair each receiver tile with all neighbouring supplier tiles
        for i in range(tile_indices_receiver.shape[0]):
            neighbourtile_indices_supplier = tile_indices_receiver_supplier[i]
            # Construct global 3D index of this receiver tile
            tile_index_r = tile_indices_receiver[i]
            tile_index_3D_r = asarray(tiling.tile_index3D(tile_index_r))
            tile_index_3D_global_r = tile_index_3D_r + tile_index_3D_r_start
            # Loop over all neighbouring receiver tiles
            # (including the tile itself).
            for         l in range(-1, 2):
                for     m in range(-1, 2):
                    for n in range(-1, 2):
                        neighbourtile_index_3D_global = asarray(
                            tile_index_3D_global_r + asarray((l, m, n)),
                            dtype=C2np['Py_ssize_t'],
                        )
                        # For domain_pair_nr == 0, all tiles in the
                        # local domain are paired with all others.
                        # Here we must not take the periodicity into
                        # account, as such interacions are performed by
                        # future domain pairings.
                        with unswitch:
                            if domain_pair_nr == 0:
                                wraparound = False
                                for dim in range(3):
                                    if not (
                                        0 <= neighbourtile_index_3D_global[dim]
                                          <  global_tile_layout_shape[dim]
                                    ):
                                        wraparound = True
                                        break
                                if wraparound:
                                    continue
                        # Take the periodicity of the domain layout
                        # into account. This should only be done
                        # along the direction(s) connecting
                        # the paired domains.
                        with unswitch:
                            if 𝔹[domain_pair_offset[0] != 0]:
                                neighbourtile_index_3D_global[0] = mod(
                                    neighbourtile_index_3D_global[0],
                                    global_tile_layout_shape[0],
                                )
                        with unswitch:
                            if 𝔹[domain_pair_offset[1] != 0]:
                                neighbourtile_index_3D_global[1] = mod(
                                    neighbourtile_index_3D_global[1],
                                    global_tile_layout_shape[1],
                                )
                        with unswitch:
                            if 𝔹[domain_pair_offset[2] != 0]:
                                neighbourtile_index_3D_global[2] = mod(
                                    neighbourtile_index_3D_global[2],
                                    global_tile_layout_shape[2],
                                )
                        # Check if a supplier tile sits at the location
                        # of the current neighbour tile.
                        tile_index_s = suppliertile_indices_3D_global_to_1D_local.get(
                            tuple(neighbourtile_index_3D_global),
                            -1,
                        )
                        if tile_index_s != -1:
                            # For domain_pair_nr == 0, all tiles in the
                            # local domain are paired with all others.
                            # To not double count, we disregard the
                            # pairing if the supplier index is lower
                            # than the receiver.
                            with unswitch:
                                if domain_pair_nr == 0:
                                    if tile_index_s < tile_index_r:
                                        continue
                            neighbourtile_indices_supplier.append(tile_index_s)
            # Convert the neighbouring supplier tile indices from a list
            # to an Py_ssize_t[::1] array.
            # We also sort the indices, though this is not necessary.
            neighbourtile_indices_supplier = asarray(
                neighbourtile_indices_supplier, dtype=C2np['Py_ssize_t'],
            )
            neighbourtile_indices_supplier.sort()
            tile_indices_receiver_supplier[i] = neighbourtile_indices_supplier
    # Transform tile_indices_receiver_supplier to an object array,
    # the elements of which is arrays of dtype Py_ssize_t.
    tile_indices_receiver_supplier = asarray(tile_indices_receiver_supplier, dtype=object)
    # If all arrays in tile_indices_receiver_supplier are of the
    # same size, it will not be stored as an object array of Py_ssize_t
    # arrays, but instead a 2D object array. In compiled mode this leads
    # to a crash, as elements of tile_indices_receiver_supplier must be
    # compatible with Py_ssize_t[::1]. We can convert it to a 2D
    # Py_ssize_t array instead, single-index elements of which exactly
    # are the Py_ssize_t[::1] arrays we need. When the arrays in
    # tile_indices_receiver_supplier are not of the same size, such a
    # conversion will fail. Since we do not care about the conversion
    # in such a case anyway, we always just attempt to do
    # the conversion. If it succeed, it was needed. If not, it was not
    # needed anyway.
    try:
        tile_indices_receiver_supplier = asarray(
            tile_indices_receiver_supplier, dtype=C2np['Py_ssize_t'])
    except ValueError:
        pass
    # Cache and return result
    tile_indices_receiver_supplier_dict[key] = tile_indices_receiver_supplier
    return tile_indices_receiver_supplier
# Cache uses by the get_tile_tile_pairs function
cython.declare(tile_indices_receiver_supplier_dict=dict)
tile_indices_receiver_supplier_dict = {}

# Generic function implementing particle-mesh interactions
# for both particle and fluid componenets.
@cython.header(
    # Arguments
    receivers=list,
    suppliers=list,
    ᔑdt=dict,
    ᔑdt_key=object,  # str or tuple
    potential=func_potential,
    potential_name=str,
    dependent=list,
    # Locals
    J_dim='FluidScalar',
    component='Component',
    dim='int',
    gradφ_dim='double[:, :, ::1]',
    i='Py_ssize_t',
    mom_dim='double*',
    posx='double*',
    posy='double*',
    posz='double*',
    representation=str,
    x='double',
    y='double',
    z='double',
    Δx_φ='double',
    φ='double[:, :, ::1]',
    φ_dict=dict,
)
def particle_mesh(receivers, suppliers, ᔑdt, ᔑdt_key, potential, potential_name, dependent):
    """This function will update the affected variables of all receiver
    components due to an interaction. This is done by constructing
    global fields by interpolating the dependent variables of all
    suppliers onto grids.
    The supplied 'dependent' argument is thus a list of variables which
    should be interpolated to the grid. For details on the structure
    of this argument, see the CIC_components2domain_grid function
    in the mesh module, where the corresponding argument is called
    quantities.

    Two global grids are used, φ_particles and φ_fluids, both of which
    contain the entire density/potential field of both particle and
    fluid components. For φ_fluids, a CIC deconvolution will take place
    on the contribution from particles only, corresponding to the
    interpolation from particles onto the grid. For φ_particles, this
    same deconvolution will take place, but in addition both the
    particle and fluid contribution will be deconvolved once more,
    corresponding to the interpolation from the grid back to
    the particles.

    The deconvolutions take place in Fourier space. Also while in
    Fourier space, the grids are transformed to the (Fourier
    transformed) potential by multiplying each grid point by
    potential(k2), where k2 = k² is the squared magnitude of the wave
    vector at the given grid point. For further details on the potential
    argument, see the construct_potential function.

    The grids are then Fourier transformed back to real space and
    differentiated along each dimension to get the force.

    This force is then applied to all receivers using the prescription
    Δmom = -mass*∂ⁱφ*ᔑdt[ᔑdt_key]
    """
    # Build the two potentials due to all particles and fluid suppliers
    masterprint(
        f'Constructing the {potential_name} due to {suppliers[0].name} ...'
        if len(suppliers) == 1 else (
            f'Constructing the {potential_name} due to {{{{{{}}}}}} ...'
            .format(', '.join([component.name for component in suppliers]))
        )
    )
    φ_dict = construct_potential(receivers, suppliers, dependent, potential)
    masterprint('done')
    # For each dimension, differentiate the potentials
    # and apply the force to all receivers.
    Δx_φ = boxsize/φ_gridsize  # Physical grid spacing of φ
    for representation, φ in φ_dict.items():
        for dim in range(3):
            masterprint(
                f'Differentiating the ({representation}) {potential_name} along the '
                f'{"xyz"[dim]}-direction and applying it ...'
            )
            # Do the differentiation of φ
            gradφ_dim = diff_domain(φ, dim, Δx_φ, order=4)
            # Apply force to all the receivers
            for component in receivers:
                if component.representation != representation:
                    continue
                if 𝔹[isinstance(ᔑdt_key, tuple)]:
                    ᔑdt_key = (ᔑdt_key[0], component.name)
                masterprint(f'Applying to {component.name} ...')
                if component.representation == 'particles':
                    # Extract variables from component
                    posx    = component.posx
                    posy    = component.posy
                    posz    = component.posz
                    mom_dim = component.mom[dim]
                    # Update the dim momentum component of all particles
                    for i in range(component.N_local):
                        # The coordinates of the i'th particle,
                        # transformed so that 0 <= x, y, z < 1.
                        x = (posx[i] - domain_start_x)/domain_size_x
                        y = (posy[i] - domain_start_y)/domain_size_y
                        z = (posz[i] - domain_start_z)/domain_size_z
                        # Look up the force via a CIC interpolation,
                        # convert it to momentum and subtract it from
                        # the momentum of particle i (subtraction
                        # because the force is the negative gradient of
                        # the potential). The factor with which to
                        # multiply gradφ_dim by to get momentum updates
                        # is -mass*Δt, where Δt = ᔑdt['1'].
                        # Here this integral over the time step is
                        # generalised and supplied by the caller.
                        mom_dim[i] -= ℝ[component.mass*ᔑdt[ᔑdt_key]
                            ]*CIC_scalargrid2coordinates(gradφ_dim, x, y, z)
                elif component.representation == 'fluid':
                    # Simply scale and extrapolate the values in
                    # gradφ_dim to the grid points of the dim'th
                    # component of the fluid variable J.
                    # First extract this fluid scalar.
                    J_dim = component.J[dim]
                    # The source term has the form
                    # ∝ -(ϱ + c⁻²𝒫)*∂ⁱφ,
                    # and so we need to multiply each grid point
                    # [i, j, k] in gradφ_dim by
                    # (ϱ[i, j, k] + c⁻²𝒫[i, j, k]). As we are interested
                    # in the momentum change, we should also multiply
                    # all grind points by the factor Δt = ᔑdt['1'].
                    # Here this integral over the time step is
                    # generalised and supplied by the caller.
                    CIC_grid2grid(J_dim.grid_noghosts, gradφ_dim,
                        fac=ℝ[-ᔑdt[ᔑdt_key]],
                        fac_grid=component.ϱ.grid_noghosts,
                        fac2=light_speed**(-2)*ℝ[-ᔑdt[ᔑdt_key]],
                        fac_grid2=component.𝒫.grid_noghosts,
                    )
                    communicate_domain(J_dim.grid_mv, mode='populate')
                masterprint('done')
            masterprint('done')

# Generic function capable of constructing potential grids out of
# components and a given expression for the potential.
@cython.header(
    # Arguments
    receivers=list,
    suppliers=list,
    quantities=list,
    potential=func_potential,
    # Locals
    any_fluid_receivers='bint',
    any_fluid_suppliers='bint',
    any_particles_receivers='bint',
    any_particles_suppliers='bint',
    deconv_factor='double',
    deconv_ijk='double',
    fft_normalization_factor='double',
    i='Py_ssize_t',
    j='Py_ssize_t',
    j_global='Py_ssize_t',
    kj='Py_ssize_t',
    kj2='Py_ssize_t',
    k='Py_ssize_t',
    ki='Py_ssize_t',
    kk='Py_ssize_t',
    k2='Py_ssize_t',
    potential_factor='double',
    receiver_representations=set,
    reciprocal_sqrt_deconv_ij='double',
    reciprocal_sqrt_deconv_ijk='double',
    reciprocal_sqrt_deconv_j='double',
    representation=str,
    representation_counter='int',
    slab='double[:, :, ::1]',
    slab_dict=dict,
    slab_fluid='double[:, :, ::1]',
    slab_fluid_jik='double*',
    slab_jik='double*',
    slab_particles_jik='double*',
    supplier_representations=set,
    φ='double[:, :, ::1]',
    φ_dict=dict,
    returns=dict,
)
def construct_potential(receivers, suppliers, quantities, potential):
    """This function populate two grids (including pseudo points and
    ghost layers) with a real-space potential corresponding to the
    Fourier-space potential function given, due to all supplier
    components. A seperate grid for particle and fluid components will
    be constructed, the difference being only the handling of
    deconvolutions needed for the interpolation to/from the grid.
    Both grids will contain the potential due to all the components.
    Which variables to extrapolate to the grid is determined by the
    quantities argument. For details on this argument, see the
    CIC_components2domain_grid function in the mesh module.

    First the variables given in 'quantities' of the components are
    interpolated to the grids; particle components to one grid and
    fluid components to a seperate grid. The two grids are then Fourier
    transformed.
    The potential function is then used to change the value of each grid
    point for both grids. Also while in Fourier space, deconvolutions
    will be carried out, in a different manner for each grid.
    The two grids are added in such a way that they both corresponds to
    the total potential of all components, but deconvolved in the way
    suitable for either particles or fluids. Note that if a fluid
    component have a gridsize different from φ_gridsize, interpolation
    will take place but no deconvolution will be made, leading to
    errors on small scales.
    The two grids are now Fourier transformed back to real space.

    In the case of normal gravity, we have
    φ(k) = -4πGa²ρ(k)/k² = -4πG a**(-3*w_eff - 1) ϱ(k)/k²,
    which can be signalled by passing
    quantities = [('particles', a**(-3*w_eff - 1)*mass/Vcell),
                  ('ϱ', a**(-3*w_eff - 1))],
    potential = lambda k2: -4*π*G_Newton/k2
    (note that it is not actally allowed to pass an untyped lambda
    function in compiled mode).
    """
    # Flags specifying whether any fluid/particle components
    # are present among the receivers and among the suppliers.
    receiver_representations = {receiver.representation for receiver in receivers}
    any_particles_receivers  = ('particles' in receiver_representations)
    any_fluid_receivers      = ('fluid'     in receiver_representations)
    if not any_particles_receivers and not any_fluid_receivers:
        abort('construct_potential() got no recognizable receivers')
    supplier_representations = {supplier.representation for supplier in suppliers}
    any_particles_suppliers  = ('particles' in supplier_representations)
    any_fluid_suppliers      = ('fluid'     in supplier_representations)
    if not any_particles_suppliers and not any_fluid_suppliers:
        abort('construct_potential() got no recognizable suppliers')
    # CIC interpolate the particles/fluid elements onto the grids.
    # The φ_dict will be a dictionary mapping representations
    # ('particles', 'fluid') to grids. If only one representation is
    # present among the suppliers, only this item will exist in the
    # dictionary. In the case where a representation is present among
    # the receivers that are not among the suppliers however, we do need
    # this missing grid in the φ_dict. To ensure at least a nullified
    # grid of the needed representations, we pass in the needed
    # representations in the 'ensure' argument.
    φ_dict = CIC_components2φ(
        suppliers, quantities, ensure=' '.join(list(receiver_representations)),
    )
    # Slab decompose the grids
    slab_dict = {
        representation: slab_decompose(φ, f'φ_{representation}_slab', prepare_fft=True)
        for representation, φ in φ_dict.items()
    }
    if 'fluid' in slab_dict:
        slab_fluid = slab_dict['fluid']
    # In the case of both particle and fluid components being present,
    # it is important that the particle slabs are handled after the
    # fluid slabs, as the deconvolution factor is only computed for
    # particle components and this is needed after combining the fluid
    # and particle slabs. It is also important that the order of
    # representations in slab_dict and φ_dict is the same.
    if 'fluid' in slab_dict and 'particles' in slab_dict:
        slab_dict = {
            representation: slab_dict[representation] for representation in ('fluid', 'particles')
        }
        φ_dict = {
            representation: φ_dict[representation] for representation in ('fluid', 'particles')
        }
    # Do a forward in-place Fourier transform of the slabs.
    # In the case of nullified grids being present solely becasue they
    # are needed due to the receiver representation,
    # we can skip the FFT.
    for representation, slab in slab_dict.items():
        if representation == 'particles' and not any_particles_suppliers:
            continue
        if representation == 'fluid'     and not any_fluid_suppliers:
            continue
        fft(slab, 'forward')
    # Multiplicative factor needed after a forward and a backward
    # Fourier transformation.
    fft_normalization_factor = float(φ_gridsize)**(-3)
    # For each grid, multiply by the potential and deconvolution
    # factors. Do fluid slabs fist, then particle slabs.
    for representation_counter, (representation, slab) in enumerate(slab_dict.items()):
        # No need to process the fluid grid if it consist purely
        # of zeros (i.e. no fluid suppliers exist).
        if 𝔹[representation == 'fluid' and not any_fluid_suppliers]:
            continue
        # Begin loop over slabs. As the first and second dimensions
        # are transposed due to the FFT, start with the j-dimension.
        for j in range(ℤ[slab.shape[0]]):
            # The j-component of the wave vector (grid units).
            # Since the slabs are distributed along the j-dimension,
            # an offset must be used.
            j_global = ℤ[slab.shape[0]*rank] + j
            if j_global > ℤ[φ_gridsize//2]:
                kj = j_global - φ_gridsize
            else:
                kj = j_global
            kj2 = kj**2
            # Reciprocal square root of the j-component of the deconvolution
            with unswitch(1):
                if 𝔹[representation == 'particles']:
                    reciprocal_sqrt_deconv_j = sinc(kj*ℝ[π/φ_gridsize])
            # Loop through the complete i-dimension
            for i in range(φ_gridsize):
                # The i-component of the wave vector (grid units)
                if i > ℤ[φ_gridsize//2]:
                    ki = i - φ_gridsize
                else:
                    ki = i
                # Reciprocal square root of the product of the i-
                # and the j-component of the deconvolution.
                with unswitch(2):
                    if 𝔹[representation == 'particles']:
                        reciprocal_sqrt_deconv_ij = (
                            sinc(ki*ℝ[π/φ_gridsize])*reciprocal_sqrt_deconv_j
                        )
                # Loop through the complete, padded k-dimension
                # in steps of 2 (one complex number at a time).
                for k in range(0, ℤ[slab.shape[2]], 2):
                    # The k-component of the wave vector (grid units)
                    kk = k//2
                    # The squared magnitude of the wave vector (grid units)
                    k2 = ℤ[ki**2 + kj2] + kk**2
                    # Pointer to the [j, i, k]'th element of the slab.
                    # The complex number is then given as
                    # Re = slab_jik[0], Im = slab_jik[1].
                    slab_jik = cython.address(slab[j, i, k:])
                    # Enforce the vanishing of the potential at |k| = 0.
                    # The real-space mean value of the potential will
                    # then be zero, as it should for a
                    # peculiar potential.
                    if k2 == 0:
                        slab_jik[0] = 0  # Real part
                        slab_jik[1] = 0  # Imag part
                        continue
                    # Get the factor from the potential function at
                    # this k². The physical squared length of the wave
                    # vector is given by (2π/boxsize*|k|)².
                    # The particles grid only need to be processed if it
                    # is not zero (i.e. particle suppliers exist).
                    with unswitch(3):
                        if 𝔹[representation == 'fluid' or any_particles_suppliers]:
                            potential_factor = potential(ℝ[(2*π/boxsize)**2]*k2)
                    # The final deconvolution factor
                    with unswitch(3):
                        if 𝔹[representation == 'particles']:
                            # Reciprocal square root of the product of
                            # all components of the deconvolution.
                            reciprocal_sqrt_deconv_ijk = (
                                reciprocal_sqrt_deconv_ij*sinc(kk*ℝ[π/φ_gridsize])
                            )
                            # The total factor
                            # for a complete deconvolution.
                            deconv_ijk = 1/reciprocal_sqrt_deconv_ijk**2
                            # A deconvolution of the particle potential
                            # is needed due to the interpolation from
                            # the particle positions to the grid.
                            deconv_factor = deconv_ijk
                            # For particle receivers we will need to do
                            # a second deconvolution due to the
                            # interpolation from the grid back to the
                            # particles. In the case where we have only
                            # particle components and thus only a
                            # particles potential, we carry out this
                            # second deconvolution now. If both particle
                            # and fluid components are present,
                            # this second deconvolution
                            # will take place later.
                            with unswitch(4):
                                if 𝔹[not any_fluid_receivers and not any_fluid_suppliers]:
                                    deconv_factor *= deconv_ijk
                        elif 𝔹[representation == 'fluid']:
                            # Do not apply any deconvolution to fluids
                            deconv_factor = 1
                    # Transform this complex grid point.
                    # The particles grid only need to be processed if it
                    # is not zero (i.e. particle suppliers exist).
                    with unswitch(3):
                        if 𝔹[representation == 'fluid' or any_particles_suppliers]:
                            slab_jik[0] *= ℝ[  # Real part
                                potential_factor*deconv_factor*fft_normalization_factor
                            ]
                            slab_jik[1] *= ℝ[  # Imag part
                                potential_factor*deconv_factor*fft_normalization_factor
                            ]
                    # If only particle components or only fluid
                    # components exist, the slabs now store the final
                    # potential in Fourier space. However, if both
                    # particle and fluid components exist, the two sets
                    # of slabs should be combined to form total
                    # potentials. We know that both representations
                    # exist and that we are done handling both (at this
                    # gridpoint) if representation_counter == 1.
                    with unswitch(3):
                        if 𝔹[representation_counter == 1]:
                            # Pointers to this element for both slabs.
                            # As we are looping over the particle slab,
                            # we may reuse the pointer above.
                            slab_particles_jik = slab_jik
                            slab_fluid_jik     = cython.address(slab_fluid[j, i, k:])
                            # Add the particle potential values
                            # to the fluid potential.
                            slab_fluid_jik[0] += slab_particles_jik[0]  # Real part
                            slab_fluid_jik[1] += slab_particles_jik[1]  # Imag part
                            # Now the fluid slabs store the total
                            # potential, with the particle part
                            # deconvolved once due to the interpolation
                            # of the particles to the grid. The particle
                            # slabs should now be a copy of what is
                            # stored in the fluid slabs, but with an
                            # additional deconvolution, accounting for
                            # the upcoming interpolation from the grid
                            # back to the particles.
                            slab_particles_jik[0] = deconv_ijk*slab_fluid_jik[0]  # Real part
                            slab_particles_jik[1] = deconv_ijk*slab_fluid_jik[1]  # Imag part
    # If a representation is present amongst the suppliers but not the
    # receivers, the corresponding (total) potential has been
    # constructed but will not be used. Remove it.
    if any_particles_suppliers and not any_particles_receivers:
        del slab_dict['particles']
        del φ_dict   ['particles']
    if any_fluid_suppliers and not any_fluid_receivers:
        del slab_dict['fluid']
        del φ_dict   ['fluid']
    if not slab_dict:
        abort(
            'Something went wrong in the construct_potential() function, '
            'as it appears that neither particles nor fluids should receive the force '
            'due to the potential'
        )
    # Fourier transform the slabs back to coordinate space
    for slab in slab_dict.values():
        fft(slab, 'backward')
    # Domain-decompose the slabs
    for φ, slab in zip(φ_dict.values(), slab_dict.values()):
        domain_decompose(slab, φ)  # Also populates pseudos and ghost
    # Return the potential grid(s)
    return φ_dict

# Function that carries out the gravitational interaction
@cython.pheader(
    # Arguments
    method=str,
    receivers=list,
    suppliers=list,
    ᔑdt=dict,
    interaction_type=str,
    printout='bint',
    pm_potential=str,
    # Locals
    dependent=list,
    potential=func_potential,
    potential_name=str,
    Δt='double',
    φ_Vcell='double',
    ᔑdt_key=object,  # str or tuple
)
def gravity(method, receivers, suppliers, ᔑdt, interaction_type, printout, pm_potential='full'):
    # Compute gravity via one of the following methods
    if method == 'p3m':
        # The particle-particle-mesh method
        if printout:
            if 𝔹['long' in interaction_type]:
                extra_message = ' (long-range only)'
            elif 𝔹['short' in interaction_type]:
                extra_message = ' (short-range only)'
            else:
                extra_message = ''
            masterprint(
                'Executing',
                shortrange_progress_messages('gravity', method, receivers, extra_message),
                '...',
            )
        # The long-range PM part
        if 𝔹['any' in interaction_type] or 𝔹['long' in interaction_type]:
            gravity('pm', receivers, suppliers, ᔑdt, interaction_type, printout, 'long-range only')
        # The short-range PP part
        if 𝔹['any' in interaction_type] or 𝔹['short' in interaction_type]:
            tabulate_shortrange_gravity()
            component_component(
                receivers, suppliers, gravity_pairwise_shortrange, ᔑdt,
                dependent=['pos'],
                affected=['mom'],
                deterministic=True,
                pairing_level='tile',
                interaction_name='gravity',
            )
        if printout:
            masterprint('done')
    elif method == 'pm':
        # The particle-mesh method.
        if pm_potential == 'full':
            # Use the full gravitational potential
            if printout:
                masterprint(
                    f'Executing gravitational interaction for {receivers[0].name} '
                    f'via the PM method ...'
                    if len(receivers) == 1 else (
                        'Executing gravitational interaction for {{{}}} via the PM method ...'
                        .format(', '.join([component.name for component in receivers]))
                    )
                )
            potential = gravity_potential
            potential_name = 'gravitational potential'
        elif 'long' in pm_potential:
            # Only use the long-range part of the
            # gravitational potential.
            potential = gravity_longrange_potential
            potential_name = 'gravitational long-range potential'
        elif master:
            abort(f'Unrecognized pm_potential = {pm_potential} in gravity()')
        # The gravitational potential is given by the Poisson equation
        # ∇²φ = 4πGa²ρ = 4πGa**(-3*w_eff - 1)ϱ.
        # The factor in front of the dependent variable ϱ is thus
        # time-varying and component-dependent. Here we use the mean
        # values over the current time step.
        φ_Vcell = ℝ[(boxsize/φ_gridsize)**3]
        Δt = ᔑdt['1']
        dependent = [
            # Particle components
            ('particles', [
                ᔑdt['a**(-3*w_eff-1)', component.name]*component.mass*ℝ[1/(Δt*φ_Vcell)]
                for component in suppliers]
            ),
            # Fluid components
            ('ϱ', [
                ᔑdt['a**(-3*w_eff-1)', component.name]*ℝ[1/Δt]
                for component in suppliers]
            ),
        ]
        # In the fluid description, the gravitational source term is
        # ∂ₜJⁱ = ⋯ -a**(-3*w_eff)*(ϱ + c⁻²𝒫)*∂ⁱφ
        # and so a**(-3*w_eff) should be integrated over the time step
        # to get ΔJⁱ. In the particle description, the gravitational
        # source term is
        # ∂ₜmomⁱ = -mass*∂ⁱφ.
        # In the general case of a changing mass, the current mass is
        # given by mass*a**(-3*w_eff), and so again, a**(-3*w_eff)
        # shoud be integrated over the time step
        # in order to obtain Δmomⁱ.
        ᔑdt_key = ('a**(-3*w_eff)', 'component')
        # Execute the gravitational particle-mesh interaction
        particle_mesh(
            receivers, suppliers, ᔑdt, ᔑdt_key, potential, potential_name, dependent,
        )
        if pm_potential == 'full':
            if printout:
                masterprint('done')
    elif method == 'pp':
        # The particle-particle method with Ewald-periodicity
        masterprint(
            'Executing',
            shortrange_progress_messages('gravity', method, receivers),
            '...',
        )
        component_component(
            receivers, suppliers, gravity_pairwise, ᔑdt,
            dependent=['pos'],
            affected=['mom'],
            deterministic=True,
            pairing_level='domain',
            interaction_name='gravity',
        )
        masterprint('done')
    elif method == 'ppnonperiodic':
        # The non-periodic particle-particle method
        if printout:
            masterprint(
                'Executing',
                shortrange_progress_messages('gravity', method, receivers),
                '...',
            )
        component_component(
            receivers, suppliers, gravity_pairwise_nonperiodic, ᔑdt,
            dependent=['pos'],
            affected=['mom'],
            deterministic=True,
            pairing_level='domain',
            interaction_name='gravity',
        )
        if printout:
            masterprint('done')
    elif master:
        abort(f'gravity() was called with the "{method}" method')

# Function that carry out the lapse interaction,
# correcting for the fact that the decay rate of species should be
# measured with respect to their individual proper time.
@cython.pheader(
    # Arguments
    method=str,
    receivers=list,
    suppliers=list,
    ᔑdt=dict,
    interaction_type=str,
    printout='bint',
    # Locals
    dependent=list,
    ᔑdt_key=object,  # str or tuple
)
def lapse(method, receivers, suppliers, ᔑdt, interaction_type, printout):
    # While the receivers list stores the correct components,
    # the suppliers store the lapse component as well as all the
    # components also present as receivers. As the lapse force should be
    # supplied solely from the lapse component, we must remove these
    # additional components.
    suppliers = oneway_force(receivers, suppliers)
    if len(suppliers) == 0:
        abort('The lapse() function got no suppliers, but expected a lapse component.')
    elif len(suppliers) > 1:
        abort(
            f'The lapse() function got the following suppliers: {suppliers}, '
            f'but expected only a lapse component.'
        )
    # For the lapse force, only the PM method is implemented
    if method == 'pm':
        if printout:
            masterprint(
                f'Executing lapse interaction for {receivers[0].name} via the PM method ...'
                if len(receivers) == 1 else (
                    'Executing lapse interaction for {{{}}} via the PM method ...'
                    .format(', '.join([component.name for component in receivers]))
                )
            )
        # As the lapse potential is implemented exactly analogous to the
        # gravitational potential, it obeys the Poisson equation
        # ∇²φ = 4πGa²ρ = 4πGa**(-3*w_eff - 1)ϱ,
        # with φ the lapse potential and ρ, ϱ and w_eff belonging to the
        # fictitious lapse species. The realized φ should take on values
        # corresponding to its mean over the time step,
        # weighted with a**(-3*w_eff - 1).
        dependent = [
            # The lapse component
            ('ϱ', [
                ᔑdt['a**(-3*w_eff-1)', component.name]/ᔑdt['1']
                for component in suppliers]
            ),
        ]
        # As the lapse potential is implemented exactly analogous to the
        # gravitational potential, the momentum updates are again
        # proportional to a**(-3*w_eff) integrated over the time step
        # (see the gravity function for a more detailed explanation).
        # The realized lapse potential is the common lapse potential,
        # indepedent on the component in question which is to receive
        # momentum updates. The actual lapse potential needed for a
        # given component is obtained by multiplying the common lapse
        # potential by Γ/H, where Γ is the decay rate of the component
        # and H is the Hubble parameter. As these are time dependent,
        # the full time step integral is then a**(-3*w_eff)*Γ/H.
        ᔑdt_key = ('a**(-3*w_eff)*Γ/H', 'component')
        # Execute the lapse particle-mesh interaction.
        # As the lapse potential is exactly analogous to the
        # gravitational potential, we may reuse the gravity_potential
        # function implementing the Poisson equation for gravity.
        particle_mesh(
            receivers, suppliers, ᔑdt, ᔑdt_key, gravity_potential, 'lapse potential', dependent,
        )
        if printout:
            masterprint('done')
    elif master:
        abort(f'lapse() was called with the "{method}" method')

# Function implementing progress messages used for the short-range
# kicks intertwined with drift operations.
@cython.pheader(
    # Arguments
    force=str,
    method=str,
    receivers=list,
    extra_message=str,
    # Locals
    component='Component',
    returns=str,
)
def shortrange_progress_messages(force, method, receivers, extra_message=' (short-range only)'):
    if force == 'gravity':
        if method == 'p3m':
            return (
                f'gravitational interaction for {receivers[0].name} via '
                f'the P³M method{extra_message}'
            ) if len(receivers) == 1 else (
                f'gravitational interaction for {{{{{{}}}}}} via the P³M method{extra_message}'
                .format(', '.join([component.name for component in receivers]))
            )
        elif method == 'pp':
            return (
                f'gravitational interaction for {receivers[0].name} via '
                f'the PP method'
            ) if len(receivers) == 1 else (
                'gravitational interaction for {{{}}} via the PP method'
                .format(', '.join([component.name for component in receivers]))
            )
        elif method == 'ppnonperiodic':
            return (
                f'gravitational interaction for {receivers[0].name} via '
                f'the non-periodic PP method'
            ) if len(receivers) == 1 else (
                'gravitational interaction for {{{}}} via the non-periodic PP method'
                .format(', '.join([component.name for component in receivers]))
            )
        else:
            abort(
                f'"{method}" is not a known method for '
                f'force "{force}" in shortrange_progress_messages()'
            )
    else:
        abort(f'Unknown force "{force}" supplied to shortrange_progress_messages()')

# Function that given lists of receiver and supplier components of a
# one-way interaction removes any components from the supplier list that
# are also present in the receiver list.
def oneway_force(receivers, suppliers):
    return [component for component in suppliers if component not in receivers]

# Function which constructs a list of interactions from a list of
# components. The list of interactions store information about which
# components interact with one another, via what force and method.
def find_interactions(components, interaction_type='any'):
    """You may specify an interaction_type to only get
    specific interactions. The options are:
    - interaction_type == 'any':
      Include every interaction.
    - interaction_type == 'long-range':
      Include long-range interactions only, i.e. ones with a method of
      either PM and P³M. Note that P³M interactions will also be
      returned for interaction_type == 'short-range'.
    - interaction_type == 'short-range':
      Include short-range interactions only, i.e. any other than PM.
      Note that P³M interactions will also be returned
      for interaction_type == 'short-range'.
    """
    # Use cached result
    interactions_list = interactions_lists.get(tuple(components + [interaction_type]))
    if interactions_list:
        return interactions_list
    # Find all (force, method) pairs in use. Store these as a (default)
    # dict mapping forces to lists of methods.
    forces_in_use = collections.defaultdict(set)
    for component in components:
        for force, method in component.forces.items():
            forces_in_use[force].add(method)
    # Check that all forces and methods assigned
    # to the components are implemented.
    for force, methods in forces_in_use.items():
        methods_implemented = forces_implemented.get(force, [])
        for method in methods:
            if not method:
                # When the method is set to an empty string it signifies
                # that this method should be used as a supplier for the
                # given force, but not receive the force itself.
                continue
            if method not in methods_implemented:
                abort(f'Method "{method}" for force "{force}" is not implemented')
    # Construct the interactions_list with (named) 4-tuples
    # in the format (force, method, receivers, suppliers),
    # where receivers is a list of all components which interact
    # via the force and should therefore receive momentum updates
    # computed via this force and the method given as the
    # second element. In the simple case where all components
    # interacting under some force using the same method, the suppliers
    # list holds the same components as the receivers list. When the
    # same force should be applied to several components using
    # different methods, the suppliers list still holds all components
    # as before, while the receivers list is limited to just those
    # components that should receive the force using the
    # specified method. Note that the receivers do not contribute to the
    # force unless they are also present in the suppliers list.
    interactions_list = []
    for force, methods in forces_implemented.items():
        for method in methods:
            if method not in forces_in_use.get(force, []):
                continue
            # Find all receiver and supplier components
            # for this (force, method) pair.
            receivers = []
            suppliers = []
            for component in components:
                if force in component.forces:
                    suppliers.append(component)
                    if component.forces[force] == method:
                        receivers.append(component)
            # Store the 4-tuple in the interactions_list
            interactions_list.append(Interaction(force, method, receivers, suppliers))
    # Cleanup the list of interactions
    def cleanup():
        nonlocal interactions_list
        # If fluid components are present as suppliers for interactions
        # using a method different from PM, remove them from the
        # suppliers list and create a new PM interaction instead.
        for i, interaction in enumerate(interactions_list):
            if interaction.method == 'pm':
                continue
            for component in interaction.suppliers:
                if component.representation == 'fluid':
                    interaction.suppliers.remove(component)
                    interactions_list.insert(
                        i + 1,
                        Interaction(
                            interaction.force, 'pm', interaction.receivers.copy(), [component],
                        )
                    )
                    return True
        # Remove interactions with no suppliers or no receivers
        interactions_list = [interaction for interaction in interactions_list
            if interaction.receivers and interaction.suppliers]
        # Merge interactions of identical force, method and receivers
        # but different suppliers, or identical force,
        # method and suppliers but different receivers.
        for     i, interaction_i in enumerate(interactions_list):
            for j, interaction_j in enumerate(interactions_list[i+1:], i+1):
                if interaction_i.force != interaction_j.force:
                    continue
                if interaction_i.method != interaction_j.method:
                    continue
                if (
                        set(interaction_i.receivers) == set(interaction_j.receivers)
                    and set(interaction_i.suppliers) != set(interaction_j.suppliers)
                ):
                    for supplier in interaction_j.suppliers:
                        if supplier not in interaction_i.suppliers:
                            interaction_i.suppliers.insert(0, supplier)
                    interactions_list.pop(j)
                    return True
                if (
                        set(interaction_i.receivers) != set(interaction_j.receivers)
                    and set(interaction_i.suppliers) == set(interaction_j.suppliers)
                ):
                    for receiver in interaction_j.receivers:
                        if receiver not in interaction_i.receivers:
                            interaction_i.receivers.insert(0, receiver)
                    interactions_list.pop(j)
                    return True
    while cleanup():
        pass
    # In the case that only some interactions should be considered,
    # remove the unwanted interactions.
    if 'long' in interaction_type:
        for interaction in interactions_list:
            if interaction.method not in {'pm', 'p3m'}:
                interaction.receivers[:] = []
        while cleanup():
            pass
    elif 'short' in interaction_type:
        for interaction in interactions_list:
            if interaction.method == 'pm':
                interaction.receivers[:] = []
        while cleanup():
            pass
    elif 'any' not in interaction_type:
        abort(f'find_interactions(): Unknown interaction_type "{interaction_type}"')
    # Cache the result and return it
    interactions_lists[tuple(components + [interaction_type])] = interactions_list
    return interactions_list
# Global dict of interaction lists populated by the above function
cython.declare(interactions_lists=dict)
interactions_lists = {}
# Create the Interaction type used in the above function
Interaction = collections.namedtuple(
    'Interaction', ('force', 'method', 'receivers', 'suppliers')
)

# Specification of implemented forces.
# The order specified here will be the order in which the forces
# are computed and applied.
# Importantly, all forces and methods should be written with purely
# alphanumeric, lowercase characters.
forces_implemented = {
    'gravity': ['ppnonperiodic', 'pp', 'p3m', 'pm'],
    'lapse'  : [                              'pm'],
}
