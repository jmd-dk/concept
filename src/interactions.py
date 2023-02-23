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
cimport(
    'from communication import     '
    '    communicate_ghosts,       '
    '    domain_subdivisions,      '
    '    get_buffer,               '
    '    rank_neighbouring_domain, '
    '    sendrecv_component,       '
)
cimport('from ewald import get_ewald_grid')
cimport(
    'from mesh import                         '
    '    copy_modes,                          '
    '    diff_domaingrid,                     '
    '    domain_decompose,                    '
    '    fft,                                 '
    '    fourier_loop,                        '
    '    fourier_operate,                     '
    '    get_fftw_slab,                       '
    '    interpolate_domaingrid_to_particles, '
    '    interpolate_upstream,                '
    '    nullify_modes,                       '
    '    slab_decompose,                      '
)
cimport(
    'from species import                        '
    '    accept_or_reject_subtiling_refinement, '
    '    tentatively_refine_subtiling,          '
)

# Pure Python imports
from mesh import group_components

# Function pointer types used in this module
pxd("""
ctypedef void (*func_interaction)(
    str,              # interaction_name
    Component,        # receiver
    Component,        # supplier
    dict,             # ᔑdt_rungs
    int,              # rank_supplier
    bint,             # only_supply
    str,              # pairing_level
    Py_ssize_t[::1],  # tile_indices_receiver
    Py_ssize_t**,     # tile_indices_supplier_paired
    Py_ssize_t*,      # tile_indices_supplier_paired_N
    dict,             # interaction_extra_args
)
""")



# Generic function implementing component-component pairing
@cython.header(
    # Arguments
    interaction_name=str,
    receivers=list,
    suppliers=list,
    interaction=func_interaction,
    ᔑdt_rungs=dict,
    pairing_level=str,
    interaction_extra_args=dict,
    # Locals
    affected=list,
    anticipate_refinement='bint',
    anticipation_period='Py_ssize_t',
    attempt_refinement='bint',
    computation_time='double',
    dependent=list,
    deterministic='bint',
    index='Py_ssize_t',
    judge_refinement='bint',
    judgement_period='Py_ssize_t',
    lowest_active_rung='signed char',
    only_supply='bint',
    pair=set,
    pairs=list,
    rank_other='int',
    receiver='Component',
    refinement_offset='Py_ssize_t',
    refinement_period='Py_ssize_t',
    rung_index='signed char',
    softening='double',
    subtiles_computation_times_N_interaction='Py_ssize_t[::1]',
    subtiles_computation_times_interaction='double[::1]',
    subtiles_computation_times_sq_interaction='double[::1]',
    subtiling='Tiling',
    subtiling_shape_judged='Py_ssize_t[::1]',
    subtiling_name=str,
    subtiling_name_2=str,
    supplier='Component',
    tile_sorted=set,
    tiling_name=str,
    returns='void',
)
def component_component(
    interaction_name, receivers, suppliers, interaction, ᔑdt_rungs,
    pairing_level, interaction_extra_args={},
):
    """This function takes care of pairings between all receiver and
    supplier components. It then calls domain_domain.
    """
    # Lookup basic information for this interaction
    interaction_info = interactions_registered[interaction_name]
    dependent     = interaction_info.dependent
    affected      = interaction_info.affected
    deterministic = interaction_info.deterministic
    # The names used to refer to the domain and tile level tiling
    # (tiles and subtiles). In the case of pairing_level == 'domain',
    # no actual tiling will take place, but we still need the
    # tile + subtile structure. For this, the trivial tiling,
    # spanning the box, is used.
    if 𝔹[pairing_level == 'tile']:
        tiling_name      = f'{interaction_name} (tiles)'
        subtiling_name   = f'{interaction_name} (subtiles)'
        subtiling_name_2 = f'{interaction_name} (subtiles 2)'
    else:  # pairing_level == 'domain':
        tiling_name = subtiling_name = 'trivial'
    # Set flags anticipate_refinement, attempt_refinement and
    # judge_refinement. The first signals whether a tentative subtiling
    # refinement attempt is coming up soon, in which case we should
    # be collecting computation time data of the current subtiling.
    # The second specifies whether a tentative refinement of the
    # subtilings in use should be performed now, meaning prior to
    # the interaction. The third specifies whether the previously
    # performed tentative refinement should be concluded, resulting in
    # either accepting or rejecting the refinement.
    anticipate_refinement = attempt_refinement = judge_refinement = False
    if 𝔹[pairing_level == 'tile'
        and shortrange_params[interaction_name]['subtiling'][0] == 'automatic'
    ]:
        # The anticipation_period and judgement_period specifies the
        # number of time steps spend collecting computation time data
        # before and after a tentative sutiling refinement.
        # The refinement will be judged after the first interaction of
        # the time step after judgement_period time steps has gone by
        # after the tentative refinement (there may be many more
        # interactions in this time step, depending on N_rungs).
        # Note that changes to anticipation_period or judgement_period
        # need to be reflected in the subtiling_refinement_period_min
        # variable, defined in the commons module. The relation is
        # subtiling_refinement_period_min = (
        #     anticipation_period + judgement_period + 1)
        anticipation_period = 4
        judgement_period = 2
        subtiles_computation_times_interaction    = subtiles_computation_times   [interaction_name]
        subtiles_computation_times_sq_interaction = subtiles_computation_times_sq[interaction_name]
        subtiles_computation_times_N_interaction  = subtiles_computation_times_N [interaction_name]
        for receiver in receivers:
            subtiling = receiver.tilings.get(subtiling_name)
            if subtiling is None:
                continue
            refinement_period = subtiling.refinement_period
            refinement_offset = subtiling.refinement_offset
            if refinement_period == 0:
                abort(
                    f'The subtiling "{subtiling_name}" is set to use automatic subtiling '
                    f'refinement, but it has a refinement period of {refinement_period}.'
                )
            # We judge the attempted refinement after 2 whole time steps
            # has gone by; this one and the next. The refinement will
            # then be judged after the first interaction on the third
            # time step (there may be many more interactions
            # if N_rungs > 1).
            if interaction_name in subtilings_under_tentative_refinement:
                anticipate_refinement = True
                judge_refinement = (
                    ℤ[universals.time_step + refinement_offset + 1] % refinement_period == 0
                )
            else:
                attempt_refinement = (
                    (ℤ[universals.time_step + refinement_offset + 1] + judgement_period
                        ) % refinement_period == 0
                )
                # We begin storing the computation time data of the
                # current subtiling 4 time steps before we tentatively
                # apply the new subtiling.
                anticipate_refinement = (
                    (ℤ[universals.time_step + refinement_offset + 1] + judgement_period
                        ) % refinement_period >= refinement_period - anticipation_period
                )
            break
    # Do the tentative subtiling refinement, if required
    if attempt_refinement:
        # Copy the old computation times to new locations in
        # subtiles_computation_times_interaction, making room for the
        # new computation times.
        for rung_index in range(N_rungs):
            subtiles_computation_times_interaction[N_rungs + rung_index] = (
                subtiles_computation_times_interaction[rung_index]
            )
            subtiles_computation_times_sq_interaction[N_rungs + rung_index] = (
                subtiles_computation_times_sq_interaction[rung_index]
            )
            subtiles_computation_times_N_interaction[N_rungs + rung_index] = (
                subtiles_computation_times_N_interaction[rung_index]
            )
            subtiles_computation_times_interaction   [rung_index] = 0
            subtiles_computation_times_sq_interaction[rung_index] = 0
            subtiles_computation_times_N_interaction [rung_index] = 0
        # Replace the subtilings with slightly refined versions
        subtilings_under_tentative_refinement.add(interaction_name)
        tentatively_refine_subtiling(interaction_name)
    # Pair each receiver with all suppliers and let them interact
    pairs = []
    tile_sorted = set()
    computation_time = 0  # Total tile-tile computation time for this call to component_component()
    for receiver in receivers:
        for supplier in suppliers:
            pair = {receiver, supplier}
            if pair in pairs:
                continue
            pairs.append(pair)
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
            # Pair up domains for the current
            # receiver and supplier component.
            domain_domain(
                interaction_name,
                receiver,
                supplier,
                interaction,
                ᔑdt_rungs,
                dependent,
                affected,
                only_supply,
                deterministic,
                pairing_level,
                interaction_extra_args,
            )
        # The interactions between the receiver and all suppliers are
        # now done. Add the accumulated computation time to the local
        # computation_time variable, then nullify the computation time
        # stored on the subtiling, so that it is ready for new data.
        # To keep the total computation time tallied up over the entire
        # time step present on the subtiling, add the currently stored
        # computation time to the computation_time_total attribute
        # before doing the nullification.
        subtiling = receiver.tilings[subtiling_name]
        computation_time += subtiling.computation_time
        subtiling.computation_time_total += subtiling.computation_time
        subtiling.computation_time = 0
    # All interactions are now done. If the measured computation time
    # should be used for automatic subtiling refinement, store this
    # outside of this function.
    if 𝔹[pairing_level == 'tile'
        and shortrange_params[interaction_name]['subtiling'][0] == 'automatic'
    ]:
        # The computation time depends drastically on which rungs are
        # currently active. We therefore store the total computation
        # time according to the current lowest active rung.
        if anticipate_refinement or attempt_refinement or judge_refinement:
            lowest_active_rung = ℤ[N_rungs - 1]
            for receiver in receivers:
                if receiver.lowest_active_rung < lowest_active_rung:
                    lowest_active_rung = receiver.lowest_active_rung
                    if lowest_active_rung == 0:
                        break
            subtiles_computation_times_interaction   [lowest_active_rung] += computation_time
            subtiles_computation_times_sq_interaction[lowest_active_rung] += computation_time**2
            subtiles_computation_times_N_interaction [lowest_active_rung] += 1
        # If it is time to judge a previously attempted refinement,
        # do so and reset the computation time.
        if judge_refinement:
            subtilings_under_tentative_refinement.remove(interaction_name)
            subtiling_shape_judged = accept_or_reject_subtiling_refinement(
                interaction_name,
                subtiles_computation_times_interaction,
                subtiles_computation_times_sq_interaction,
                subtiles_computation_times_N_interaction,
            )
            subtiles_computation_times_interaction   [:] = 0
            subtiles_computation_times_sq_interaction[:] = 0
            subtiles_computation_times_N_interaction [:] = 0
        else:
            subtiling_shape_judged = subtiling_shape_rejected
        # Gather information about the acceptance of the new subtiling
        # and print out any positive results.
        Gather(subtiling_shape_judged, subtiling_shapes_judged)
        if master:
            for rank_other in range(nprocs):
                index = 3*rank_other
                if subtiling_shapes_judged[index] == 0:
                    continue
                subtiling_shape_judged = subtiling_shapes_judged[index:index+3]
                masterprint(
                    f'Rank {rank_other}: Refined subtile decomposition ({interaction_name}):',
                    '×'.join(list(map(str, subtiling_shape_judged)))
                )
# Containers and array used by the component_component() function.
# The subtiles_computation_times and subtiles_computation_times_N are
# used to store total computation times and numbers for performed
# interactions. They are indexed as
# subtiles_computation_times[interaction_name][rung_index],
# resulting in the accumulated computation time for this interaction
# when the lowest active rung corresponds to rung_index.
# The subtilings_under_tentative_refinement set contain names of
# interactions the subtilings of which are currently under
# tentative refinement.
# The subtiling_shape_rejected are used only for signalling purposes,
# while the subtiling_shapes_judged are used to gather subtiling shapes
# from all processes into the master process.
cython.declare(
    subtiles_computation_times=object,
    subtiles_computation_times_sq=object,
    subtiles_computation_times_N=object,
    subtilings_under_tentative_refinement=set,
    subtiling_shape_rejected='Py_ssize_t[::1]',
    subtiling_shapes_judged='Py_ssize_t[::1]',
)
subtiles_computation_times = collections.defaultdict(
    lambda: zeros(ℤ[2*N_rungs], dtype=C2np['double'])
)
subtiles_computation_times_sq = collections.defaultdict(
    lambda: zeros(ℤ[2*N_rungs], dtype=C2np['double'])
)
subtiles_computation_times_N = collections.defaultdict(
    lambda: zeros(ℤ[2*N_rungs], dtype=C2np['Py_ssize_t'])
)
subtilings_under_tentative_refinement = set()
subtiling_shape_rejected = zeros(3, dtype=C2np['Py_ssize_t'])
subtiling_shapes_judged = empty(3*nprocs, dtype=C2np['Py_ssize_t']) if master else None

# Generic function implementing domain-domain pairing
@cython.header(
    # Arguments
    interaction_name=str,
    receiver='Component',
    supplier='Component',
    interaction=func_interaction,
    ᔑdt_rungs=dict,
    dependent=list,
    affected=list,
    only_supply='bint',
    deterministic='bint',
    pairing_level=str,
    interaction_extra_args=dict,
    # Locals
    domain_pair_nr='Py_ssize_t',
    instantaneous='bint',
    interact='bint',
    only_supply_communication='bint',
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
    tile_indices_supplier_paired='Py_ssize_t**',
    tile_indices_supplier_paired_N='Py_ssize_t*',
    tile_pairings_index='Py_ssize_t',
    returns='void',
)
def domain_domain(
    interaction_name, receiver, supplier, interaction, ᔑdt_rungs, dependent, affected,
    only_supply, deterministic, pairing_level, interaction_extra_args,
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
    """
    # To satisfy the compiler
    tile_indices_receiver = tile_indices_supplier = None
    tile_indices_supplier_paired = tile_indices_supplier_paired_N = NULL
    # Flag specifying whether or not this interaction is instantaneous.
    # For instantaneous interactions, we need to apply the updates to
    # the affected variables after each domain-domain pairing.
    instantaneous = interactions_registered[interaction_name].instantaneous
    # Get the process ranks to send to and receive from.
    # When only_supply is True, each domain will be paired with every
    # other domain, either in the entire box (pairing_level == 'domain')
    # or just among the neighbouring domains (pairing_level == 'tile').
    # When only_supply is False, the results of an interaction
    # computed on one process will be send back to the other
    # participating process and applied, cutting the number of domain
    # pairs roughly in half. Note however that even if only_supply is
    # False, we may not cut the number of domain pairs in half if the
    # receiver and supplier are separate components; all domains of
    # the receiver then need to be paired with all domains of the
    # supplier. That is, "only_supply" really serve two distinct usages:
    # (1) it is passed to the interaction() function so that
    # it knows whether to also update the supplier, (2) it determines
    # the interprocess communication pattern. As this latter usage also
    # depends upon whether the receiver and supplier is really the same
    # component, we extract usage (2) into its own flag,
    # "only_supply_communication".
    only_supply_communication = (only_supply if 𝔹[receiver is supplier] else True)
    ranks_send, ranks_recv = domain_domain_communication(pairing_level, only_supply_communication)
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
        # interaction function. It is important that the passed
        # interaction function do not update the affected variables
        # directly (e.g. mom for gravity), but instead update the
        # corresponding buffers (e.g. Δmom for gravity). The exception
        # is when the interaction is instantaneous, in which case the
        # affected variables should be updated directly, while also
        # updating the corresponding buffer for the supplier. The
        # buffers are what will be communicated. Also, Δmom is used to
        # figure out which short-range rung any given particle belongs
        # to. Special cases described below may change whether or not
        # the interaction between this particular domain pair should be
        # carried out on the local process (specified by the interact
        # flag), or whether the only_supply flag should be changed.
        interact = True
        only_supply = only_supply_passed
        with unswitch:
            if 𝔹[receiver is supplier] and 𝔹[pairing_level == 'domain']:
                if rank_send == rank_recv != rank:
                    # We are dealing with the special case where the
                    # local process and some other (with a rank given by
                    # rank_send == rank_recv) both send all of their
                    # particles belonging to the same component to each
                    # other, after which the exact same interaction
                    # takes place on both processes. In such a case,
                    # even when only_supply is False, there is no need
                    # to communicate the interaction results, as these
                    # are already known to both processes. Thus, we
                    # always use only_supply = True in such cases.
                    # Note that this is not true for
                    # pairing_level == 'tile', as here not all of the
                    # particles within the domains are communicated, but
                    # rather particles within completely disjoint sets
                    # of tiles, and so the interactions taking place on
                    # the two processes will not be identical.
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
                            only_supply = only_supply_passed
        # Communicate the dependent variables (e.g. pos for gravity) of
        # the supplier. For pairing_level == 'domain', communicate all
        # local particles. For pairing_level == 'tile', we only need to
        # communicate particles within the tiles that are going to
        # interact during the current domain-domain pairing.
        with unswitch:
            if 𝔹[pairing_level == 'tile']:
                # Find interacting tiles
                tile_indices = domain_domain_tile_indices(
                    interaction_name, receiver,
                    only_supply_communication, domain_pair_nr,
                )
                tile_indices_receiver = tile_indices[0, :]
                tile_indices_supplier = tile_indices[1, :]
            else:  # pairing_level == 'domain'
                # For domain level pairing we make use of
                # the trivial tiling, containing a single tile.
                tile_indices_receiver = tile_indices_supplier = tile_indices_trivial
                tile_indices_supplier_paired = tile_indices_trivial_paired
                tile_indices_supplier_paired_N = tile_indices_trivial_paired_N
        supplier_extrl = sendrecv_component(
            supplier_local, dependent, pairing_level, interaction_name, tile_indices_supplier,
            dest=rank_send, source=rank_recv,
        )
        # Let the local receiver interact with the external
        # supplier_extrl. This will update the affected variable buffers
        # (e.g. Δmom for gravity) of the local receiver, and of the
        # external supplier if only_supply is False.
        if interact:
            with unswitch:
                if 𝔹[pairing_level == 'tile']:
                    # Get the supplier tiles with which to pair each
                    # receiver tile and perform the interaction
                    # at the tile level.
                    tile_pairings_index = get_tile_pairings(
                        interaction_name,
                        receiver,
                        rank_recv,
                        only_supply_communication,
                        domain_pair_nr,
                        tile_indices_receiver,
                        tile_indices_supplier,
                    )
                    tile_indices_supplier_paired   = tile_pairings_cache  [tile_pairings_index]
                    tile_indices_supplier_paired_N = tile_pairings_N_cache[tile_pairings_index]
            # Perform the interaction
            interaction(
                𝕊[interaction_name if 𝔹[pairing_level == 'tile'] else 'trivial'],
                receiver,
                supplier_extrl,
                ᔑdt_rungs,
                rank_recv,
                only_supply,
                pairing_level,
                tile_indices_receiver,
                tile_indices_supplier_paired,
                tile_indices_supplier_paired_N,
                interaction_extra_args,
            )
        # Send the populated buffers (e.g. Δmom for gravity) back to the
        # process from which the external supplier_extrl came. Note that
        # we should not do this in the case of a local interaction
        # (rank_send == rank) or in a case where only_supply is True.
        if rank_send != rank and not only_supply:
            # For non-instantaneous interactions, the received Δ values
            # should be added to the Δ's of the local supplier_local.
            # For instantaneous interactions, the received Δ values
            # should be added directly to the data of the
            # local supplier_local.
            sendrecv_component(
                supplier_extrl, affected, pairing_level,
                interaction_name, tile_indices_supplier,
                dest=rank_recv, source=rank_send, component_recv=supplier_local,
                use_Δ_recv=(not instantaneous),
            )
            # Nullify the Δ buffers of the external supplier_extrl,
            # leaving this with no leftover junk.
            supplier_extrl.nullify_Δ(affected, only_active=False)
# Tile indices for the trivial tiling,
# used by the domain_domain function.
cython.declare(
    tile_indices_trivial='Py_ssize_t[::1]',
    tile_indices_trivial_paired='Py_ssize_t**',
    tile_indices_trivial_paired_N='Py_ssize_t*',
)
tile_indices_trivial = zeros(1, dtype=C2np['Py_ssize_t'])
tile_indices_trivial_paired = malloc(1*sizeof('Py_ssize_t*'))
tile_indices_trivial_paired[0] = cython.address(tile_indices_trivial[:])
tile_indices_trivial_paired_N = malloc(1*sizeof('Py_ssize_t'))
tile_indices_trivial_paired_N[0] = tile_indices_trivial.shape[0]

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
        ranks_send = empty(N_domain_pairs, dtype=C2np['int'])
        ranks_recv = empty(N_domain_pairs, dtype=C2np['int'])
        for i in range(N_domain_pairs):
            ranks_send[i] = mod(rank + i, nprocs)
            ranks_recv[i] = mod(rank - i, nprocs)
        domain_domain_communication_dict[pairing_level, only_supply] = (ranks_send, ranks_recv)
    elif pairing_level == 'tile':
        # When only_supply is True, each domain should be paired with
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
        offsets = asarray([0, 0, 0], dtype=C2np['int'])
        offsets_list.append(offsets.copy())
        ranks_send.append(rank_neighbouring_domain(*(+offsets)))
        ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
        # - Domains at the 6 faces
        #   (when only_supply is False, send right, forward, upward)
        direction = asarray([+1, 0, 0], dtype=C2np['int'])
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
        direction = asarray([+1, +1,  0], dtype=C2np['int'])
        flip      = asarray([-1, +1, +1], dtype=C2np['int'])
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
        offsets = asarray([+1, +1, +1], dtype=C2np['int'])
        offsets_list.append(offsets.copy())
        ranks_send.append(rank_neighbouring_domain(*(+offsets)))
        ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
        if only_supply:
            offsets_list.append(-offsets)
            ranks_send.append(rank_neighbouring_domain(*(-offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
        direction = asarray([+1, +1, -1], dtype=C2np['int'])
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
            (asarray(ranks_send, dtype=C2np['int']), asarray(ranks_recv, dtype=C2np['int']))
        )
        domain_domain_communication_dict[pairing_level, only_supply, 'domain_pair_offsets'] = (
            asarray(offsets_list, dtype=C2np['Py_ssize_t'])
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

# Function returning the indices of the tiles of the local receiver and
# supplier which take part in tile-tile interactions under the
# domain-domain pairing with number domain_pair_nr.
@cython.header(
    # Arguments
    interaction_name=str,
    component='Component',
    only_supply='bint',
    domain_pair_nr='Py_ssize_t',
    # Locals
    dim='int',
    domain_pair_offsets='Py_ssize_t[:, ::1]',
    domain_pair_offset='Py_ssize_t[::1]',
    key=tuple,
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
def domain_domain_tile_indices(interaction_name, component, only_supply, domain_pair_nr):
    key = (interaction_name, only_supply)
    tile_indices_all = domain_domain_tile_indices_dict.get(key)
    if tile_indices_all is None:
        tile_indices_all = [None]*27
        domain_domain_tile_indices_dict[key] = tile_indices_all
    else:
        tile_indices = tile_indices_all[domain_pair_nr]
        if tile_indices is not None:
            return tile_indices
    tile_layout_slice_start = empty(3, dtype=C2np['Py_ssize_t'])
    tile_layout_slice_end   = empty(3, dtype=C2np['Py_ssize_t'])
    domain_pair_offsets = domain_domain_communication_dict[
        'tile', only_supply, 'domain_pair_offsets']
    domain_pair_offset = domain_pair_offsets[domain_pair_nr, :]
    tiling_name = f'{interaction_name} (tiles)'
    tiling = component.tilings[tiling_name]
    tile_layout = tiling.layout
    tile_indices_list = []
    for sign in range(-1, 2, 2):
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

# Function that given arrays of receiver and supplier tiles
# returns them in paired format.
@cython.header(
    # Arguments
    interaction_name=str,
    component='Component',
    rank_supplier='int',
    only_supply='bint',
    domain_pair_nr='Py_ssize_t',
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier='Py_ssize_t[::1]',
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
    pairings='Py_ssize_t**',
    pairings_N='Py_ssize_t*',
    pairs_N='Py_ssize_t',
    suppliertile_indices_3D_global_to_1D_local=dict,
    tile_index3D='Py_ssize_t*',
    tile_index_3D_global_s=tuple,
    tile_index_r='Py_ssize_t',
    tile_index_s='Py_ssize_t',
    tile_indices_supplier_paired='Py_ssize_t[::1]',
    tile_indices_supplier_paired_ptr='Py_ssize_t*',
    tile_layout='Py_ssize_t[:, :, ::1]',
    tile_pairings_index='Py_ssize_t',
    tiling='Tiling',
    tiling_name=str,
    wraparound='bint',
    returns='Py_ssize_t',
)
def get_tile_pairings(
    interaction_name, component, rank_supplier, only_supply,
    domain_pair_nr, tile_indices_receiver, tile_indices_supplier,
):
    global tile_pairings_cache, tile_pairings_N_cache, tile_pairings_cache_size
    # Lookup index of the required tile pairings in the global cache
    key = (interaction_name, domain_pair_nr, only_supply)
    tile_pairings_index = tile_pairings_cache_indices.get(key, tile_pairings_cache_size)
    if tile_pairings_index < tile_pairings_cache_size:
        return tile_pairings_index
    # No cached results found. We will now compute the supplier tile
    # indices to be paired with each of the receiver tiles.
    # Below is a list of lists storing the supplier tile indices
    # for each receiver tile. The type of this data structure will
    # change during the computation.
    tile_indices_receiver_supplier = [[] for i in range(tile_indices_receiver.shape[0])]
    # Get the shape of the local (domain) tile layout,
    # as well as of the global (box) tile layout.
    tiling_name = f'{interaction_name} (tiles)'
    tiling = component.tilings[tiling_name]
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
                f'get_tile_pairings() got rank_supplier = {rank_supplier} != rank = {rank} '
                f'at domain_pair_nr == 0'
            )
        if not np.all(asarray(tile_indices_receiver) == asarray(tile_indices_supplier)):
            abort(
                f'get_tile_pairings() got tile_indices_receiver != tile_indices_supplier '
                f'at domain_pair_nr == 0'
            )
        i = 0
        for         l in range(ℤ[tile_layout.shape[0]]):
            for     m in range(ℤ[tile_layout.shape[1]]):
                for n in range(ℤ[tile_layout.shape[2]]):
                    if i != tile_layout[l, m, n]:
                        abort(
                            f'It looks as though the tile layout of {component.name} is incorrect'
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
                                # As domain_pair_nr == 0, all tiles in
                                # the local domain are paired with all
                                # others. To not double count, we
                                # disregard the pairing if the supplier
                                # tile index is lower than the receiver
                                # tile index (i). However, if
                                # only_supply is True, there is no
                                # double counting to be considered (the
                                # two components are presumably
                                # different), and so here we do not
                                # disregard the pairing.
                                with unswitch:
                                    if not only_supply:
                                        if tile_index_s < i:
                                            continue
                                neighbourtile_indices_supplier.append(tile_index_s)
                    tile_indices_receiver_supplier[i] = asarray(
                        neighbourtile_indices_supplier, dtype=C2np['Py_ssize_t'],
                    )
                    i += 1
    else:
        # Get relative offsets of the domains currently being paired
        domain_pair_offset = domain_domain_communication_dict[
            'tile', only_supply, 'domain_pair_offsets'][domain_pair_nr, :]
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
            tile_index3D = tiling.tile_index3D(tile_index_s)
            tile_index_3D_s = asarray([tile_index3D[dim] for dim in range(3)])
            tile_index_3D_global_s = tuple(tile_index_3D_s + tile_index_3D_s_start)
            suppliertile_indices_3D_global_to_1D_local[tile_index_3D_global_s] = tile_index_s
        # Pair each receiver tile with all neighbouring supplier tiles
        for i in range(tile_indices_receiver.shape[0]):
            neighbourtile_indices_supplier = tile_indices_receiver_supplier[i]
            # Construct global 3D index of this receiver tile
            tile_index_r = tile_indices_receiver[i]
            tile_index3D = tiling.tile_index3D(tile_index_r)
            tile_index_3D_r = asarray([tile_index3D[dim] for dim in range(3)])
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
                        # account, as such interactions are performed by
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
                            # pairing if the supplier tile index is
                            # lower than the receiver tile index.
                            # However, if only_supply is True, there is
                            # no double counting to be considered (the
                            # two components are presumably different),
                            # and so here we do not disregard
                            # the pairing.
                            with unswitch:
                                if domain_pair_nr == 0 and not only_supply:
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
    # the elements of which are arrays of dtype Py_ssize_t.
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
    # Cache the result. This cache is not actually used, but it ensures
    # that Python will not garbage collect the data.
    tile_indices_receiver_supplier_dict[key] = tile_indices_receiver_supplier
    # Now comes the caching that is actually used, where we use pointers
    # rather than Python objects.
    pairs_N = tile_indices_receiver_supplier.shape[0]
    pairings   = malloc(pairs_N*sizeof('Py_ssize_t*'))
    pairings_N = malloc(pairs_N*sizeof('Py_ssize_t'))
    for i in range(pairs_N):
        tile_indices_supplier_paired = tile_indices_receiver_supplier[i]
        tile_indices_supplier_paired_ptr = cython.address(tile_indices_supplier_paired[:])
        pairings[i] = tile_indices_supplier_paired_ptr
        pairings_N[i] = tile_indices_supplier_paired.shape[0]
    tile_pairings_cache_indices[key] = tile_pairings_index
    tile_pairings_cache_size += 1
    tile_pairings_cache = realloc(
        tile_pairings_cache,
        tile_pairings_cache_size*sizeof('Py_ssize_t**'),
    )
    tile_pairings_N_cache = realloc(
        tile_pairings_N_cache,
        tile_pairings_cache_size*sizeof('Py_ssize_t*'),
    )
    tile_pairings_cache  [tile_pairings_index] = pairings
    tile_pairings_N_cache[tile_pairings_index] = pairings_N
    return tile_pairings_index
# Caches used by the get_tile_pairings function
cython.declare(
    tile_indices_receiver_supplier_dict=dict,
    tile_pairings_cache_indices=dict,
    tile_pairings_cache_size='Py_ssize_t',
    tile_pairings_cache='Py_ssize_t***',
    tile_pairings_N_cache='Py_ssize_t**',
)
tile_indices_receiver_supplier_dict = {}
tile_pairings_cache_indices = {}
tile_pairings_cache_size = 0
tile_pairings_cache   = malloc(tile_pairings_cache_size*sizeof('Py_ssize_t**'))
tile_pairings_N_cache = malloc(tile_pairings_cache_size*sizeof('Py_ssize_t*'))

# Function responsible for constructing pairings between subtiles within
# the supplied subtiling, including the corresponding subtiles in the 26
# neighbour tiles. Subtiles further away than the supplied forcerange
# will not be paired.
@cython.header(
    # Arguments
    subtiling='Tiling',
    forcerange='double',
    only_supply='bint',
    # Locals
    all_pairings='Py_ssize_t***',
    all_pairings_N='Py_ssize_t**',
    dim='int',
    extent_over_range_dim='double',
    key=tuple,
    key_quick=tuple,
    pairing_index='Py_ssize_t',
    pairings='Py_ssize_t**',
    pairings_N='Py_ssize_t*',
    pairings_r='Py_ssize_t*',
    r_dim='Py_ssize_t',
    r2='double',
    same_tile='bint',
    shape='Py_ssize_t[::1]',
    size='Py_ssize_t',
    subtile_index_r='Py_ssize_t',
    subtile_index_s='Py_ssize_t',
    subtile_index3D='Py_ssize_t*',
    subtile_index3D_r='Py_ssize_t[::1]',
    subtile_index3D_s='Py_ssize_t*',
    subtile_pairings_index='Py_ssize_t',
    tile_extent='double[::1]',
    tile_pair_index='int',
    tiles_offset='Py_ssize_t[::1]',
    tiles_offset_i='Py_ssize_t',
    tiles_offset_j='Py_ssize_t',
    tiles_offset_k='Py_ssize_t',
    tiles_offset_ptr='Py_ssize_t*',
    returns='Py_ssize_t',
)
def get_subtile_pairings(subtiling, forcerange, only_supply):
    global subtile_pairings_cache, subtile_pairings_N_cache, subtile_pairings_cache_size
    # Lookup index of the required subtile pairings in the global cache.
    # We first try a quick lookup using a key containing the passed
    # subtiling instance. The attributes (e.g. shape and extent)
    # on a subtiling instance must then never be redefined.
    key_quick = (subtiling, forcerange, only_supply)
    subtile_pairings_index = subtile_pairings_cache_indices.get(
        key_quick,
        subtile_pairings_cache_size,
    )
    if subtile_pairings_index < subtile_pairings_cache_size:
        return subtile_pairings_index
    # The subtile pairings was not found in the cache. It is possible
    # that a different subtiling instance with the same shape and the
    # same extent in units of the forcerange is present in the cache.
    # All results are therefore also stored using keys containing the
    # shape and extent/forcerange. Try this more involved lookup.
    for dim in range(3):
        extent_over_range_dim = subtiling.extent[dim]*ℝ[1/forcerange]
        extent_over_range[dim] = float(f'{extent_over_range_dim:.12g}')
    shape = subtiling.shape
    key = (tuple(shape), tuple(extent_over_range), forcerange, only_supply)
    subtile_pairings_index = subtile_pairings_cache_indices.get(key, subtile_pairings_cache_size)
    if subtile_pairings_index < subtile_pairings_cache_size:
        # Found in cache. Add the missing, quick key.
        subtile_pairings_cache_indices[key_quick] = subtile_pairings_index
        return subtile_pairings_index
    # No cached results found. Create subtile pairings
    # for each of the 27 cases of neighbour tiles.
    size = subtiling.size
    tile_extent = subtiling.tile_extent
    all_pairings   = malloc(27*sizeof('Py_ssize_t**'))
    all_pairings_N = malloc(27*sizeof('Py_ssize_t*'))
    tiles_offset      = empty(3, dtype=C2np['Py_ssize_t'])
    subtile_index3D_r = empty(3, dtype=C2np['Py_ssize_t'])
    tiles_offset_ptr = cython.address(tiles_offset[:])
    same_tile = False
    for tiles_offset_i in range(-1, 2):
        tiles_offset_ptr[0] = tiles_offset_i
        for tiles_offset_j in range(-1, 2):
            tiles_offset_ptr[1] = tiles_offset_j
            for tiles_offset_k in range(-1, 2):
                tiles_offset_ptr[2] = tiles_offset_k
                # Does the tile offset correspond to
                # a tile being paired with itself?
                with unswitch:
                    if not only_supply:
                        same_tile = (tiles_offset_i == tiles_offset_j == tiles_offset_k == 0)
                # Get 1D tile pair index from the 3D offset
                tile_pair_index = get_neighbourtile_pair_index(
                    tiles_offset_i, tiles_offset_j, tiles_offset_k,
                )
                # Allocate memory for subtile pairings
                # for this particular tile pair.
                pairings   = malloc(size*sizeof('Py_ssize_t*'))
                pairings_N = malloc(size*sizeof('Py_ssize_t'))
                all_pairings  [tile_pair_index] = pairings
                all_pairings_N[tile_pair_index] = pairings_N
                # Loop over all receiver subtiles
                for subtile_index_r in range(size):
                    # Get 3D subtile index. As the tile_index3D() method
                    # return a view over internal data and we mutate
                    # subtile_index3D_r below, we take a copy of the
                    # returned data.
                    subtile_index3D = subtiling.tile_index3D(subtile_index_r)
                    for dim in range(3):
                        subtile_index3D_r[dim] = subtile_index3D[dim]
                    # The receiver and supplier subtiles belong to
                    # (potentially) different tiles, with a relative
                    # offset given by tiles_offset_*, so that the
                    # supplier tile is at the receiver tile location
                    # plus tiles_offset_*. We now subtract this offset
                    # from the receiver 3D subtile index, so that the
                    # difference in subtile indices between the receiver
                    # and supplier subtile is proportional to their
                    # physical separation. Note that subtile_index3D_r
                    # no longer represents the actual index in memory.
                    for dim in range(3):
                        subtile_index3D_r[dim] -= tiles_offset_ptr[dim]*shape[dim]
                    # Allocate memory for subtile pairings with this
                    # particular receiver subtile.
                    # We give it the maximum possible needed memory.
                    pairings_r = malloc(size*sizeof('Py_ssize_t'))
                    # Pair receiver subtile with every supplier subtile,
                    # unless the tile is being paired with itself.
                    # In that case, we need to not double count the
                    # subtile pairing (while still pairing every subtile
                    # with themselves).
                    pairing_index = 0
                    for subtile_index_s in range(subtile_index_r if same_tile else 0, size):
                        subtile_index3D_s = subtiling.tile_index3D(subtile_index_s)
                        # Measure (squared) distance between the subtile
                        # pair and reject if larger than the passed
                        # forcerange.
                        r2 = 0
                        for dim in range(3):
                            # Distance between the same point in the two
                            # subtiles along the dim'th dimension,
                            # in subtile grid units.
                            r_dim = abs(subtile_index3D_r[dim] - subtile_index3D_s[dim])
                            if r_dim > 0:
                                # The two subtiles are offset along the
                                # dim'th dimension. Subtract one unit
                                # from the length, making the length
                                # between the closest two points
                                # in the two subtiles.
                                r_dim -= 1
                            r2 += (r_dim*tile_extent[dim])**2
                        if r2 > ℝ[forcerange**2]:
                            continue
                        # Add this supplier subtile to the list of
                        # pairing partners for this receiver subtile.
                        pairings_r[pairing_index] = subtile_index_s
                        pairing_index += 1
                    # All pairs found for this receiver subtile.
                    # Truncate the allocated memory as to only contain
                    # the used chunk.
                    pairings_r = realloc(
                        pairings_r, pairing_index*sizeof('Py_ssize_t'),
                    )
                    pairings[subtile_index_r] = pairings_r
                    # Record the size of this pairing array
                    pairings_N[subtile_index_r] = pairing_index
    # Store results in global caches
    subtile_pairings_cache_indices[key_quick] = subtile_pairings_index
    subtile_pairings_cache_indices[key      ] = subtile_pairings_index
    subtile_pairings_cache_size += 1
    subtile_pairings_cache = realloc(
        subtile_pairings_cache, subtile_pairings_cache_size*sizeof('Py_ssize_t***'),
    )
    subtile_pairings_N_cache = realloc(
        subtile_pairings_N_cache, subtile_pairings_cache_size*sizeof('Py_ssize_t**'),
    )
    subtile_pairings_cache  [subtile_pairings_index] = all_pairings
    subtile_pairings_N_cache[subtile_pairings_index] = all_pairings_N
    # Return cached results in form of the cache index
    return subtile_pairings_index
# Caches used by the get_subtile_pairings function
cython.declare(
    extent_over_range='double[::1]',
    subtile_pairings_cache_indices=dict,
    subtile_pairings_cache_size='Py_ssize_t',
    subtile_pairings_cache='Py_ssize_t****',
    subtile_pairings_N_cache='Py_ssize_t***',
)
extent_over_range = empty(3, dtype=C2np['double'])
subtile_pairings_cache_indices = {}
subtile_pairings_cache_size = 0
subtile_pairings_cache   = malloc(subtile_pairings_cache_size*sizeof('Py_ssize_t***'))
subtile_pairings_N_cache = malloc(subtile_pairings_cache_size*sizeof('Py_ssize_t**'))

# Helper function for the get_subtile_pairings function
@cython.header(
    # Arguments
    tiles_offset_i='Py_ssize_t',
    tiles_offset_j='Py_ssize_t',
    tiles_offset_k='Py_ssize_t',
    # Locals
    i='int',
    j='int',
    k='int',
    returns='int',
)
def get_neighbourtile_pair_index(tiles_offset_i, tiles_offset_j, tiles_offset_k):
    # The passed tiles_offset is the relative offset between a pair of
    # neighbouring tiles, and so each of its three elements has to be
    # in {-1, 0, +1}. If any element is outside this range, it is due
    # to the periodic boundaries. Fix this now, as we do not care about
    # whether the tile pair is connected through the box boundary.
    # To avoid jumps we write this out arithmetically.
    i = (
        + (tiles_offset_i == 1) - (tiles_offset_i == -1)
        + (tiles_offset_i < -1) - (tiles_offset_i > 1)
    )
    j = (
        + (tiles_offset_j == 1) - (tiles_offset_j == -1)
        + (tiles_offset_j < -1) - (tiles_offset_j > 1)
    )
    k = (
        + (tiles_offset_k == 1) - (tiles_offset_k == -1)
        + (tiles_offset_k < -1) - (tiles_offset_k > 1)
    )
    # Compute 1D index from a 3×3×3 shape. We add 1 to each element,
    # as they range from -1 to +1. The index is then given by
    # ((i + 1)*3 + (j + 1))*3 + (k + 1), which we write out below.
    return i*9 + j*3 + k + 13

# Generic function implementing particle-particle pairing.
# Note that this function returns a generator and so should only be
# called within a loop.
@cython.iterator(
    depends=[
        # Global variables used by particle_particle()
        'periodic_offset',
        'tile_location_r',
        'tile_location_r_ptr',
        'tile_location_s',
        'tile_location_s_ptr',
        'tiles_offset',
        'tiles_offset_ptr',
        # Functions used by particle_particle()
        'get_subtile_pairings',
            # Global variables used by get_subtile_pairings()
            'extent_over_range',
            'subtile_pairings_cache_indices',
            'subtile_pairings_cache_size',
            'subtile_pairings_cache',
            'subtile_pairings_N_cache',
        'get_neighbourtile_pair_index',
    ]
)
def particle_particle(
    receiver, supplier, pairing_level,
    tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
    rank_supplier, interaction_name, only_supply, factors, forcerange=-1,
):
    # Cython declarations for variables used for the iteration,
    # not including those to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Keyword arguments
        forcerange='double',
        # Locals
        N_subtiles='Py_ssize_t',
        all_subtile_pairings='Py_ssize_t***',
        all_subtile_pairings_N='Py_ssize_t**',
        dim='int',
        highest_populated_rung_r='signed char',
        highest_populated_rung_s='signed char',
        indexᵖ_i='Py_ssize_t',
        local_interaction_flag_0='bint',
        local_interaction_flag_1='bint',
        local_interaction_flag_2='bint',
        lowest_active_rung_r='signed char',
        lowest_active_rung_s='signed char',
        lowest_populated_rung_r='signed char',
        lowest_populated_rung_s='signed char',
        only_supply_communication='bint',
        periodic_offset_ptr='double*',
        pos_r='double*',
        pos_s='double*',
        rung_particle_index_r='Py_ssize_t',
        rung_particle_index_s='Py_ssize_t',
        rung_particle_index_s_start='Py_ssize_t',
        rung_N_r='Py_ssize_t',
        rung_N_s='Py_ssize_t',
        rung_index_r='signed char',
        rung_index_r_bgn='signed char',
        rung_index_r_end='signed char',
        rung_index_s_bgn='signed char',
        rung_index_s_end='signed char',
        rung_indices_jumped_r='signed char*',
        rung_r='Py_ssize_t*',
        rung_s='Py_ssize_t*',
        rungs_N_r='Py_ssize_t*',
        rungs_N_s='Py_ssize_t*',
        subtile_contain_jumping_r='bint',
        subtile_contain_onlyinactive_r='bint',
        subtile_contain_particles_r='signed char',
        subtile_contain_particles_s='signed char',
        subtile_index_r='Py_ssize_t',
        subtile_index_s='Py_ssize_t',
        subtile_pairings='Py_ssize_t**',
        subtile_pairings_N='Py_ssize_t*',
        subtile_pairings_N_r='Py_ssize_t',
        subtile_pairings_index='Py_ssize_t',
        subtile_pairings_r='Py_ssize_t*',
        subtile_r='Py_ssize_t**',
        subtile_s='Py_ssize_t**',
        subtiles_contain_particles_r='signed char*',
        subtiles_contain_particles_s='signed char*',
        subtiles_r='Py_ssize_t***',
        subtiles_rungs_N_r='Py_ssize_t**',
        subtiles_rungs_N_s='Py_ssize_t**',
        subtiles_s='Py_ssize_t***',
        subtiling_name=str,
        subtiling_name_2=str,
        subtiling_s='Tiling',
        subtiling_s_2='Tiling',
        tile_contain_onlyinactive_r='bint',
        tile_contain_particles_r='signed char',
        tile_contain_particles_s='signed char',
        tile_extent='double*',
        tile_index_r='Py_ssize_t',
        tile_index_s='Py_ssize_t',
        tile_index3D_r='Py_ssize_t*',
        tile_index3D_s='Py_ssize_t*',
        tile_indices_supplier='Py_ssize_t*',
        tile_indices_supplier_N='Py_ssize_t',
        tile_location_s_dim='double',
        tile_pair_index='int',
        tile_separation='double',
        tiles_contain_particles_r='signed char*',
        tiles_contain_particles_s='signed char*',
        tiles_r='Py_ssize_t***',
        tiles_s='Py_ssize_t***',
        tiling_location_r='double*',
        tiling_location_s='double*',
        tiling_name=str,
        tiling_r='Tiling',
        tiling_s='Tiling',
        xi='double',
        yi='double',
        zi='double',
    )
    # Extract particle variables from the receiver component
    pos_r = receiver.pos
    lowest_active_rung_r     = receiver.lowest_active_rung
    lowest_populated_rung_r  = receiver.lowest_populated_rung
    highest_populated_rung_r = receiver.highest_populated_rung
    rung_indices_jumped_r = receiver.rung_indices_jumped
    # Extract particle variables from the supplier
    # (the external) component.
    pos_s = supplier.pos
    lowest_active_rung_s     = supplier.lowest_active_rung
    lowest_populated_rung_s  = supplier.lowest_populated_rung
    highest_populated_rung_s = supplier.highest_populated_rung
    # The names used to refer to the domain and tile level tiling
    # (tiles and subtiles). In the case of pairing_level == 'domain',
    # we always use the trivial tiling.
    if 𝔹[pairing_level == 'tile']:
        tiling_name    = f'{interaction_name} (tiles)'
        subtiling_name = f'{interaction_name} (subtiles)'
    else:  # pairing_level == 'domain':
        tiling_name = subtiling_name = 'trivial'
    # Extract tiling variables from receiver
    tiling_r = receiver.tilings[tiling_name]
    tiling_location_r         = cython.address(tiling_r.location[:])
    tile_extent               = cython.address(tiling_r.tile_extent[:])  # the same for receiver and supplier
    tiles_r                   = tiling_r.tiles
    tiles_contain_particles_r = tiling_r.contain_particles
    # Extract subtiling variables from receiver
    subtiling_r = receiver.tilings[subtiling_name]
    subtiles_r                   = subtiling_r.tiles
    subtiles_contain_particles_r = subtiling_r.contain_particles
    N_subtiles                   = subtiling_r.size  # The same for receiver and supplier
    # Extract tiling variables from supplier
    tiling_s = supplier.tilings[tiling_name]
    tiling_location_s         = cython.address(tiling_s.location[:])
    tiles_s                   = tiling_s.tiles
    tiles_contain_particles_s = tiling_s.contain_particles
    # Extract subtiling variables from supplier.
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
    # separate Tiling instance. If however the subtiling in use is the
    # trivial tiling, the re-sorting has no effect, and so we do not
    # have to worry.
    subtiling_s = supplier.tilings[subtiling_name]
    if 𝔹[receiver.name == supplier.name and rank == rank_supplier and subtiling_name != 'trivial']:
        subtiling_name_2 = f'{interaction_name} (subtiles 2)'
        if subtiling_name_2 not in supplier.tilings:
            supplier.tilings.pop(subtiling_name)
            subtiling_s_2 = supplier.init_tiling(subtiling_name)
            supplier.tilings[subtiling_name  ] = subtiling_s
            supplier.tilings[subtiling_name_2] = subtiling_s_2
        subtiling_s = supplier.tilings[subtiling_name_2]
    subtiles_s                   = subtiling_s.tiles
    subtiles_contain_particles_s = subtiling_s.contain_particles
    # Get subtile pairings between each
    # of the 27 possible tile pairings.
    only_supply_communication = (only_supply if receiver.name == supplier.name else True)
    if forcerange == -1:
        forcerange = get_shortrange_param((receiver, supplier), interaction_name, 'range')
    subtile_pairings_index = get_subtile_pairings(
        subtiling_r, forcerange, only_supply_communication,
    )
    all_subtile_pairings = subtile_pairings_cache[subtile_pairings_index]
    all_subtile_pairings_N = subtile_pairings_N_cache[subtile_pairings_index]
    # Range of receiver and supplier rungs.
    # Note that rung_index_s_bgn will be set later.
    if only_supply:
        rung_index_r_bgn = lowest_active_rung_r
    else:
        rung_index_r_bgn = lowest_populated_rung_r
    rung_index_r_end = highest_populated_rung_r + 1
    rung_index_s_end = highest_populated_rung_s + 1
    # Local pointer into the global array of particle position offsets
    # due to the periodicity.
    periodic_offset_ptr = cython.address(periodic_offset[:])
    # Some default values to yield
    rung_index_i = 0
    factor_i = 0
    # Flags specifying whether the force between particle i and j
    # should be applied to i and j. If only_supply is True,
    # the values below are correct. Otherwise, other values
    # will be set further down.
    apply_to_i = True
    apply_to_j = False
    # The current time. This is yielded back to the caller,
    # where time() - particle_particle_t_begin should be added to the
    # computation_time of the receiver subtiling. This is used for the
    # automatic subtiling refinement and the load imbalancing printout.
    particle_particle_t_begin = time()
    # Loop over the requested tiles in the receiver
    for tile_index_r in range(ℤ[tile_indices_receiver.shape[0]]):
        # Lookup supplier tile indices with which to pair the current
        # receiver tile.
        tile_indices_supplier   = tile_indices_supplier_paired  [tile_index_r]
        tile_indices_supplier_N = tile_indices_supplier_paired_N[tile_index_r]
        # Now make tile_index_r an actual receiver tile index
        tile_index_r = tile_indices_receiver[tile_index_r]
        # Skip tile if it does not contain any particles at all,
        # or only inactive particles when only_supply is True.
        tile_contain_particles_r = tiles_contain_particles_r[tile_index_r]
        with unswitch(1):
            if 𝔹[not only_supply]:
                if tile_contain_particles_r == 0:
                    continue
            else:
                if tile_contain_particles_r < 2:
                    continue
        tile_contain_onlyinactive_r = (tile_contain_particles_r == 1)
        # Sort particles within the receiver tile into subtiles
        tile_index3D_r = tiling_r.tile_index3D(tile_index_r)
        for dim in range(3):
            tile_location_r_ptr[dim] = (
                tiling_location_r[dim] + tile_index3D_r[dim]*tile_extent[dim]
            )
        subtiling_r.relocate(tile_location_r)
        subtiling_r.sort(tiling_r, tile_index_r)
        subtiles_rungs_N_r = subtiling_r.tiles_rungs_N
        # Loop over the requested tiles in the supplier
        for tile_index_s in range(tile_indices_supplier_N):
            tile_index_s = tile_indices_supplier[tile_index_s]
            # Skip tile if it does not contain any particles at all
            tile_contain_particles_s = tiles_contain_particles_s[tile_index_s]
            if tile_contain_particles_s == 0:
                continue
            # If both the receiver and supplier tile contains particles
            # on inactive rows only, we skip this tile pair.
            with unswitch(1):
                if tile_contain_onlyinactive_r:
                    if tile_contain_particles_s == 1:
                        continue
            # Sort particles within the supplier tile into subtiles
            tile_index3D_s = tiling_s.tile_index3D(tile_index_s)
            for dim in range(3):
                # While in this loop, also determine the tile offset
                tiles_offset_ptr[dim] = ℤ[tile_index3D_s[dim]] - tile_index3D_r[dim]
                # Set floating supplier tile location
                tile_location_s_dim = (
                    tiling_location_s[dim] + ℤ[tile_index3D_s[dim]]*tile_extent[dim]
                )
                tile_location_s_ptr[dim] = tile_location_s_dim
                # While in this loop, also determine
                # the periodic particle offset.
                tile_separation = tile_location_s_dim - tile_location_r_ptr[dim]
                if tile_separation > ℝ[0.5*boxsize]:
                    periodic_offset_ptr[dim] = boxsize
                elif tile_separation < ℝ[-0.5*boxsize]:
                    periodic_offset_ptr[dim] = ℝ[-boxsize]
                else:
                    periodic_offset_ptr[dim] = 0
            subtiling_s.relocate(tile_location_s)
            subtiling_s.sort(tiling_s, tile_index_s)
            subtiles_rungs_N_s = subtiling_s.tiles_rungs_N
            # Extract the values from periodic_offset_ptr
            periodic_offset_x = periodic_offset_ptr[0]
            periodic_offset_y = periodic_offset_ptr[1]
            periodic_offset_z = periodic_offset_ptr[2]
            # Get the needed subtile pairings for the selected receiver
            # and supplier tiles (which should be neighbour tiles).
            tile_pair_index = get_neighbourtile_pair_index(
                tiles_offset_ptr[0], tiles_offset_ptr[1], tiles_offset_ptr[2],
            )
            subtile_pairings   = all_subtile_pairings  [tile_pair_index]
            subtile_pairings_N = all_subtile_pairings_N[tile_pair_index]
            # Flag specifying whether this is a local interaction
            local_interaction_flag_0 = (
                𝔹[receiver.name == supplier.name and rank == rank_supplier]
                and (tile_index_r == tile_index_s)
            )
            # Loop over all subtiles in the selected receiver tile
            for subtile_index_r in range(N_subtiles):
                # Skip subtile if it does not contain
                # any particles at all, or only inactive particles
                # when only_supply is True.
                subtile_contain_particles_r = subtiles_contain_particles_r[subtile_index_r]
                with unswitch(3):
                    if 𝔹[not only_supply]:
                        if subtile_contain_particles_r == 0:
                            continue
                        subtile_contain_onlyinactive_r = (subtile_contain_particles_r == 1)
                    else:
                        if subtile_contain_particles_r < 2:
                            continue
                # Set and extract various receiver subtile variables
                subtile_contain_jumping_r = (subtile_contain_particles_r == 3)
                subtile_r            = subtiles_r        [subtile_index_r]
                rungs_N_r            = subtiles_rungs_N_r[subtile_index_r]
                subtile_pairings_r   = subtile_pairings  [subtile_index_r]
                subtile_pairings_N_r = subtile_pairings_N[subtile_index_r]
                # Loop over the needed supplier subtiles
                for subtile_index_s in range(subtile_pairings_N_r):
                    subtile_index_s = subtile_pairings_r[subtile_index_s]
                    # Skip subtile if it does not contain
                    # any particles at all.
                    subtile_contain_particles_s = subtiles_contain_particles_s[subtile_index_s]
                    if subtile_contain_particles_s == 0:
                        continue
                    # If both the receiver and supplier subtile contains
                    # particles on inactive rows only, we skip this
                    # subtile pair.
                    with unswitch(4):
                        if 𝔹[not only_supply]:
                            with unswitch(1):
                                if subtile_contain_onlyinactive_r:
                                    if subtile_contain_particles_s == 1:
                                        continue
                    # Set and extract various supplier subtile variables
                    subtile_contain_jumping_s = (subtile_contain_particles_s == 3)
                    subtile_s = subtiles_s        [subtile_index_s]
                    rungs_N_s = subtiles_rungs_N_s[subtile_index_s]
                    # Flag specifying whether this is a local interaction
                    local_interaction_flag_1 = (
                        local_interaction_flag_0
                        and (subtile_index_r == subtile_index_s)
                    )
                    # Loop over all rungs in the receiver subtile
                    for rung_index_r in range(rung_index_r_bgn, rung_index_r_end):
                        rung_N_r = rungs_N_r[rung_index_r]
                        if rung_N_r == 0:
                            continue
                        rung_r = subtile_r[rung_index_r]
                        # We need to pair all active receiver rungs
                        # with all supplier rungs. All inactive
                        # receiver rungs need only to be paired with
                        # the active supplier rungs (i.e. we do not need
                        # to pair up two inactive rungs).
                        # If only_supply is True, the values already set
                        # will be used.
                        rung_index_s_bgn = lowest_populated_rung_s
                        with unswitch(5):
                            if 𝔹[not only_supply]:
                                if rung_index_r < lowest_active_rung_r:
                                    # Only the supplier should receive
                                    # a kick.
                                    apply_to_i = False
                                    rung_index_s_bgn = lowest_active_rung_s
                                else:
                                    # The receiver and the supplier
                                    # should receive a kick.
                                    apply_to_i = True
                        # We need to make sure not to double count the
                        # rung pairs for local interactions. Here,
                        # local means that the current components,
                        # domains, tiles and subtiles for the receiver
                        # and supplier are all the same.
                        if local_interaction_flag_1 and (rung_index_s_bgn < rung_index_r):
                            rung_index_s_bgn = rung_index_r
                        # Loop over the needed supplier rungs
                        for rung_index_s in range(rung_index_s_bgn, rung_index_s_end):
                            rung_N_s = rungs_N_s[rung_index_s]
                            if rung_N_s == 0:
                                continue
                            rung_s = subtile_s[rung_index_s]
                            # Flag whether we need to apply the force to
                            # the supplier particles in this rung (if
                            # not, we still apply the force to the
                            # receiver particles).
                            with unswitch(6):
                                if 𝔹[not only_supply]:
                                    apply_to_j = (rung_index_s >= lowest_active_rung_s)
                            # Flag specifying whether this is a local interaction
                            local_interaction_flag_2 = (
                                local_interaction_flag_1
                                and (rung_index_r == rung_index_s)
                            )
                            # Loop over all particles
                            # in the receiver rung.
                            for rung_particle_index_r in range(rung_N_r):
                                # Get receiver particle index
                                indexᵖ_i = rung_r[rung_particle_index_r]
                                # Get the jumped rung index for
                                # particle i, which is given by
                                # rung_indices_jumped_r[i]. In the
                                # common case of no jumping, this
                                # equals the non-jumped rung index,
                                # which is given by rung_index_r.
                                # In an attempt to avoid the lookup,
                                # we test whether the current receiver
                                # subtile contains any particles that
                                # jumps and get the jumped/non-jumped
                                # index accordingly.
                                with unswitch(2):
                                    if apply_to_i:
                                        with unswitch(4):
                                            if subtile_contain_jumping_r:
                                                rung_index_i = rung_indices_jumped_r[indexᵖ_i]
                                            else:
                                                rung_index_i = rung_index_r
                                        # Fetch the corresponding factor
                                        factor_i = factors[rung_index_i]
                                # Get coordinates of receiver particle
                                indexˣ_i = 3*indexᵖ_i
                                xi = pos_r[indexˣ_i + 0]
                                yi = pos_r[indexˣ_i + 1]
                                zi = pos_r[indexˣ_i + 2]
                                # We need to make sure not to double
                                # count the particle pairs for local
                                # interactions. Here, local means that
                                # the current components, domains,
                                # tiles, subtiles and rungs for the
                                # receiver and supplier are all
                                # the same.
                                rung_particle_index_s_start = (
                                    local_interaction_flag_2*(rung_particle_index_r + 1)
                                )
                                # Loop over the needed particles
                                # in the supplier rung.
                                for rung_particle_index_s in range(
                                    rung_particle_index_s_start, rung_N_s,
                                ):
                                    # Get supplier particle index
                                    indexᵖ_j = rung_s[rung_particle_index_s]
                                    # "Vector" from particle j
                                    # to particle i.
                                    indexˣ_j = 3*indexᵖ_j
                                    x_ji = xi - pos_s[indexˣ_j + 0]
                                    y_ji = yi - pos_s[indexˣ_j + 1]
                                    z_ji = zi - pos_s[indexˣ_j + 2]
                                    # Yield the needed variables
                                    yield indexˣ_i, indexᵖ_j, indexˣ_j, rung_index_i, rung_index_s, x_ji, y_ji, z_ji, periodic_offset_x, periodic_offset_y, periodic_offset_z, apply_to_i, apply_to_j, factor_i, subtile_contain_jumping_s, particle_particle_t_begin, subtiling_r
# Variables used by the particle_particle() function
cython.declare(
    periodic_offset='double[::1]',
    tile_location_r='double[::1]',
    tile_location_r_ptr='double*',
    tile_location_s='double[::1]',
    tile_location_s_ptr='double*',
    tiles_offset='Py_ssize_t[::1]',
    tiles_offset_ptr='Py_ssize_t*',
)
periodic_offset = empty(3, dtype=C2np['double'])
tile_location_r = empty(3, dtype=C2np['double'])
tile_location_s = empty(3, dtype=C2np['double'])
tiles_offset    = empty(3, dtype=C2np['Py_ssize_t'])
tile_location_r_ptr = cython.address(tile_location_r[:])
tile_location_s_ptr = cython.address(tile_location_s[:])
tiles_offset_ptr    = cython.address(tiles_offset[:])

# Function for converting a pair of softening lengths
# into a single softening length.
@cython.header(
    # Arguments
    ϵᵢ='double',
    ϵⱼ='double',
    # Locals
    ϵ='double',
    returns='double',
)
def combine_softening_lengths(ϵᵢ, ϵⱼ):
    # Combining softening lengths may be done in several
    # different ways, e.g.
    #   ϵ = sqrt(ϵᵢ² + ϵⱼ²)
    #   ϵ = (ϵᵢ + ϵⱼ)/2
    #   ϵ = min(ϵᵢ, ϵⱼ)
    #   ϵ = max(ϵᵢ, ϵⱼ)
    # Here we settle for the arithmetic mean.
    # This has been used by e.g. Hernquist & Barnes 1990.
    ϵ = 0.5*(ϵᵢ + ϵⱼ)
    return ϵ

# Function for computing the softened r⁻³.
# Instead of calling this function many times for some fixed
# softening length, consider tabulating its results.
@cython.header(
    # Arguments
    r2='double',
    ϵ='double',
    # Locals
    h='double',
    r='double',
    r2_softened='double',
    r3='double',
    u='double',
    returns='double',
)
def get_softened_r3inv(r2, ϵ):
    # The ϵ argument is the Plummer softening length,
    # regardless of which softening kernel is used (set by the
    # softening_kernel parameter). Each implemented softening kernel is
    # described in the code below. What is returned is always a
    # softened version of r⁻³, intended for replacement in the
    # gravitational force F⃗(r⃗) ∝ r⃗ r⁻³.
    #
    # References:
    #   [1] https://arxiv.org/abs/astro-ph/0011568
    if 𝔹[softening_kernel == 'none']:
        # Do not soften the force at all.
        # In terms of a kernel, this corresponds to
        #      W(r) = δ³(r)
        #   ⟹  F⃗(r⃗) ∝ r⃗ r⁻³.
        # Using this in simulations with any appreciable clustering
        # will lead to generation of large amounts of spurious energy.
        # To at least remove the divergence at r = 0,
        # we return 0 here.
        r3 = r2*sqrt(r2)
        return (0 if r3 == 0 else 1/r3)
    elif 𝔹[softening_kernel == 'plummer']:
        # This is the simplest softening kernel. As it has non-compact
        # support it softens the force at all scales.
        #      W(r) = 3/(4πϵ³) [1 + (r/ϵ)²]⁻⁵ᐟ²
        #   ⟹  F⃗(r⃗) ∝ r⃗ (r² + ϵ²)⁻³ᐟ².
        r2_softened = r2 + ϵ**2
        return 1/(r2_softened*sqrt(r2_softened))
    elif 𝔹[softening_kernel == 'spline']:
        # This is the cubic spline kernel of
        # Monaghan & Lattanzio (1985), often used in SPH.
        # It is the gravitational softening used in GADGET-2.
        #                     ⎧ 1 - 6(r/h)² + 6(r/h)³    0 ≤ r < h/2
        #      W(r) = 8/(πh³) ⎨ 2(1 - r/h)³            h/2 ≤ r < h
        #                     ⎩ 0                        h ≤ r
        #               ⎧ 32/h³ [1/3 - 6/5(r/h)² + (r/h)³]                                 0 ≤ r < h/2
        #   ⟹  F⃗(r⃗) ∝ r⃗ ⎨ 32/h³ [2/3 - 3/2(r/h) + 6/5(r/h)² - 1/3(r/h)³ - 1/480(r/h)⁻³]  h/2 ≤ r < h
        #               ⎩ r⁻³                                                              h ≤ r,
        # where h is the spline softening length "equivalent" to the
        # Plummer softening length ϵ. If we require F⃗(ϵ) for the
        # spline softening to equal F⃗(ϵ) for the Plummer softening,
        # we get h = 2.7116122709425334ϵ. GADGET-2 uses h = 2.8ϵ.
        # We choose to follow GADGET-2.
        h = 2.8*ϵ
        r = sqrt(r2)
        if r >= h:
            return 1/(r2*r)
        u = r/h
        if u < 0.5:
            return 32/h**3*(1./3. + u**2*(-6./5. + u))
        return 32/(3*r**3)*(u**3*(2 + u*(-9./2. + u*(18./5. - u))) - 3./480.)
    elif 𝔹[softening_kernel == 'epanechnikov']:
        # This kernel is described in e.g. [1] (where it is called F₁),
        # where it is shown to be superior to the cubic spline kernel
        # as it features a smaller mean integrated
        # squared force error.
        abort('Softening kernel "Epanechnikov" not yet implemented')
    elif 𝔹[softening_kernel == 'compensate']:
        # This is the kernel described in [1] under the name K₁.
        # It is reminiscent of the Epanechnikov kernel,
        # but compensates for the softened force at small r by
        # overestimating the force around the softening length,
        # supposedly leading to an even smaller mean integrated
        # squared force error.
        abort('Softening kernel "compensate" not yet implemented')
    # The specified softening kernel is not implemented
    abort(f'Softening kernel "{softening_kernel}" not understood')
    return 0  # To satisfy the compiler

# Generic function implementing particle-mesh interactions
# for both particle and fluid components.
@cython.header(
    # Arguments
    receivers=list,
    suppliers=list,
    gridsize_global='Py_ssize_t',
    quantity=str,
    force=str,
    method=str,
    potential=str,
    deconvolve_downstream='bint',
    deconvolve_upstream='bint',
    interpolation_order='int',
    interlace='bint',
    ᔑdt=dict,
    ᔑdt_key=object,  # str or tuple
    # Locals
    all_receiver_downstream_gridsizes_equal_global='bint',
    all_supplier_upstream_gridsizes_equal_global='bint',
    at_last_differentiation_order='bint',
    at_last_representation='bint',
    deconv_order_downstream='int',
    deconv_order_global='int',
    differentiation_order='int',
    dim='int',
    downstream_description=str,
    downstream_description_gridsize=str,
    downstream_description_representation=str,
    factor='double',
    fourier_diff='bint',
    grid_downstream='double[:, :, ::1]',
    gridsize_downstream='Py_ssize_t',
    group=dict,
    groups=dict,
    index='Py_ssize_t',
    k2='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    mutate_slab_downstream_ok='bint',
    nullification=str,
    only_particle_receivers='bint',
    only_particle_suppliers='bint',
    receiver='Component',
    receivers_differentiations=list,
    receivers_gridsizes_downstream=list,
    representation=str,
    slab_downstream='double[:, :, ::1]',
    slab_downstream_representation='double[:, :, ::1]',
    slab_global='double[:, :, ::1]',
    slab_global_ptr='double*',
    subgroup=list,
    subgroups=dict,
    supplier='Component',
    suppliers_description=str,
    suppliers_gridsizes_upstream=list,
    Δx='double',
    θ='double',
    ᐁᵢgrid_downstream='double[:, :, ::1]',
    returns='void',
)
def particle_mesh(
    receivers, suppliers, gridsize_global, quantity, force, method, potential, interpolation_order,
    deconvolve_upstream, deconvolve_downstream, interlace,
    ᔑdt, ᔑdt_key,
):
    """
    This function will update the momenta of all receiver components due
    to an interaction with the supplier components. The steps taken are
    outlined below.
    - Interpolate the specified quantity of all supplier components
      onto upstream (component specific) grids.
    - Transform to upstream Fourier slabs.
    - Perform deconvolution and interlacing of upstream Fourier slabs of
      particle suppliers, if specified.
    - Add upstream Fourier slabs together, producing global Fourier
      slabs.
    - Convert the values of the global Fourier slabs to potential
      values, with the formula to use given by the 'potential' argument.
    - Obtain downstream potential Fourier slabs for each receiver.
      For particle receivers, another deconvolution is now performed due
      to the upcoming interpolation.
    - For receivers obtaining the force grid from the
      real-space potential:
      - Transform to real space downstream potentials.
      - For each dimension, differentiate the downstream potentials to
        get the real space downstream force.
    - For receivers obtaining the force grid from the
      Fourier-space potential:
      - For each dimension, differentiate the downstream potentials in
        Fourier space to obtain the downstream force grid
        in Fourier space.
      - Transform to real space downstream force grid.
    - Interpolate the real-space downstream force grid onto
      the receivers, applying the force.
      The force application uses the prescription
        Δmom = -component.mass*∂ⁱφ*ᔑdt[ᔑdt_key].
    """
    if potential not in {'gravity', 'gravity long-range'}:
        abort(
            f'particle_mesh() got potential "{potential}" ∉ {{"gravity", "gravity long-range"}}'
        )
    suppliers_gridsizes_upstream = [
        supplier.potential_gridsizes[force][method].upstream
        for supplier in suppliers
    ]
    receivers_gridsizes_downstream = [
        receiver.potential_gridsizes[force][method].downstream
        for receiver in receivers
    ]
    suppliers_description = ', '.join([
        supplier.name + (
            f' ({supplier_gridsize_upstream})'
            if 𝔹[np.any(asarray(suppliers_gridsizes_upstream) != gridsize_global)] else ''
        )
        for supplier, supplier_gridsize_upstream in zip(suppliers, suppliers_gridsizes_upstream)
    ])
    if len(suppliers) > 1:
        suppliers_description = f'{{{suppliers_description}}}'
    masterprint(
        f'Constructing potential of grid size {gridsize_global} due to {suppliers_description} ...'
    )
    # If we only have particle suppliers (receivers) and all
    # upstream (downstream) grid sizes equal the global grid size,
    # we perform the upstream (downstream) deconvolution right within
    # this function while constructing the global potential,
    # rather than through calls to interpolate_upstream()
    # or fourier_operate().
    only_particle_suppliers = all([
        supplier.representation == 'particles'
        for supplier in suppliers
    ])
    only_particle_receivers = all([
        receiver.representation == 'particles'
        for receiver in receivers
    ])
    all_supplier_upstream_gridsizes_equal_global = np.all(
        asarray(suppliers_gridsizes_upstream) == gridsize_global
    )
    all_receiver_downstream_gridsizes_equal_global = np.all(
        asarray(receivers_gridsizes_downstream) == gridsize_global
    )
    deconv_order_global = 0
    if deconvolve_upstream:
        if only_particle_suppliers and all_supplier_upstream_gridsizes_equal_global:
            # Promote upstream deconvolution to global deconvolution
            deconvolve_upstream = False
            deconv_order_global += 1
    if deconvolve_downstream:
        if only_particle_receivers and all_receiver_downstream_gridsizes_equal_global:
            # Promote downstream deconvolution to global deconvolution
            deconvolve_downstream = False
            deconv_order_global += 1
    deconv_order_global *= interpolation_order
    # Interpolate suppliers onto global Fourier slabs by first
    # interpolating them onto individual upstream grids, transforming to
    # Fourier space and then adding them together.
    slab_global = interpolate_upstream(
        suppliers, suppliers_gridsizes_upstream, gridsize_global, quantity, interpolation_order,
        ᔑdt, deconvolve_upstream, interlace, output_space='Fourier',
    )
    slab_global_ptr = cython.address(slab_global[:, :, :])
    # Convert slab_global values to potential
    # and possibly perform upstream and/or downstream deconvolutions.
    for index, ki, kj, kk, factor, θ in fourier_loop(gridsize_global,
        skip_origin=True, deconv_order=deconv_order_global,
    ):
        k2 = ℤ[ℤ[ℤ[kj**2] + ki**2] + kk**2]
        # The potential factor, for converting the slab values
        # to the desired potential.
        with unswitch(5):
            # The physical squared length of the wave
            # vector is given by |k|² = (2π/boxsize)**2*k2.
            if 𝔹[potential == 'gravity']:
                # The Poisson equation, the factor of which is
                #   -4πG/|k|².
                factor *= ℝ[-boxsize**2*G_Newton/π]/k2
            else:  # potential == 'gravity long-range'
                # The Poisson equation with a Gaussian
                # cutoff, resulting in the factor
                #   -4πG/|k|² * exp(-rₛ²*|k|²).
                factor *= (
                    ℝ[-boxsize**2*G_Newton/π]/k2
                    *exp(k2*ℝ[-(2*π/boxsize*shortrange_params['gravity']['scale'])**2])
                )
        # Apply factor from deconvolution and potential
        slab_global_ptr[index    ] *= factor  # real part
        slab_global_ptr[index + 1] *= factor  # imag part
    # Ensure nullified origin
    nullify_modes(slab_global, 'origin')
    masterprint('done')
    # Group receivers according to their downstream grid size.
    # The order does not matter, except that we want the group with the
    # downstream grid size equal to the global grid size to be the last,
    # if indeed such a group exist.
    groups = group_components(receivers, receivers_gridsizes_downstream, [..., gridsize_global])
    # For each group, obtain downstream potential, compute downstream
    # forces and apply these to the receivers within the group.
    for gridsize_downstream, group in groups.items():
        downstream_description_gridsize = (
            str(gridsize_downstream)
            if not all_receiver_downstream_gridsizes_equal_global else ''
        )
        # Physical grid spacing of downstream potential grid
        Δx = boxsize/gridsize_downstream
        # Obtain downstream slab potential from global slab potential
        if gridsize_downstream == gridsize_global:
            # The downstream and global grid sizes are the same, so we
            # use the global slab as the downstream slab. The global
            # slab is then mutated by the code below (e.g. by the
            # in-place FFT), but as we are guaranteed that this is the
            # last downstream grid/group, this is OK.
            slab_downstream = slab_global
        else:
            # Fetch a slab to be used for the downstream potential.
            # As this is constructed through down- or up-scaling of the
            # global potential, we need to nullify all elements not set
            # by the down-/up-scaling. We could of course just always
            # perform a complete nullification.
            if gridsize_downstream < gridsize_global:
                # Downscaling. All elements of slab_downstream will be
                # set except the Nyquist planes, which then have to be
                # nullified.
                nullification = 'nyquist'
            else:
                # Upscaling. Only elements within a cube centred at
                # the origin and of size gridsize_global will be set.
                # We then need to nullify all elements beyond this cube.
                nullification = f'beyond cube of |k| < {gridsize_global//2}'
            slab_downstream = get_fftw_slab(
                gridsize_downstream, 'slab_updownstream', nullification,
            )
            # Obtain downstream slab potential through up-/down-scaling
            copy_modes(slab_global, slab_downstream, operation='=')
        # Handle each (possible) representation of the receivers in turn
        for representation in ('fluid', 'particles'):
            if representation not in group:
                continue
            at_last_representation = (
                𝔹[representation == 'particles'] or 'particles' not in group
            )
            downstream_description_representation = (
                representation
                if len(group) == 2 else ''
            )
            downstream_description = ', '.join([
                description
                for description in (
                    downstream_description_gridsize,
                    downstream_description_representation,
                )
                if description
            ])
            if downstream_description:
                downstream_description = f'({downstream_description}) '
            # The downstream deconvolution order, taking the
            # representation into account.
            deconv_order_downstream = interpolation_order*(
                𝔹[representation == 'particles'] and deconvolve_downstream
            )
            # Further group the receivers of the current representation
            # within the current group into subgroups based on their
            # potential differentiation order.
            receivers_differentiations = [
                receiver.potential_differentiations[force][method]
                for receiver in group[representation]
            ]
            subgroups = group_components(
                group[representation],
                receivers_differentiations,
                sorted(receivers_differentiations, reverse=True),
                split_representations=False,
            )
            for differentiation_order, subgroup in subgroups.items():
                at_last_differentiation_order = (
                    differentiation_order == ℤ[np.min(receivers_differentiations)]
                )
                # A differentiation order of 0
                # signals differentiation in Fourier space.
                fourier_diff = (differentiation_order == 0)
                # The downstream potential for all receivers within the
                # group (and thus subgroup) is stored as
                # slab_downstream. We want to use this directly if
                # possible, but if we need to mutate it and we are not
                # at the last subgroup, we need to take a copy.
                mutate_slab_downstream_ok = False
                if fourier_diff:
                    if at_last_representation and at_last_differentiation_order:
                        # Really only True for the last dim
                        mutate_slab_downstream_ok = True
                else:
                    if at_last_representation and at_last_differentiation_order:
                        mutate_slab_downstream_ok = True
                    elif deconv_order_downstream == 0:
                        # Mutation really not OK, but as the slab will
                        # not be differentiated nor deconvolved, there
                        # is no reason to take a copy.
                        mutate_slab_downstream_ok = True
                # Obtain the force grid either in Fourier or real space
                if fourier_diff:
                    # Fourier space differentiation.
                    # For each dimension, differentiate the grid
                    # to obtain the force and apply this force.
                    for dim in range(3):
                        masterprint(f'Obtaining and applying the {"xyz"[dim]}-force ...')
                        # Get reference to or copy of slab_downstream
                        if mutate_slab_downstream_ok and dim == 2:
                            slab_downstream_subgroup = slab_downstream
                        else:
                            slab_downstream_subgroup = get_fftw_slab(
                                gridsize_downstream, 'slab_updownstream_subgroup',
                            )
                            slab_downstream_subgroup[...] = slab_downstream
                        # Do the in-place Fourier differentiation along
                        # with the possible downstream deconvolution.
                        fourier_operate(
                            slab_downstream_subgroup,
                            deconv_order_downstream,
                            diff_dim=dim,
                        )
                        # Transform to real space
                        # and perform domain decomposition.
                        masterprint(
                            f'Transforming to real space force {downstream_description}...'
                        )
                        fft(slab_downstream_subgroup, 'backward')
                        grid_downstream = domain_decompose(
                            slab_downstream_subgroup,
                            'grid_updownstream',
                            do_ghost_communication=True,
                        )
                        masterprint('done')
                        # Apply force
                        apply_particle_mesh_force(
                            grid_downstream, dim, group[representation], interpolation_order,
                            ᔑdt, ᔑdt_key,
                        )
                        masterprint('done')
                else:
                    # Real space differentiation.
                    # Get reference to or copy of slab_downstream.
                    if mutate_slab_downstream_ok:
                        slab_downstream_subgroup = slab_downstream
                    else:
                        slab_downstream_subgroup = get_fftw_slab(
                            gridsize_downstream, 'slab_updownstream_subgroup',
                        )
                        slab_downstream_subgroup[...] = slab_downstream
                    # Perform possible downstream deconvolution.
                    fourier_operate(
                        slab_downstream_subgroup,
                        deconv_order=deconv_order_downstream,
                    )
                    # Transform to real space
                    # and perform domain decomposition.
                    masterprint(
                        f'Transforming to real space potential {downstream_description}...'
                    )
                    fft(slab_downstream_subgroup, 'backward')
                    grid_downstream = domain_decompose(
                        slab_downstream_subgroup,
                        'grid_updownstream',
                        do_ghost_communication=True,
                    )
                    masterprint('done')
                    # For each dimension, differentiate the grid
                    # to obtain the force and apply this force.
                    for dim in range(3):
                        masterprint(f'Obtaining and applying the {"xyz"[dim]}-force ...')
                        # Differentiate the downstream potential in real
                        # space using finite difference. We need to
                        # properly populate the ghost points in the
                        # differentiated grid, as ghost points are
                        # needed for particle interpolation. For fluids,
                        # having proper ghost points in the
                        # differentiated grid means that the momentum
                        # grid will automatically get ghost points
                        # populated correctly as well.
                        ᐁᵢgrid_downstream = diff_domaingrid(
                            grid_downstream, dim, differentiation_order,
                            Δx, 'force_downstream',
                            do_ghost_communication=True,
                        )
                        # Apply force
                        apply_particle_mesh_force(
                            ᐁᵢgrid_downstream, dim, group[representation], interpolation_order,
                            ᔑdt, ᔑdt_key,
                        )
                        masterprint('done')

# Function for applying a scalar grid of the force along the dim'th
# dimension to receiver components.
@cython.header(
    # Arguments
    grid='double[:, :, ::1]',
    dim='int',
    receivers=list,
    interpolation_order='int',
    ᔑdt=dict,
    ᔑdt_key=object,  # str or tuple
    # Locals
    Jᵢ='FluidScalar',
    Jᵢ_ptr='double*',
    grid_ptr='double*',
    index='Py_ssize_t',
    receiver='Component',
    substitute_ᔑdt_key='bint',
    ϱ_ptr='double*',
    𝒫_ptr='double*',
    returns='void',
)
def apply_particle_mesh_force(grid, dim, receivers, interpolation_order, ᔑdt, ᔑdt_key):
    if not receivers:
        return
    if not (0 <= dim < 3):
        abort(f'apply_particle_mesh_force() called with dim = {dim} ∉ {{0, 1, 2}}')
    # When ᔑdt_key is a (2-)tuple, the last element needs to be
    # substituted with the name of a given component. The variable below
    # acts as a flag for this substitution.
    substitute_ᔑdt_key = isinstance(ᔑdt_key, tuple)
    # Apply the force grid to each receiver
    grid_ptr = cython.address(grid[:, :, :])
    for receiver in receivers:
        masterprint(f'Applying force to {receiver.name} ...')
        if substitute_ᔑdt_key:
            ᔑdt_key = (ᔑdt_key[0], receiver.name)
        if receiver.representation == 'particles':
            # Update the dim'th momentum component of all particles
            # through interpolation in the grid. To convert from force
            # to momentum change we should multiply by -mass*Δt (minus
            # as the force is the negative gradient of the potential),
            # where Δt = ᔑdt['1']. Here this integral over the time step
            # is generalised and supplied by the caller.
            interpolate_domaingrid_to_particles(
                grid, receiver, 'mom', dim, interpolation_order,
                receiver.mass*ℝ[-ᔑdt[ᔑdt_key]],
            )
        else:  # receiver.representation == 'fluid'
            # The source term has the form
            #   ΔJ ∝ -(ϱ + c⁻²𝒫)*ᐁφ.
            # The proportionality factor above is something like
            # Δt = ᔑdt['1']. Here this integral over the time step is
            # generalised and supplied by the caller.
            Jᵢ = receiver.J[dim]
            Jᵢ_ptr = Jᵢ.grid
            ϱ_ptr = receiver.ϱ.grid
            𝒫_ptr = receiver.𝒫.grid
            for index in range(receiver.size):
                Jᵢ_ptr[index] += ℝ[-ᔑdt[ᔑdt_key]]*(
                    ϱ_ptr[index] + ℝ[light_speed**(-2)]*𝒫_ptr[index]
                )*grid_ptr[index]
        masterprint('done')


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
def shortrange_progress_message(force, method, receivers, extra_message=' (short-range only)'):
    # Lookup appropriate form of the name of the force
    force = interactions_registered[force].conjugated_name
    # Print the progress message
    if method == 'p3m':
        if len(receivers) == 1:
            return f'{force} interaction for {receivers[0].name} via the P³M method{extra_message}'
        else:
            return (
                f'{force} interaction for {{{{{{}}}}}} via the P³M method{extra_message}'
                .format(', '.join([component.name for component in receivers]))
            )
    elif method == 'pp':
        if len(receivers) == 1:
            return f'{force} interaction for {receivers[0].name} via the PP method'
        else:
            return (
                f'{force} interaction for {{{{{{}}}}}} via the PP method'
                .format(', '.join([component.name for component in receivers]))
            )
    elif method == 'ppnonperiodic':
        if len(receivers) == 1:
            return f'{force} interaction for {receivers[0].name} via the non-periodic PP method'
        else:
            return (
                f'{force} interaction for {{{{{{}}}}}} via the non-periodic PP method'
                .format(', '.join([component.name for component in receivers]))
            )
    else:
        abort(f'The method "{method}" is unknown to shortrange_progress_message()')

# Function that given lists of receiver and supplier components of a
# one-way interaction removes any components from the supplier list that
# are also present in the receiver list.
def oneway_force(receivers, suppliers):
    return [component for component in suppliers if component not in receivers]

# Function which constructs a list of interactions from a list of
# components. The list of interactions store information about which
# components interact with one another, via what force and method.
def find_interactions(components, interaction_type='any', instantaneous='both'):
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
    Furthermore you may specify instantaneous to filter out interactions
    that are (not) instantaneous:
    - instantaneous == 'both':
      Include both instantaneous and non-instantaneous interactions.
    - instantaneous == True:
      Include only non-instantaneous interactions.
    - instantaneous == False:
      Include only instantaneous interactions.
    """
    # Use cached result
    key = (interaction_type, instantaneous, tuple(components))
    interactions_list = interactions_lists.get(key)
    if interactions_list is not None:
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
        interaction_info = interactions_registered.get(force)
        if interaction_info is None:
            abort(f'Force "{force}" is not implemented')
        for method in interaction_info.methods:
            if not method:
                # Should never happen
                abort(f'Falsy method "{method}" registered for force "{force}')
            if method not in methods_implemented:
                abort(f'Force method "{method}" not recognised')
        for method in methods:
            if not method:
                # When the method is set to an empty string it signifies
                # that this method should be used as a supplier for the
                # given force, but not receive the force itself.
                continue
            if method not in methods_implemented:
                abort(f'Force method "{method}" not recognised')
            if method not in interaction_info.methods:
                if len(interaction_info.methods) == 1:
                    abort(
                        f'Method "{method}" for force "{force}" is not implemented. '
                        f'Did you mean "{interaction_info.methods[0]}"?'
                    )
                else:
                    abort(
                        f'Method "{method}" for force "{force}" is not implemented. '
                        f'Did you mean one of {interaction_info.methods}?'
                    )
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
    for force, interaction_info in interactions_registered.items():
        methods = interaction_info.methods
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
        # Merge interactions of identical force, method and receivers,
        # but different suppliers; or identical force,
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
    # In the case that only long-/short-range interactions should be
    # considered, remove the unwanted interactions.
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
    # In the case that only (non-)instantaneous interactions should be
    # considered, remove the unwanted interactions.
    if 'True' in str(instantaneous):
        for interaction in interactions_list:
            if not interactions_registered[interaction.force].instantaneous:
                interaction.receivers[:] = []
        while cleanup():
            pass
    elif 'False' in str(instantaneous):
        for interaction in interactions_list:
            if interactions_registered[interaction.force].instantaneous:
                interaction.receivers[:] = []
        while cleanup():
            pass
    elif 'both' not in str(instantaneous):
        abort(f'find_interactions(): Unknown instantaneous value "{instantaneous}"')
    # Cache the result and return it
    interactions_lists[key] = interactions_list
    return interactions_list
# Global dict of interaction lists populated by the above function
cython.declare(interactions_lists=dict)
interactions_lists = {}
# Create the Interaction type used in the above function
Interaction = collections.namedtuple(
    'Interaction', ('force', 'method', 'receivers', 'suppliers')
)

# Function for registering interactions
def register(
    force, methods, conjugated_name=None,
    *,
    dependent=('pos', ), affected=('mom', ),
    deterministic=True, instantaneous=False,
):
    """Every implemented interaction should be registered by a call to
    this function. The order in which interactions are registered will
    be the order in which they are carried out, with the exception that
    short-range instantaneous interactions will be carried out before
    short-range non-instantaneous interactions.
    """
    # Canonicalize force and method names
    def canonicalize(s):
        s = s.lower()
        for char in ' _-^()':
            s = s.replace(char, '')
        for n in range(10):
            s = s.replace(unicode_superscript(str(n)), str(n))
        return s
    force = canonicalize(force)
    methods = [canonicalize(method) for method in any2list(methods)]
    # If no "conjugated" version of the force name is given,
    # set it equal to the normal name of the force.
    if conjugated_name is None:
        conjugated_name = force
    # Store the information globally as an InteractionInfo instance
    interactions_registered[force] = InteractionInfo(
        force, methods, conjugated_name,
        any2list(dependent), any2list(affected),
        deterministic, instantaneous,
    )
# Global dict of interaction infos populated by the above function
cython.declare(interactions_registered=dict)
interactions_registered = {}
# Create the InteractionInfo type used in the above function
InteractionInfo = collections.namedtuple(
    'InteractionInfo',
    (
        'force', 'methods', 'conjugated_name',
        'dependent', 'affected',
        'deterministic', 'instantaneous',
    ),
)

# Function which looks up quantities defined between pairs of
# components within a passed dict.
@cython.header(
    # Arguments
    receivers=list,
    suppliers=list,
    select_dict=dict,
    name=str,
    # Locals
    key=tuple,
    pair=set,
    pairs=list,
    quantities=dict,
    quantity=object,
    quantity_r=object,
    quantity_s=object,
    receiver='Component',
    supplier='Component',
    returns=dict,
)
def get_pairwise_quantities(receivers, suppliers, select_dict, name=''):
    """The "name" argument is only used in relation with the caching.
    """
    # Attempt lookup in cache
    if name:
        key = (
            name,
            frozenset([receiver.name for receiver in receivers]),
            frozenset([supplier.name for supplier in suppliers]),
        )
        quantities = pairwise_quantities.get(key)
        if quantities is not None:
            return quantities
    # Result not present in cache. Do the lookups.
    quantities = {}
    pairs = []
    for receiver in receivers:
        for supplier in suppliers:
            pair = {receiver, supplier}
            if pair in pairs:
                continue
            pairs.append(pair)
            # Look up the quantity for this {receiver, supplier} pair
            quantity = is_selected((receiver, supplier), select_dict)
            if quantity is None:
                if receiver.name == supplier.name:
                    quantity = is_selected(receiver, select_dict)
                else:
                    quantity_r = is_selected(receiver, select_dict)
                    quantity_s = is_selected(supplier, select_dict)
                    if quantity_r == quantity_s:
                        quantity = quantity_r
            if quantity is None:
                if name:
                    abort(
                        f'get_pairwise_quantities(): No pairwise quantity "{name}" '
                        f'for {{{receiver.name}, {supplier.name}}} specified in the passed dict.'
                    )
                else:
                    abort(
                        f'get_pairwise_quantities(): No pairwise quantity '
                        f'for {{{receiver.name}, {supplier.name}}} specified in the passed dict.'
                    )
            # Store the found quantity symmetrically
            # with respect to the receiver and supplier.
            quantities[receiver.name, supplier.name] = quantity
            quantities[supplier.name, receiver.name] = quantity
    # Save the found quantities to the cache
    if name:
        pairwise_quantities[key] = quantities
    return quantities
# Cache used by the above function
cython.declare(pairwise_quantities=dict)
pairwise_quantities = {}

# Function for looking up various information needed for potential
# (i.e. particle mesh) interactions.
@cython.header(
    # Arguments
    force=str,
    method=str,
    receivers=list,
    suppliers=list,
    # Locals
    deconvolve=object,  # PotentialDeconvolutions
    gridsize='Py_ssize_t',
    gridsizes=set,
    interlace='bint',
    interpolation_order='int',
    key=tuple,
    potential_specs=object,  # PotentialInfo
    receiver='Component',
    supplier='Component',
    returns=object,  # PotentialInfo
)
def get_potential_specs(force, method, receivers, suppliers):
    # Cache lookup
    key = (force, method, tuple(receivers), tuple(suppliers))
    potential_specs = potential_specs_cache.get(key)
    if potential_specs is not None:
        return potential_specs
    # Look up the global potential grid size
    gridsize = 𝕆[potential_options['gridsize']['global']][force][method]
    if gridsize == -1:
        # No global grid size specified. If all supplier upstream
        # and receiver downstream grid sizes are equal, use this for
        # the global grid size.
        gridsizes = (
            {
                supplier.potential_gridsizes[force][method].upstream
                for supplier in suppliers
            } | {
                receiver.potential_gridsizes[force][method].downstream
                for receiver in receivers
            }
        )
        if len(gridsizes) != 1:
            abort(
                f'No global potential grid size specified for force "{force}" with '
                f'method "{method}". As multiple upstream and/or downstream grid sizes '
                f'are in use, the global grid size could not be set automatically.'
            )
        gridsize = gridsizes.pop()
    # Fetch interpolation, deconvolution and interlacing specifications
    interpolation_order   = 𝕆[potential_options['interpolation'  ]][force][method]
    deconvolve            = 𝕆[potential_options['deconvolve'     ]][force][method]
    interlace             = 𝕆[potential_options['interlace'      ]][force][method]
    # Cache and return
    potential_specs = PotentialInfo(gridsize, interpolation_order, deconvolve, interlace)
    potential_specs_cache[key] = potential_specs
    return potential_specs
# Cache and type used by the above function
cython.declare(potential_specs_cache=dict)
potential_specs_cache = {}
PotentialInfo = collections.namedtuple(
    'PotentialInfo', ('gridsize', 'interpolation_order', 'deconvolve', 'interlace'),
)



#########################################
# Implement specific interactions below #
#########################################

# Gravity
cimport('from gravity import *')
register('gravity', ['ppnonperiodic', 'pp', 'p3m', 'pm'], 'gravitational')
@cython.pheader(
    # Arguments
    method=str,
    receivers=list,
    suppliers=list,
    ᔑdt=dict,
    interaction_type=str,
    printout='bint',
    # Locals
    extra_message=str,
    force=str,
    potential=str,
    potential_specs=object,  # PotentialInfo
    quantity=str,
    ᔑdt_key=tuple,
)
def gravity(method, receivers, suppliers, ᔑdt, interaction_type, printout):
    force = 'gravity'
    # Set up variables used by potential/grid (PM and P³M) methods
    if method in {'pm', 'p3m'}:
        potential_specs = get_potential_specs(force, method, receivers, suppliers)
        # The gravitational potential is given by the Poisson equation
        # ∇²φ = 4πGa²ρ = 4πGa**(-3*w_eff - 1)ϱ,
        # summed over all suppliers.
        # The component dependent quantity is then a²ρ.
        # In the case of the long-range part only (for use with P³M),
        # though the expression for the potential is altered, we still
        # have φ ∝ a²ρ.
        quantity = 'a²ρ'
        # In the fluid description, the gravitational source term is
        # ∂ₜJⁱ = ⋯ -a**(-3*w_eff)*(ϱ + c⁻²𝒫)*∂ⁱφ
        # and so a**(-3*w_eff) should be integrated over the time step
        # to get ΔJⁱ. In the particle description, the gravitational
        # source term is
        # ∂ₜmomⁱ = -mass*∂ⁱφ.
        # In the general case of a changing mass, the current mass is
        # given by mass*a**(-3*w_eff), and so again, a**(-3*w_eff)
        # should be integrated over the time step
        # in order to obtain Δmomⁱ.
        ᔑdt_key = ('a**(-3*w_eff)', 'component')
    # Compute gravity via one of the following methods
    if method == 'pm':
        # The particle-mesh method
        if printout:
            masterprint(
                f'Executing gravitational interaction for {receivers[0].name} '
                f'via the PM method ...'
                if len(receivers) == 1 else (
                    'Executing gravitational interaction for {{{}}} via the PM method ...'
                    .format(', '.join([component.name for component in receivers]))
                )
            )
        potential = 'gravity'
        particle_mesh(
            receivers, suppliers, potential_specs.gridsize, quantity, force, method, potential,
            potential_specs.interpolation_order,
            potential_specs.deconvolve.upstream, potential_specs.deconvolve.downstream,
            potential_specs.interlace,
            ᔑdt, ᔑdt_key,
        )
        if printout:
            masterprint('done')
    elif method == 'p3m':
        # The particle-particle-mesh method
        if printout:
            extra_message = ''
            if 𝔹['long' in interaction_type]:
                extra_message = ' (long-range only)'
            elif 𝔹['short' in interaction_type]:
                extra_message = ' (short-range only)'
            masterprint(
                'Executing',
                shortrange_progress_message(force, method, receivers, extra_message),
                '...',
            )
        # The long-range PM part
        if 𝔹['any' in interaction_type] or 𝔹['long' in interaction_type]:
            potential = 'gravity long-range'
            particle_mesh(
                receivers, suppliers, potential_specs.gridsize, quantity, force, method, potential,
                potential_specs.interpolation_order,
                potential_specs.deconvolve.upstream, potential_specs.deconvolve.downstream,
                potential_specs.interlace,
                ᔑdt, ᔑdt_key,
            )
        # The short-range PP part
        if 𝔹['any' in interaction_type] or 𝔹['short' in interaction_type]:
            component_component(
                force, receivers, suppliers, gravity_pairwise_shortrange, ᔑdt,
                pairing_level='tile',
            )
        if printout:
            masterprint('done')
    elif method == 'pp':
        # The particle-particle method with Ewald-periodicity
        if printout:
            masterprint(
                'Executing',
                shortrange_progress_message(force, method, receivers),
                '...',
            )
        get_ewald_grid()
        component_component(
            force, receivers, suppliers, gravity_pairwise, ᔑdt,
            pairing_level='domain',
        )
        if printout:
            masterprint('done')
    elif method == 'ppnonperiodic':
        # The non-periodic particle-particle method
        if printout:
            masterprint(
                'Executing',
                shortrange_progress_message(force, method, receivers),
                '...',
            )
        component_component(
            force, receivers, suppliers, gravity_pairwise_nonperiodic, ᔑdt,
            pairing_level='domain',
        )
        if printout:
            masterprint('done')
    elif master:
        abort(f'gravity() was called with the "{method}" method')

# The lapse force
register('lapse', 'pm')
@cython.pheader(
    # Arguments
    method=str,
    receivers=list,
    suppliers=list,
    ᔑdt=dict,
    interaction_type=str,
    printout='bint',
    # Locals
    force=str,
    potential=str,
    potential_specs=object,  # PotentialInfo
    quantity=str,
    ᔑdt_key=tuple,
)
def lapse(method, receivers, suppliers, ᔑdt, interaction_type, printout):
    force = 'lapse'
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
        # Get interaction specifications
        potential_specs = get_potential_specs(force, method, receivers, suppliers)
        # As the lapse potential is implemented exactly analogous to the
        # gravitational potential, it obeys the Poisson equation
        # ∇²φ = 4πGa²ρ = 4πGa**(-3*w_eff - 1)ϱ,
        # with φ the lapse potential and ρ, ϱ and w_eff belonging to the
        # fictitious lapse species.
        quantity = 'a²ρ'
        potential = 'gravity'
        # As the lapse potential is implemented exactly analogous to the
        # gravitational potential, the momentum updates are again
        # proportional to a**(-3*w_eff) integrated over the time step
        # (see the gravity function for a more detailed explanation).
        # The realised lapse potential is the common lapse potential,
        # independent of the component in question which is to receive
        # momentum updates. The actual lapse potential needed for a
        # given component is obtained by multiplying the common lapse
        # potential by Γ/H, where Γ is the decay rate of the component
        # and H is the Hubble parameter. As these are time dependent,
        # the full time step integral is then a**(-3*w_eff)*Γ/H.
        ᔑdt_key = ('a**(-3*w_eff)*Γ/H', 'component')
        # Execute the lapse interaction
        particle_mesh(
            receivers, suppliers, potential_specs.gridsize, quantity, force, method, potential,
            potential_specs.interpolation_order,
            potential_specs.deconvolve.upstream, potential_specs.deconvolve.downstream,
            potential_specs.interlace,
            ᔑdt, ᔑdt_key,
        )
        if printout:
            masterprint('done')
    elif master:
        abort(f'lapse() was called with the "{method}" method')
