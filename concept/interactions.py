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
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
from mesh import diff_domain
cimport('from gravity import build_φ, p3m, pm, pp')



# # Generic function implementing pairwise interactions
# # between receivers and suppliers.
# @cython.header(# Arguments
#                receivers='list',
#                suppliers='list',
#                ᔑdt='dict',
#                dependencies='str',
#                # Locals
#                mutual='bint',
#                only_supply='bint',
#                rank_send='int',
#                rank_recv='int',
#                receiver='Component',
#                receiver_i='Component',
#                receiver_j='Component',
#                supplier='Component',
#                )
# def pairwise(receivers, suppliers, ᔑdt, dependencies='positions'):
#     """This is the master function for pairwise interactions.
#     It takes care of all pairings, which include
#     - Receivers with themselves (interactions bwtween
#       particles/fluid elements within a component).
#     - Receivers with other receivers.
#     - Receivers with suppliers.
#     """
#     # List of all particles participating in this interaction
#     components = receivers + suppliers
#     # Pair each receiver with all receivers and suppliers
#     for component_1 in receivers:
#         for index_component, component_2 in enumerate(components):
#             # Flag specifying whether component_2 should only supply
#             # forces to component_1 and not receive any momentum
#             # updates itself.
#             only_supply = (index_component >= ℝ[len(receivers)])
#             # Pair this process/domain with every other process/domain.
#             # The pairing pattern is as follows. This process is paired
#             # with two other processes simultaneously; those with a rank
#             # given by (the local) rank ± i. The local particles/fluid
#             # elements will be sent to the process rank + i, which then
#             # compute the pairwise forces between the received
#             # particles/fluid elements and its local particles/fluid 
#             # elements. These forces are then applied locally to update
#             # the local momenta on the process, but external momentum
#             # updates are also send back to the orginal process where
#             # these are then applied.
#             # Note that the first pairing is between this process and
#             # itself (inter-domain interactions). For even nprocs, the
#             # last pairing will include this process and a single other
#             # process. In this case, the momentum upates are not send
#             # back to the other process, as both processes in the pair
#             # now hold all of the interaction information.
#             for i in range(ℤ[1 + nprocs//2]):
#                 # Process ranks to send/receive to/from
#                 rank_send = mod(rank + i, nprocs)
#                 rank_recv = mod(rank - i, nprocs)
#                 # Flag specifying whether the interaction is mutual,
#                 # meaning that both component_1 and component_2 should
#                 # receive momentum updates due to the interaction.
#                 if only_supply:
#                     mutual = False
#                 else:
#                     mutual = (rank_send != rank_recv)
#                 # Do the pairwise interaction
#                 pairwise_domains(component_1, component_2, ᔑdt, dependencies,
#                                  rank_send, rank_recv, mutual)

# @cython.header(# Arguments
#                receivers='list',
#                suppliers='list',
#                ᔑdt='dict',
#                dependencies='str',
#                rank_send='int',
#                rank_recv='int',
#                mutual='bint',
#                # Locals
#                Jx_1='double[:, :, :]',
#                Jy_1='double[:, :, :]',
#                Jz_1='double[:, :, :]',
#                momentum_dependent='bint',
#                position_dependent='bint',
#                ϱ_1='double[:, :, :]',
#                )
# def pairwise_domains(component_1, component_2, ᔑdt, dependencies, rank_send, rank_recv, mutual):
#     """
#     """
#     # Determine force dependencies
#     position_dependent = 'pos' in dependencies.lower()
#     momentum_dependent = 'mom' in dependencies.lower()
#     # Extract local particle/fluid element data of component_1
#     mass_local1 = component_1.mass
#     if component_1.representation == 'particles':
#         N_local1    = component_local1.N
#         posx_local1 = component_local1.posx
#         posy_local1 = component_local1.posy
#         posz_local1 = component_local1.posz
#         momx_local1 = component_local1.momx
#         momy_local1 = component_local1.momy
#         momz_local1 = component_local1.momz
#     elif component_1.representation == 'fluid':
#         gridsize_local1 = component_local1.gridsize
#         ϱ_local1        = component_local1.ϱ .grid_noghosts
#         Jx_local1       = component_local1.Jx.grid_noghosts
#         Jy_local1       = component_local1.Jy.grid_noghosts
#         Jz_local1       = component_local1.Jz.grid_noghosts
#     # Extract local particle/fluid element data of component_2
#     mass_local2 = component_2.mass
#     if component_2.representation == 'particles':
#         N_local2    = component_local2.N
#         posx_local2 = component_local2.posx
#         posy_local2 = component_local2.posy
#         posz_local2 = component_local2.posz
#         momx_local2 = component_local2.momx
#         momy_local2 = component_local2.momy
#         momz_local2 = component_local2.momz
#     elif component_2.representation == 'fluid':
#         gridsize_local2 = component_local2.gridsize
#         ϱ_local2        = component_local2.ϱ .grid_noghosts
#         Jx_local2       = component_local2.Jx.grid_noghosts
#         Jy_local2       = component_local2.Jy.grid_noghosts
#         Jz_local2       = component_local2.Jz.grid_noghosts
#     # Communicate particle/fluid element data.
#     # We choose to communicate component_2 (send local component_2 and
#     # receive external component_2) and not component_1, as sometimes
#     # the interaction is not mutual, but only component_2
#     # acts on component_1.
#     mass_extnl2 = mass_local2
#     if rank == rank_send:
#         # This is a local (inter-domain) interaction,
#         # so no communication is needed.
#         if component_2.representation == 'particles':
#             N_extnl2    = N_local2
#             posx_extnl2 = posx_local2
#             posy_extnl2 = posy_local2
#             posz_extnl2 = posz_local2
#             momx_extnl2 = momx_local2
#             momy_extnl2 = momy_local2
#             momz_extnl2 = momz_local2
#         elif component_2.representation == 'fluid':
#             gridsize_extnl2 = gridsize_local2
#             ϱ_extnl2        = ϱ_local2
#             Jx_extnl2       = Jx_local2
#             Jy_extnl2       = Jy_local2
#             Jz_extnl2       = Jz_local2
#     else:
#         # Send and receive component_2
#         if component_2.representation == 'particles':
#             N_extnl2    = N_local2
#             if position_dependent:
#                 # Communicate positions
#                 ...
#             if momentum_dependent:
#                 # Communicate momenta
#                 ...


#     for i in range(N_local_1):
#         j_start = 0
#         for j in range(j_start, N_local_1):





# Function that carry out the gravitational interaction
@cython.pheader(# Arguments
                method='str',
                receivers='list',
                suppliers='list',
                ᔑdt='dict',
                # Locals
                component='Component',
                components='list',
                dim='int',
                gradφ_dim='double[:, :, ::1]',
                h='double',
                φ='double[:, :, ::1]',
                )
def gravity(method, receivers, suppliers, ᔑdt):
    # List of all particles participating in this interaction
    components = receivers + suppliers
    # Compute gravity via one of the following methods
    if method == 'pp':
        # The particle-particle method.
        # So far, this method is only implemented between particles
        # in a single component.
        if len(components) != 1:
            abort('The PP method can only be used with a single gravitating component')
        component = components[0]
        # So far, this method is only implemented for
        # particle components, not fluids.
        if component.representation != 'particles':
            abort('The PP method can only be used with particle components')
        masterprint('Gravitationally (PP) accelerating {} ...'.format(component.name))
        pp(component, ᔑdt)
        masterprint('done')
    elif method == 'pm':
        # The particle-mesh method.
        # Construct the gravitational potential from all components
        # which interacts gravitationally.
        φ = build_φ(components, ᔑdt)
        # For each dimension, differentiate φ and apply the force to
        # all receiver components.
        h = boxsize/φ_gridsize  # Physical grid spacing of φ
        for dim in range(3):
            # Do the differentiation of φ
            gradφ_dim = diff_domain(φ, dim, h, order=4)
            # Apply PM force to all the receivers
            for component in receivers:
                masterprint('Applying gravitational (PM) forces along the {}-direction to {} ...'
                            .format('xyz'[dim], component.name)
                            )
                pm(component, ᔑdt, gradφ_dim, dim)
                masterprint('done')
    elif method == 'p3m':
        # The particle-particle-mesh method.
        # So far, this method is only implemented between particles
        # in a single component.
        if len(components) != 1:
            abort('The P³M method can only be used with a single gravitating component')
        component = components[0]
        # So far, this method is only implemented for
        # particle components, not fluids.
        if component.representation != 'particles':
            abort('The P³M method can only be used with particle components')
        # Construct the long-range gravitational potential from all
        # components which interacts gravitationally.
        φ = build_φ(components, ᔑdt, only_long_range=True)
        # For each dimension, differentiate φ and apply the force to
        # all receiver components.
        h = boxsize/φ_gridsize  # Physical grid spacing of φ
        for dim in range(3):
            # Do the differentiation of φ
            gradφ_dim = diff_domain(φ, dim, h, order=4)
            # Apply long-range P³M force to all the receivers
            for component in receivers:
                masterprint('Applying gravitational (P³M, long-range) forces along the {}-direction to {} ...'
                            .format('xyz'[dim], component.name)
                            )
                pm(component, ᔑdt, gradφ_dim, dim)
                masterprint('done')
        # Now apply the short-range gravitational forces
        masterprint('Gravitationally (P³M, short-range) accelerating {} ...'.format(component.name))
        p3m(component, ᔑdt)
        masterprint('done')
    elif master:
        abort('gravity was called with the "{}" method'.format(method))

# Function which constructs a list of interactions from a list of
# components. The list of interactions store information about which
# components interact with one another, via what force and method.
@cython.header(# Arguments
               components='list',
               # Locals
               component='Component',
               force='str',
               force_method='tuple',
               forces_in_use='set',
               interactions_list='list',
               method='str',
               receivers='list',
               suppliers='list',
               returns='list',
               )
def find_interactions(components):
    # Find all unique (force, method) pairs in use
    forces_in_use = {force_method for component in components
                                  for force_method in component.forces.items()}
    # Check that all forces and methods assigned
    # to the components are implemented.
    for force_method in forces_in_use:
        if force_method not in forces_implemented:
            abort('The force "{}" with method "{}" is not implemented'.format(*force_method))
    # Construct the interactions_list with (named) 4-tuples
    # in the format (force, method, receivers, suppliers),
    # where receivers is a list of all components which interact
    # via the force and should therefore receive momentum updates
    # computed via this force and the method given as the
    # second element. The list suppliers contain all components
    # which interact via the same force but using a different method.
    # These will supply momentum updates to the receivers, but will not
    # themselves receive any momentum updates. This does not break
    # Newton's third law (up to numerical precision) because the missing
    # momentum updates will be supplied by another method, given in
    # another 4-tuple. Note that the receivers not only receive but also
    # supply momentum updates to other receivers. Note also that the
    # same force can appear in multiple 4-tuples but with
    # different methods.
    interactions_list = []
    for force, method in forces_implemented:
        if (force, method) not in forces_in_use:
            continue
        # Find all receiver and supplier components
        # for this (force, method) pair.
        receivers = []
        suppliers = []
        for component in components:
            if force in component.forces:
                if component.forces[force] == method:
                    receivers.append(component)
                else:
                    suppliers.append(component)
        # Store the 4-tuple in the interactions_list
        interactions_list.append(Interaction(force, method, receivers, suppliers))
    return interactions_list
# Create the Interaction type used in the above function
Interaction = collections.namedtuple('Interaction', ('force',
                                                     'method',
                                                     'receivers',
                                                     'suppliers',
                                                     )
                                     )

# Specification of implemented forces.
# The order specified here will be the order in which the forces
# are computed and applied. Each element of the list should be a
# 2-tuple in the format (force, method).
cython.declare(forces_implemented='list')
forces_implemented = [('gravity', 'pp' ),
                      ('gravity', 'p3m'),
                      ('gravity', 'pm' ),
                      ]
