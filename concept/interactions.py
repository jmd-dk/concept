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
cimport('from gravity import build_φ, P3M, PM, PP')



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
        PP(component, ᔑdt)
        masterprint('done')
    elif method == 'pm':
        # The particle-mesh method.
        # Construct the gravitational potential from all components
        # which interacts gravitationally.
        φ = build_φ(components)
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
                PM(component, ᔑdt, gradφ_dim, dim)
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
        φ = build_φ(components, only_long_range=True)
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
                PM(component, ᔑdt, gradφ_dim, dim)
                masterprint('done')
        # Now apply the short-range gravitational forces
        masterprint('Gravitationally (P³M, short range) accelerating {} ...'.format(component.name))
        P3M(component, ᔑdt)
        masterprint('done')
    elif master:
        abort('gravity was called with the "{}" method'.format(method))



# Function which constructs a list of interactions from a list of
# components. The list of interactions store information about which
# components interact with one another, via what force and method.
@cython.header(# Arguments
               components='list',
               # Locals
               Interaction='object',  # A new class (via namedtuple)
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
    Interaction = collections.namedtuple('Interaction', ('force',
                                                         'method',
                                                         'receivers',
                                                         'suppliers',
                                                         ))
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

# Specification of implemented forces.
# The order specified here will be the order in which the forces
# are computed and applied. Each element of the list should be a
# 2-tuple in the format (force, method).
cython.declare(forces_implemented='list')
forces_implemented = [('gravity', 'pp' ),
                      ('gravity', 'p3m'),
                      ('gravity', 'pm' ),
                      ]
