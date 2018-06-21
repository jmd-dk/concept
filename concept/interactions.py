# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2018 Jeppe Mosgaard Dakin.
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
cimport('from communication import sendrecv_component')
cimport('from mesh import CIC_components2φ, diff_domain, domain_decompose, fft, slab_decompose')
cimport('from mesh import CIC_components2φ_general')
# Import interactions defined in other modules
cimport('from gravity import *')
# DELETE WHEN DONE with gravity_old.py !!!
cimport('from gravity_old import build_φ, p3m, pm, pp')

# Function pointer types used in this module
pxd("""
#                                       component_1, component_2, rank_2, ᔑdt , local, mutual, extra_args
ctypedef void   (*func_interaction    )(Component  , Component  , int   , dict, bint , bint  , dict      )
#                                       k2
ctypedef double (*func_potential      )(double)
#                                       component, ᔑdt , gradφ_dim        , dim
ctypedef void   (*func_apply_potential)(Component, dict, double[:, :, ::1], int)
""")



# Generic function implementing domain-domain pairing
@cython.header(# Arguments
              receivers=list,
              suppliers=list,
              ᔑdt=dict,
              interaction=func_interaction,
              interaction_name=str,
              dependent=list,  # list of str's
              affected=list,   # list of str's
              deterministic='bint',
              extra_args=dict,
              # Locals
              N_domain_pairs='Py_ssize_t',
              assisted='bint',
              components=list,
              component_1='Component',
              component_2_extrl='Component',
              component_2_local='Component',
              i='Py_ssize_t',
              index_component_1='Py_ssize_t',
              index_component_2='Py_ssize_t',
              local='bint',
              mutual='bint',
              only_supply='bint',
              rank_send='int',
              rank_recv='int',
              synchronous='bint',
              )
def domain_domain(receivers, suppliers, ᔑdt, interaction, interaction_name,
                  dependent, affected, deterministic, extra_args={}):
    """This function takes care of pairings between all components
    and between all domains. The component-pairings include:
    - Receivers with themselves (interactions between
      particles/fluid elements within a component).
    - Receivers with other receivers.
    - Receivers with suppliers.
    The receiver components are denoted component_1. These will not
    be communicated (except when implicitly used as suppliers).
    The supplier components will be denoted component_2. These will
    be sent to other processes (domains) and also recieved back from
    other processes. Thus both local and external versions of
    component_2 exist, called component_2_local and component_2_extrl.

    If affected is an empty list, this is not really an interaction.
    In this case, every domain will both send and receive from every
    other domain.
    """
    # List of all particles participating in this interaction
    components = receivers + suppliers
    # Determine whether this "interaction" have any direct effect
    # on the components at all. If not, this will change the
    # communication pattern.
    assisted = True
    if not affected:
        assisted = False
    # Pair each receiver with all receivers and suppliers
    for     index_component_1, component_1       in enumerate(receivers):
        for index_component_2, component_2_local in enumerate(components[index_component_1:]):
            if component_2_local.representation != 'particles':  # !!! Generalize to fluids also
                abort('The domain_domain function is only implemented for particles')
            # Flag specifying whether component_2 should only supply
            # forces to component_1 and not receive any momentum
            # updates itself.
            only_supply = (index_component_2 >= ℤ[len(receivers)])
            # Display progress message
            if interaction_name:
                if index_component_1 == index_component_2:
                    masterprint('Letting {} interact under {} ...'
                                .format(component_1.name, interaction_name)
                                )
                elif only_supply:
                    masterprint('Letting {} interact with {} under {} ...'
                                .format(component_1.name, component_2_local.name, interaction_name)
                                )
                else:
                    masterprint('Letting {} and {} interact under mutual {} ...'
                                .format(component_1.name, component_2_local.name, interaction_name)
                                )
            # Pair this process/domain with every other process/domain.
            # The pairing pattern is as follows. This process is paired
            # with two other processes simultaneously; those with a rank
            # given by (the local) rank ± i:
            # - Local component_2 -> rank + i
            # - Extrl component_2 <- rank - i
            # On each process, the local component_1 and the external
            # (received) component_2 then interact.
            N_domain_pairs = ℤ[1 + nprocs//2] if assisted else nprocs
            for i in range(N_domain_pairs):
                # Process ranks to send to and receive from
                rank_send = mod(rank + i, nprocs)
                rank_recv = mod(rank - i, nprocs)
                # Determine whther component_2 should be updated due to
                # its interaction with component_1. This is usually the
                # case. The exceptions are
                # - component_2 is a supplier and not a receiver.
                #   When this is the case, only_supply is True.
                # - component_2 is really the same as component_1
                #   and this is a local interaction, meaning that
                #   both rank_send and rank_recv is really just the
                #   local rank. In this case, the supplied interaction
                #   function should also update the data of component_2
                #   (not the buffers). This special case is flagged
                #   by the 'local' variable being True.
                # - component_2 is really the same as component_1
                #   and rank_send == rank_recv (but different from the
                #   local rank). In this case, the external updates to
                #   component_2 should not be send back and applied, as
                #   these updates are already done locally on the other
                #   process. This special case is flagged by the
                #   'synchronous' variable being True. The exception
                #   (to this exception) is when the interaction is non-
                #   deterministic, in which case this (supposedly
                #   identical) interaction must not be computed by both
                #   processes synchronously, as they may produce
                #   different results.
                local = False
                if index_component_1 == index_component_2 and rank_send == rank == rank_recv:
                    local = True
                synchronous = False
                if not local and index_component_1 == index_component_2 and rank_send == rank_recv:
                    synchronous = True
                mutual = True
                if only_supply or local or (synchronous and deterministic) or not assisted:
                    mutual = False
                # Communicate the dependent variables
                # (e.g. pos for gravity) of component_2.
                component_2_extrl = sendrecv_component(component_2_local,
                                                       dependent, dest=rank_send,
                                                                  source=rank_recv,
                                                       )
                # Let the local component_1 interaction with the
                # external component_2. This will update the affected
                # variables (e.g. mom for gravity) of the local
                # component_1 and populate the affected variable buffers
                # of the external component_2, if mutual is True.
                # If this is a synchronous interaction (meaning that it
                # is between only two processes, rank_send == rank_recv)
                # and also non-deterministic, perform the interaction
                # only on one of the two processes. The process with
                # the lower rank is chosen for the job.
                if (    not synchronous
                    or (    synchronous and     deterministic)
                    or (    synchronous and not deterministic and rank < rank_send)
                    ):
                    interaction(component_1, component_2_extrl,
                                rank_recv, ᔑdt, local, mutual, extra_args)
                if mutual:
                    # Send the populated buffers back to the process
                    # from which the external component_2 came.
                    # Add the received values in the buffers to the
                    # affected variables (e.g. mom for gravity) of
                    # the local component_2.
                    sendrecv_component(component_2_extrl,
                                       affected,
                                       dest=rank_recv,
                                       source=rank_send,
                                       component_recv=component_2_local,
                                       )
                    # Nullify the Δ buffers of the external component_2,
                    # leaving this with no leftover junk.
                    component_2_extrl.nullify_Δ(affected)
            masterprint('done')

# Generic function implementing particle-mesh interactions
@cython.header(# Arguments
               receivers=list,
               suppliers=list,
               ᔑdt=dict,
               potential=func_potential,
               potential_name=str,
               dependent=list,
               apply_potential=func_apply_potential,
               # Locals
               component='Component',
               components=list,
               dim='int',
               gradφ_dim='double[:, :, ::1]',
               h='double',
               φ='double[:, :, ::1]',
               )
def particle_mesh(receivers, suppliers, ᔑdt, potential, potential_name,
                  dependent, apply_potential):
    """This function will update the affected variables of all receiver
    components due to an interaction. This is done by constructing a
    global field by interpolating the dependent variables of all
    receivers and suppliers onto a grid (the φ grid is used for this).
    The supplied 'dependent' argument is thus a list of variables which
    should be interpolated to the grid. For details on the structure
    of this argument, see the CIC_components2domain_grid function
    in the mesh module, where the corresponding argument is called
    quantities.

    The field is then Fourier transformed. To transform the grid to
    the (Fourier transformed) potential, each grid point is multiplied
    by potential(k2), where k2 = k² is the squared magnitude of the wave
    vector at the given grid point. For further details on the potential
    argument, see the construct_potential function.

    The grid is then Fourier transformed back to real space and
    differentiated along each dimension to get the force. This force is
    passed to the apply_potential function for each receiver and
    each dimension.
    """
    # Build the potential due to all components
    components = receivers + suppliers
    masterprint('Constructing the {} due to {} ...'
                .format(potential_name, ', '.join([component.name for component in components])))
    φ = construct_potential(components, dependent, potential)
    masterprint('done')
    # For each dimension, differentiate φ and apply the force to
    # all receiver components.
    h = boxsize/φ_gridsize  # Physical grid spacing of φ
    for dim in range(3):
        masterprint('Differentiating the {} along the {}-direction and applying it ...'
                    .format(potential_name, 'xyz'[dim])
                    )
        # Do the differentiation of φ
        gradφ_dim = diff_domain(φ, dim, h, order=4)
        # Apply force to all the receivers
        for component in receivers:
            masterprint('Applying to {} ...'.format(component.name))
            apply_potential(component, ᔑdt, gradφ_dim, dim)
            masterprint('done')
        masterprint('done')

# Generic function capable of constructing a potential grid out of
# components and a given expression for the potential.
@cython.header(# Arguments
               components=list,
               quantities=list,
               potential=func_potential,
               # Locals
               double_deconv='double',
               fft_normalization_factor='double',
               i='Py_ssize_t',
               j='Py_ssize_t',
               j_global='Py_ssize_t',
               k='Py_ssize_t',
               ki='Py_ssize_t',
               kj='Py_ssize_t',
               kj2='Py_ssize_t',
               kk='Py_ssize_t',
               k2='Py_ssize_t',
               slab='double[:, :, ::1]',
               slab_jik='double*',
               reciprocal_sqrt_deconv_ij='double',
               reciprocal_sqrt_deconv_ijk='double',
               reciprocal_sqrt_deconv_j='double',
               φ='double[:, :, ::1]',
               returns='double[:, :, ::1]',
               )
def construct_potential(components, quantities, potential):
    """This function populate the φ grid (including pseudo points and
    ghost layers) with a real-space potential corresponding to the
    Fourier-space potential function given, due to all the components.
    First the variables given in 'quantities' of the components are
    interpolated to the φ grid, then the grid is Fourier transformed,
    then the potential function is used to change the value of each grid
    point, then the grid is Fourirer transformed back to real space.
    Which variables to extrapolate to the grid is determined by the
    quantities argument. For details on this argument, see the
    CIC_components2domain_grid function in the mesh module.

    To transform the grid with interpolated component variables to
    an actual potential, the grid is Fourier transformed after which the
    each grid point is multiplied by potential(k2), potential is the
    supplied function and k2 = k² is the squared magnitude of the wave
    vector at the given grid point, in physical units. For normal
    gravity, we have φ(k) = -4πGa²ρ(k)/k² = -4πG a**(-3*w_eff - 1) ϱ(k)/k²,
    which can be signalled by passing
    quantities = [('particles', a**(-1)*mass/Vcell),
                  ('ϱ', a**(-3*w_eff - 1))],
    potential = lambda k2: -4*π*G_Newton/k2
    (note: it is not allowed to actually pass a lambda function,
    in compiled mode anyway).
    """
    # CIC interpolate the particles/fluid elements onto the slabs
    φ = CIC_components2φ(components, quantities)
    slab = slab_decompose(φ, prepare_fft=True)
    # Do forward Fourier transform on the slabs
    # containing the density field.
    fft(slab, 'forward')
    # Multiplicative factor needed after a forward and a backward
    # Fourier transformation.
    fft_normalization_factor = float(φ_gridsize)**(-3)
    # Loop through the local j-dimension
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
            reciprocal_sqrt_deconv_ij = sinc(ki*ℝ[π/φ_gridsize])*reciprocal_sqrt_deconv_j
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
                # The real-space mean value of the potential will then
                # be zero, as it should for a peculiar potential.
                if k2 == 0:
                    slab_jik[0] = 0  # Real part
                    slab_jik[1] = 0  # Imag part
                    continue
                # Reciprocal square root of the product of
                # all components of the deconvolution.
                reciprocal_sqrt_deconv_ijk = reciprocal_sqrt_deconv_ij*sinc(kk*ℝ[π/φ_gridsize])
                # A full deconvolution is now
                # 1/reciprocal_sqrt_deconv_ijk**2.
                # We need however two such full deconvolutions, one for
                # each CIC interpolation (the component assignment and
                # the upcoming force interpolation).
                double_deconv = 1/reciprocal_sqrt_deconv_ijk**4
                # Get the factor from the potential function at this k².
                # The physical squared length of the wave vector is
                # given by (2π/boxsize*|k|)².
                potential_factor = potential(ℝ[(2*π/boxsize)**2]*k2)
                # Transform the grid in the following ways:
                # - Multiply by potential_factor, converting the grid
                #   values to actual potential values
                #   (in Fourier space).
                # - Multiply by double_deconv, taking care of the
                #   deconvolution needed due to the
                #   two CIC-interpolations.
                # - Multiply by fft_normalization_factor, needed to
                #   normalize the grid values after a forwards and a
                #   backwards Fourier transformation.
                slab_jik[0] *= ℝ[potential_factor*double_deconv*fft_normalization_factor]
                slab_jik[1] *= ℝ[potential_factor*double_deconv*fft_normalization_factor]
    # Fourier transform the slabs back to coordinate space.
    # Now the slabs store potential values.
    fft(slab, 'backward')
    # Communicate the potential stored in the slabs to φ
    domain_decompose(slab, φ)  # This also populates pseudo and ghost points
    # Return the potential grid (though this is a global and is often
    # imported directly into other modules).
    return φ

# Generic function implementing particle-mesh interactions
# for both particle and fluid componenets.
@cython.header(# Arguments
               receivers=list,
               suppliers=list,
               ᔑdt=dict,
               potential=func_potential,
               potential_name=str,
               dependent=list,
               apply_potential=func_apply_potential,
               # Locals
               component='Component',
               components=list,
               dim='int',
               gradφ_dim='double[:, :, ::1]',
               h='double',
               representation=str,
               φ='double[:, :, ::1]',
               φ_dict=dict,
               )
def particle_mesh_general(receivers, suppliers, ᔑdt, potential, potential_name,
                          dependent, apply_potential):
    """This function will update the affected variables of all receiver
    components due to an interaction. This is done by constructing
    global fields by interpolating the dependent variables of all
    receivers and suppliers onto grids.
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
    argument, see the construct_potential_general function.

    The grids are then Fourier transformed back to real space and
    differentiated along each dimension to get the force. This force is
    passed to the apply_potential function for each receiver and
    each dimension.
    """
    # Build the two potentials due to all particles and fluid components
    components = receivers + suppliers
    masterprint('Constructing the {} due to {} ...'
                .format(potential_name, ', '.join([component.name for component in components])))
    φ_dict = construct_potential_general(receivers, suppliers, dependent, potential)
    masterprint('done')
    # For each dimension, differentiate the potentials
    # and apply the force to all receiver components.
    h = boxsize/φ_gridsize  # Physical grid spacing of φ
    for representation, φ in φ_dict.items():
        for dim in range(3):
            masterprint(f'Differentiating the ({representation}) {potential_name} along the '
                        f'{"xyz"[dim]}-direction and applying it ...'
                        )
            # Do the differentiation of φ
            gradφ_dim = diff_domain(φ, dim, h, order=4)
            # Apply force to all the receivers
            for component in receivers:
                if component.representation != representation:
                    continue
                masterprint(f'Applying to {component.name} ...')
                apply_potential(component, ᔑdt, gradφ_dim, dim)
                masterprint('done')
            masterprint('done')

# Generic function capable of constructing potential grids out of
# components and a given expression for the potential.
@cython.header(# Arguments
               receivers=list,
               suppliers=list,
               quantities=list,
               potential=func_potential,
               # Locals
               any_fluid='bint',
               any_fluid_receivers='bint',
               any_particles='bint',
               any_particles_receivers='bint',
               components=list,
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
               receiver_representations=list,
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
               slab_ordereddict=object,  # OrderedDict
               slab_particles='double[:, :, ::1]',
               slab_particles_jik='double*',
               φ='double[:, :, ::1]',
               φ_dict=dict,
               returns=dict,
               )
def construct_potential_general(receivers, suppliers, quantities, potential):
    """This function populate two grids (including pseudo points and
    ghost layers) with a real-space potential corresponding to the
    Fourier-space potential function given, due to all the components.
    A seperate grid for particle and fluid components will be
    constructed, the difference being only the handling of
    deconvolutions needed for the interpolation two/from the grid.
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
    The two grids are now Fourirer transformed back to real space.

    In the case of normal gravity, we have
    φ(k) = -4πGa²ρ(k)/k² = -4πG a**(-3*w_eff - 1) ϱ(k)/k²,
    which can be signalled by passing
    quantities = [('particles', a**(-1)*mass/Vcell),
                  ('ϱ', a**(-3*w_eff - 1))],
    potential = lambda k2: -4*π*G_Newton/k2
    (note that it is not actally allowed to pass an untyped lambda
    function in compiled mode).
    """
    # CIC interpolate the particles/fluid elements onto the grids.
    # The φ_dict will be a dictionary mapping representations
    # ('particles', 'fluid') to grids. If only one representation is
    # present, only this item will exist in the dictionary.
    components = receivers + suppliers
    φ_dict = CIC_components2φ_general(components, quantities)
    # Flags specifying whether any fluid/particle components are present
    any_particles = ('particles' in φ_dict)
    any_fluid     = ('fluid'     in φ_dict)
    # Slab decompose the grids
    slab_dict = {representation: slab_decompose(φ, f'φ_{representation}_slab', prepare_fft=True)
                 for representation, φ in φ_dict.items()}
    # In the case of both particle and fluid components being present,
    # it is important that the particle slabs are handled after the
    # fluid slabs, as the deconvolution factor is only computed for
    # particle components and this is needed after combining the fluid
    # and particle slabs.
    slab_ordereddict = collections.OrderedDict()
    if any_fluid:
        slab_fluid = slab_dict['fluid']
        slab_ordereddict['fluid'] = slab_fluid
    if any_particles:
        slab_particles = slab_dict['particles']
        slab_ordereddict['particles'] = slab_particles
    # Do a forward in-place Fourier transform of the slabs
    for slab in slab_dict.values():
        fft(slab, 'forward')
    # Multiplicative factor needed after a forward and a backward
    # Fourier transformation.
    fft_normalization_factor = float(φ_gridsize)**(-3)
    # For each grid, multiply by the potential and deconvolution
    # factors. Do fluid slabs fist, then particle slabs.
    for representation_counter, (representation, slab) in enumerate(slab_ordereddict.items()):
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
                if i > φ_gridsize/2:
                    ki = i - φ_gridsize
                else:
                    ki = i
                # Reciprocal square root of the product of the i-
                # and the j-component of the deconvolution.
                with unswitch(2):
                    if 𝔹[representation == 'particles']:
                        reciprocal_sqrt_deconv_ij = (sinc(ki*ℝ[π/φ_gridsize])
                                                     *reciprocal_sqrt_deconv_j)
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
                    potential_factor = potential(ℝ[(2*π/boxsize)**2]*k2)
                    # The final deconvolution factor
                    with unswitch(3):
                        if 𝔹[representation == 'particles']:
                            # Reciprocal square root of the product of
                            # all components of the deconvolution.
                            reciprocal_sqrt_deconv_ijk = (reciprocal_sqrt_deconv_ij
                                                          *sinc(kk*ℝ[π/φ_gridsize]))
                            # The total factor
                            # for a complete deconvolution.
                            deconv_ijk = 1/reciprocal_sqrt_deconv_ijk**2
                            # A deconvolution of the particle potential
                            # is needed due to the interpolation from
                            # the particle positions to the grid.
                            deconv_factor = deconv_ijk
                            # For particle components we will need to do
                            # a second deconvolutions due to the
                            # interpolation from the grid back to the
                            # particles. We carry out this second
                            # deconvolution now if we only have particle
                            # components. If both particle and fluid
                            # components are present, this second
                            # deconvolution will take place later.
                            with unswitch(4):
                                if not any_fluid:
                                    deconv_factor *= deconv_ijk
                        elif 𝔹[representation == 'fluid']:
                            # Do not apply any deconvolution to fluids
                            deconv_factor = 1
                    # Transform this complex grid point
                    slab_jik[0] *= ℝ[potential_factor*deconv_factor  # Real part
                                     *fft_normalization_factor]
                    slab_jik[1] *= ℝ[potential_factor*deconv_factor  # Imag part
                                     *fft_normalization_factor]
                    # If only particle components or only fluid
                    # components exist, the slabs now store the final
                    # potential in Fourier space. However, if both
                    # particle and fluid components exist, the two sets
                    # of slabs should be combined to form total
                    # potentials. We know that both representations
                    # exist and that we are done handling both (at this
                    # gridpoint) if representation_counter == 1.
                    with unswitch(3):
                        if representation_counter == 1:
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
    # In the general case of both particles and fluids taking part in
    # the creation of the potential, slab_dict now contains slab-
    # decomposed potential grids in Fourier space, one for particles and
    # one for fluids. Importantly, both of these are the total potential
    # due to both particles and fluids. The difference is solely in the
    # amount of deconvolution. Now, though both particles and fluids
    # contribute to the potential, they may not both receive forces due
    # to it. If this is the case, we now through away
    # this superfluous potential.
    receiver_representations = [receiver.representation for receiver in receivers]
    any_particles_receivers = ('particles' in receiver_representations)
    any_fluid_receivers     = ('fluid'     in receiver_representations)
    if any_particles and not any_particles_receivers:
        del slab_dict['particles']
        del φ_dict   ['particles']
    if any_fluid and not any_fluid_receivers:
        del slab_dict['fluid']
        del φ_dict   ['fluid']
    if not slab_dict:
        abort(
            'Something went wrong in the construct_potential_general() function, '
            'as it appears that neither particles nor fluids should receive the force '
            'due to the potential'
        )
    # Fourier transform the slabs back to coordinate space
    for slab in slab_dict.values():
        fft(slab, 'backward')
    # Domain-decompose the slabs
    for φ, slab in zip(φ_dict.values(), slab_dict.values()):
        domain_decompose(slab, φ)  # Also populates pseudos and ghost
    # Return the potential grids
    return φ_dict

# Function implementing pairwise nearest neighbour search
@cython.header(# Arguments
               component_1='Component',
               component_2='Component',
               rank_2='int',
               ᔑdt=dict,
               local='bint',
               mutual='bint',
               extra_args=dict,
               # Locals
               N_2='Py_ssize_t',
               i='Py_ssize_t',
               index='Py_ssize_t',
               j='Py_ssize_t',
               neighbour_components=list,
               neighbour_distances2='double[::1]',
               neighbour_indices='Py_ssize_t[::1]',
               neighbour_ranks='int[::1]',
               posx_1='double*',
               posx_2='double*',
               posy_1='double*',
               posy_2='double*',
               posz_1='double*',
               posz_2='double*',
               r2='double',
               r2_min='double',
               selected=dict,
               selected_indices='Py_ssize_t[::1]',
               x_ji='double',
               xi='double',
               y_ji='double',
               yi='double',
               z_ji='double',
               zi='double',
               returns='void',
               )
def find_nearest_neighbour_pairwise(component_1, component_2, rank_2, ᔑdt, local, mutual, extra_args):
    if component_1.representation != 'particles' or component_2.representation != 'particles':
        abort('find_nearest_neighbour_pairwise is only implemented for particle components')
    # Extract extra arguments
    selected = extra_args['selected']
    if not component_1 in selected:
        return
    selected_indices = selected[component_1]
    neighbour_components = extra_args['neighbour_components'][component_1]
    neighbour_indices = extra_args['neighbour_indices'][component_1]
    neighbour_ranks = extra_args['neighbour_ranks'][component_1]
    neighbour_distances2 = extra_args['neighbour_distances2'][component_1]
    # Extract variables from the first (the local) component
    posx_1 = component_1.posx
    posy_1 = component_1.posy
    posz_1 = component_1.posz
    # Extract variables from the second (the external) component
    N_2 = component_2.N_local
    posx_2 = component_2.posx
    posy_2 = component_2.posy
    posz_2 = component_2.posz
    # Loop over the selected particles
    for i in range(selected_indices.shape[0]):
        index = selected_indices[i]
        r2_min = neighbour_distances2[i] if neighbour_components[i] else ထ
        xi = posx_1[index]
        yi = posy_1[index]
        zi = posz_1[index]
        # Loop over all particles in the external component
        for j in range(N_2):
            # "Vector" from particle j to particle i
            x_ji = xi - posx_2[j]
            y_ji = yi - posy_2[j]
            z_ji = zi - posz_2[j]
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
            # Squared distance
            r2 = x_ji**2 + y_ji**2 + z_ji**2
            if r2 < r2_min and not (i == j and local):
                # New neighbour candidate found
                r2_min = r2
                neighbour_components[i] = component_2.name
                neighbour_indices[i] = j
                neighbour_ranks[i] = rank_2
                neighbour_distances2[i] = r2

# Function that finds the nearest neighbour particles
# to a selected subset of particle components.
@cython.header(# Arguments
               components=list,
               selected=dict,
               # Locals
               component='Component',
               indices='Py_ssize_t[::1]',
               neighbour_components=dict,
               neighbour_distances2=dict,
               neighbour_indices=dict,
               neighbour_ranks=dict,
               returns=tuple,
               )
def find_nearest_neighbour(components, selected):
    """Here selected is a dict with Component instances as keys
    and corresponding particle indices (type Py_ssize_t[::1]) as values.
    These indices are the selected particles which neighbours should
    be found among all the given components.
    """
    neighbour_components = {component: ['']*indices.shape[0]
                            for component, indices in selected.items()}
    neighbour_ranks      = {component: zeros(indices.shape[0], dtype=C2np['int'])
                            for component, indices in selected.items()}
    neighbour_indices    = {component: zeros(indices.shape[0], dtype=C2np['Py_ssize_t'])
                            for component, indices in selected.items()}
    neighbour_distances2 = {component: zeros(indices.shape[0], dtype=C2np['double'])
                            for component, indices in selected.items()}
    domain_domain(components, [], {}, find_nearest_neighbour_pairwise, '',
                  dependent=['pos'], affected=[], deterministic=True,
                  extra_args={'selected': selected,
                              'neighbour_components': neighbour_components,
                              'neighbour_indices': neighbour_indices,
                              'neighbour_ranks': neighbour_ranks,
                              'neighbour_distances2': neighbour_distances2,
                              },
                  )
    return neighbour_components, neighbour_ranks, neighbour_indices, neighbour_distances2

# Function that carry out the gravitational interaction
@cython.pheader(# Arguments
                method=str,
                receivers=list,
                suppliers=list,
                ᔑdt=dict,
                # Locals
                component='Component',
                components=list,
                dependent=list,
                i='Py_ssize_t',
                Δt='double',
                φ_Vcell='double',
                # DELETE BELOW WHEN DONE WITH gravity_old.py !!!
                dim='int',
                gradφ_dim='double[:, :, ::1]',
                h='double',
                φ='double[:, :, ::1]',
                )
def gravity(method, receivers, suppliers, ᔑdt):
    # Regardless of the method, it may happen that some fluid components
    # classified as receivers are incapable of receiving the
    # gravitational force due to the lack of the non-linear J
    # fluid variable. In this case, it becomes a supplier in stead.
    for i, component in enumerate(receivers.copy()):
        if component.representation == 'fluid' and component.is_linear(1):
            suppliers.append(component)
            del receivers[i]
    # If no receivers exist at all, no interaction should take place
    if not receivers:
        return
    # List of all particles participating in this interaction
    components = receivers + suppliers
    # Compute gravity via one of the following methods
    if method == 'ppnonperiodic':
        # The non-periodic particle-particle method
        domain_domain(receivers, suppliers, ᔑdt, gravity_pairwise, 'gravitation (PP (non-periodic))',
                      dependent=['pos'], affected=['mom'], deterministic=True,
                      extra_args={'periodic'        : False,
                                  'only_short_range': False,
                                  },
                      )
    elif method == 'pp':
        # The particle-particle method with Ewald-periodicity
        domain_domain(receivers, suppliers, ᔑdt, gravity_pairwise, 'gravitation (PP)',
                      dependent=['pos'], affected=['mom'], deterministic=True,
                      extra_args={'periodic'        : True,
                                  'only_short_range': False,
                                  },
                      )
    elif method == 'pm':
        # The particle-mesh method.
        # The gravitational potential is given by the Poisson equation
        # ∇²φ = 4πGa²ρ = 4πGa**(-3*w_eff - 1)ϱ.
        # The factor in front of the dependent variable ϱ is thus
        # time-varying and component-dependent. Here we use the mean
        # values over the current time step.
        φ_Vcell = ℝ[(boxsize/φ_gridsize)**3]
        Δt = ᔑdt['1']
        dependent = [# Particle components
                     ('particles', [ℝ[ᔑdt['a**(-1)']/(Δt*φ_Vcell)]*component.mass
                                    for component in components]),
                     # Fluid components
                     ('ϱ', [ᔑdt['a**(-3*w_eff-1)', component]*ℝ[1/Δt]
                            for component in components]),
                     ]
        particle_mesh_general(receivers, suppliers, ᔑdt, gravity_potential, 'gravitational potential (PM)',
                              dependent, apply_gravity_potential)
    elif method == 'p3m':
        # List of all particles participating in this interaction
        components = receivers + suppliers
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
               components=list,
               # Locals
               component='Component',
               force=str,
               forces_in_use=object,  # collections.defaultdict
               interactions_list=list,
               method=str,
               methods=list,
               methods_implemented=list,
               receivers=list,
               suppliers=list,
               returns=list,
               )
def find_interactions(components):
    # Find all (force, method) pairs in use. Store these as a (default)
    # dict mapping forces to lists of methods.
    forces_in_use = collections.defaultdict(list)
    for component in components:
        for force, method in component.forces.items():
            forces_in_use[force].append(method)
    # Check that all forces and methods assigned
    # to the components are implemented.
    for force, methods in forces_in_use.items():
        methods_implemented = forces_implemented.get(force, [])
        for method in methods:
            if method not in methods_implemented:
                abort(f'Method "{method}" for force "{force}" is not implemented')
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
    for force, method in forces_implemented_ordered:
        if method not in forces_in_use.get(force, []):
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
# are computed and applied.
# Importantly, all forces and methods should be written with purely
# alphanumeric, lowercase characters.
cython.declare(forces_implemented_ordered=list)
forces_implemented_ordered = [
    ('gravity', 'ppnonperiodic'),
    ('gravity', 'pp'           ),
    ('gravity', 'p3m'          ),
    ('gravity', 'pm'           ),
]
# Non-ordered version of forces_implemented_ordered, implemented as a
# (default) dict mapping forces to list of methods.
forces_implemented = collections.defaultdict(list)
for force, method in forces_implemented_ordered:
    forces_implemented[force].append(method)
