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
    'communicate_domain, sendrecv_component, rank_neighbouring_domain')
cimport('from communication import domain_size_x , domain_size_y , domain_size_z' )
cimport('from communication import domain_start_x, domain_start_y, domain_start_z')
cimport('from mesh import diff_domain, domain_decompose, fft, slab_decompose')
cimport('from mesh import CIC_components2φ, CIC_grid2grid, CIC_scalargrid2coordinates')
# Import interactions defined in other modules
cimport('from gravity import *')

# Function pointer types used in this module
pxd("""
#                                   component_1, component_2, rank_2, ᔑdt , local, mutual, extra_args
ctypedef void   (*func_interaction)(Component  , Component  , int   , dict, bint , bint  , dict      )
#                                   k2
ctypedef double (*func_potential  )(double)
""")



# Generic function implementing domain-domain pairing
@cython.header(
    # Arguments
    receivers=list,
    suppliers=list,
    ᔑdt=dict,
    interaction=func_interaction,
    dependent=list,  # list of str's
    affected=list,   # list of str's
    deterministic='bint',
    extra_args=dict,
    communication_pattern=str,
    # Locals
    assisted='bint',
    component_pair=set,
    component_r='Component',
    component_s_extrl='Component',
    component_s_local='Component',
    i='Py_ssize_t',
    local='bint',
    mutual='bint',
    only_supply='bint',
    pairings=list,
    rank_recv='int',
    rank_send='int',
    ranks_recv='int[::1]',
    ranks_send='int[::1]',
    synchronous='bint',
)
def domain_domain(
    receivers, suppliers, ᔑdt, interaction, dependent, affected, deterministic,
    communication_pattern, extra_args={},
):
    """This function takes care of pairings between all receiver and
    supplier components.
    The receiver components are denoted component_r. These will not be
    communicated. The supplier components will be denoted component_s.
    These will be sent to other processes (domains) and also received
    back from other processes. Thus both local and external versions of
    component_s exist, called component_s_local and component_s_extrl.

    If affected is an empty list, this is not really an interaction.
    In this case, every domain will both send and receive from every
    other domain.
    """
    # Determine whether this "interaction" have any direct effect
    # on the components at all. If not, this will change the
    # communication pattern.
    assisted = bool(affected)
    # Get the process ranks to send to and receive from
    ranks_send, ranks_recv = domain_domain_communication(communication_pattern, assisted)
    # Pair each receiver with all suppliers
    pairings = []
    for component_r in receivers:
        for component_s_local in suppliers:
            component_pair = {component_r, component_s_local}
            if component_pair in pairings:
                continue
            pairings.append(component_pair)
            # Flag specifying whether component_s should only supply
            # forces to component_1 and not receive any momentum
            # updates itself.
            only_supply = (component_s_local not in receivers)
            # Display progress message
            if only_supply:
                masterprint(f'Pairing {component_r.name} ⟵ {component_s_local.name} ...')
            else:
                masterprint(f'Pairing {component_r.name} ⟷ {component_s_local.name} ...')
            # Pair this process/domain with whichever other
            # processes/domains are needed. This process is paired
            # with two other processes simultaneously. This rank sends
            # a copy of the local component_s to rank_send, while
            # receiving the external component_s from rank_recv.
            # On each process, the local component_r and the external
            # (received) component_s then interact.
            for i in range(ranks_send.shape[0]):
                # Process ranks to send to and receive from
                rank_send = ranks_send[i]
                rank_recv = ranks_recv[i]
                fancyprint(f'rank {rank} sends to {rank_send} and recv from {rank_recv}')
                # Determine whther component_s should be updated due to
                # its interaction with component_r. This is usually the
                # case. The exceptions are
                # - component_s is a supplier and not a receiver.
                #   When this is the case, only_supply is True.
                # - component_s is really the same as component_r
                #   and this is a local interaction, meaning that
                #   both rank_send and rank_recv is really just the
                #   local rank. In this case, the supplied interaction
                #   function should also update the data of component_s
                #   (not the buffers). This special case is flagged
                #   by the 'local' variable being True.
                # - component_s is really the same as component_r
                #   and rank_send == rank_recv (but different from the
                #   local rank). In this case, the external updates to
                #   component_s should not be send back and applied, as
                #   these updates are already done locally on the other
                #   process. This special case is flagged by the
                #   'synchronous' variable being True. The exception
                #   (to this exception) is when the interaction is non-
                #   deterministic, in which case this (supposedly
                #   identical) interaction must not be computed by both
                #   processes synchronously, as they may produce
                #   different results.
                local = False
                if 𝔹[component_r is component_s_local] and rank_send == rank == rank_recv:
                    local = True
                synchronous = False
                if not local and 𝔹[component_r is component_s_local] and rank_send == rank_recv:
                    synchronous = True
                mutual = True
                if only_supply or local or (synchronous and deterministic) or not assisted:
                    mutual = False
                # Communicate the dependent variables
                # (e.g. pos for gravity) of component_s.
                component_s_extrl = sendrecv_component(
                    component_s_local, dependent, dest=rank_send, source=rank_recv,
                )
                # Let the local component_r interaction with the
                # external component_s. This will update the affected
                # variables (e.g. mom for gravity) of the local
                # component_r and populate the affected variable buffers
                # of the external component_s, if mutual is True.
                # If this is a synchronous interaction (meaning that it
                # is between only two processes, rank_send == rank_recv)
                # and also non-deterministic, perform the interaction
                # only on one of the two processes. The process with
                # the lower rank is chosen for the job.
                if (    not synchronous
                    or (    synchronous and     deterministic)
                    or (    synchronous and not deterministic and rank < rank_send)
                ):
                    interaction(
                        component_r, component_s_extrl, rank_recv, ᔑdt, local, mutual, extra_args,
                    )
                if mutual:
                    # Send the populated buffers back to the process
                    # from which the external component_s came.
                    # Add the received values in the buffers to the
                    # affected variables (e.g. mom for gravity) of
                    # the local component_s.
                    sendrecv_component(
                        component_s_extrl, affected,
                        dest=rank_recv, source=rank_send, component_recv=component_s_local,
                    )
                    # Nullify the Δ buffers of the external component_s,
                    # leaving this with no leftover junk.
                    component_s_extrl.nullify_Δ(affected)
            masterprint('done')
# Function returning iterators over domains/processes with which to
# pair the local domain in the domain_domain function,
# depending on the communication_pattern.
@cython.pheader(
    # Arguments
    communication_pattern=str,
    assisted='bint',
    # Locals
    i='Py_ssize_t',
    returns=tuple,
)
def domain_domain_communication(communication_pattern, assisted):
    ranks = domain_domain_communication_dict[communication_pattern].get(assisted)
    if ranks:
        return ranks
    if communication_pattern == 'pp':
        # Each process should be paired with all processes. In the usual
        # case of assisted being True, a process sends and receivers
        # from two processes simultaneously, meaning that the number of
        # pairings is cut (roughly) in half.
        N_domain_pairs = 1 + nprocs//2 if assisted else nprocs
        ranks_send = np.empty(N_domain_pairs, dtype=C2np['int'])
        ranks_recv = np.empty(N_domain_pairs, dtype=C2np['int'])
        for i in range(N_domain_pairs):
            ranks_send[i] = mod(rank + i, nprocs)
            ranks_recv[i] = mod(rank - i, nprocs)
        domain_domain_communication_dict[communication_pattern][assisted] = (
            (ranks_send, ranks_recv)
        )
    elif communication_pattern == 'p3m':
        ...
    else:
        abort(
            f'domain_domain_communication() got '
            f'communication_pattern = {communication_pattern} ∉ {{"pp", "p3m"}}'
        )
    return domain_domain_communication_dict[communication_pattern][assisted]
# Cached results of the domain_domain_communication function
# are stored in the dict below.
cython.declare(domain_domain_communication_dict=dict)
domain_domain_communication_dict = {'pp': {}, 'p3m': {}}

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
        f'Constructing the {potential_name} due to {{}} ...'
        .format(', '.join([component.name for component in suppliers]))
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
                    ᔑdt_key = (ᔑdt_key[0], component)
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
    domain_domain(
        components, [], {}, find_nearest_neighbour_pairwise,
        dependent=['pos'], affected=[], deterministic=True, communication_pattern='pp',
        extra_args={
            'selected'            : selected,
            'neighbour_components': neighbour_components,
            'neighbour_indices'   : neighbour_indices,
            'neighbour_ranks'     : neighbour_ranks,
            'neighbour_distances2': neighbour_distances2,
        },
    )
    return neighbour_components, neighbour_ranks, neighbour_indices, neighbour_distances2

# Function that carry out the gravitational interaction
@cython.pheader(
    # Arguments
    method=str,
    receivers=list,
    suppliers=list,
    ᔑdt=dict,
    pm_potential=str,
    # Locals
    dependent=list,
    potential=func_potential,
    potential_name=str,
    Δt='double',
    φ_Vcell='double',
    ᔑdt_key=object,  # str or tuple
)
def gravity(method, receivers, suppliers, ᔑdt, pm_potential='full'):
    # Compute gravity via one of the following methods
    if method == 'ppnonperiodic':
        # The non-periodic particle-particle method
        masterprint(
            f'Executing gravitational interaction for {{}} via the non-periodic PP method ...'
            .format(', '.join([component.name for component in receivers]))
        )
        domain_domain(
            receivers, suppliers, ᔑdt, gravity_pairwise,
            dependent=['pos'], affected=['mom'], deterministic=True, communication_pattern='pp',
            extra_args={
                'periodic'        : False,
                'only_short_range': False,
            },
        )
        masterprint('done')
    elif method == 'pp':
        # The particle-particle method with Ewald-periodicity
        masterprint(
            f'Executing gravitational interaction for {{}} via the PP method ...'
            .format(', '.join([component.name for component in receivers]))
        )
        domain_domain(
            receivers, suppliers, ᔑdt, gravity_pairwise,
            dependent=['pos'], affected=['mom'], deterministic=True, communication_pattern='pp',
            extra_args={
                'periodic'        : True,
                'only_short_range': False,
            },
        )
        masterprint('done')
    elif method == 'pm':
        # The particle-mesh method.
        if pm_potential == 'full':
            # Use the full gravitational potential
            masterprint(
                f'Executing gravitational interaction for {{}} via the PM method ...'
                .format(', '.join([component.name for component in receivers]))
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
                ᔑdt['a**(-3*w_eff-1)', component]*component.mass*ℝ[1/(Δt*φ_Vcell)]
                for component in suppliers]
            ),
            # Fluid components
            ('ϱ', [
                ᔑdt['a**(-3*w_eff-1)', component]*ℝ[1/Δt]
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
            masterprint('done')
    elif method == 'p3m':
        # The particle-particle-mesh method
        masterprint(
            f'Executing gravitational interaction for {{}} via the P³M method ...'
            .format(', '.join([component.name for component in receivers]))
        )
        # The long-range PM part
        gravity('pm', receivers, suppliers, ᔑdt, 'long-range only')
        # The short-range PP part
        masterprint('Applying direct short-range forces ...')
        domain_domain(
            receivers, suppliers, ᔑdt, gravity_pairwise,
            dependent=['pos'], affected=['mom'], deterministic=True, communication_pattern='pp',
            extra_args={'only_short_range': True},
        )
        masterprint('done')
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
    # Locals
    dependent=list,
    ᔑdt_key=object,  # str or tuple
)
def lapse(method, receivers, suppliers, ᔑdt):
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
        masterprint(
            f'Executing lapse interaction for {{}} via the PM method ...'
            .format(', '.join([component.name for component in receivers]))
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
                ᔑdt['a**(-3*w_eff-1)', component]/ᔑdt['1']
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
        masterprint('done')
    elif master:
        abort(f'lapse() was called with the "{method}" method')

# Function that given lists of receiver and supplier components of a
# one-way interaction removes any components from the supplier list that
# are also present in the receiver list.
def oneway_force(receivers, suppliers):
    return [component for component in suppliers if component not in receivers]

# Function which constructs a list of interactions from a list of
# components. The list of interactions store information about which
# components interact with one another, via what force and method.
def find_interactions(components):
    # Use cached result
    interactions_list = interactions_lists.get(tuple(components))
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
    # interacting under some force use the same method, the suppliers
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
                    interactions_list = (
                          interactions_list[:i+1]
                        + [Interaction(interaction.force, 'pm', interaction.receivers, [component])]
                        + interactions_list[i+1:]
                    )
                    return True
        # Remove interactions with no suppliers or no receivers
        interactions_list = [interaction for interaction in interactions_list
            if interaction.receivers and interaction.suppliers]
        # Merge interactions of identical force, method and receivers
        # but different suppliers, or identical force, method and suppliers
        # but different receivers.
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
    # Cache the result and return it
    interactions_lists[tuple(components)] = interactions_list
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
