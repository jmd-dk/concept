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
cimport('from mesh import diff_domain')



# Entry point for the MacCormack method,
# which does time evolution of a fluid component.
@cython.header(# Arguments
               component='Component',
               ᔑdt='dict',
               # Locals
               attempt='int',
               i='Py_ssize_t',
               max_vacuum_corrections='int[::1]',
               mc_step='int',
               steps='Py_ssize_t[::1]',
               )
def maccormack(component, ᔑdt):
    # If only a single fluid variable (ϱ) exist,
    # it only evolve through source terms.
    if component.N_fluidvars == 1:
        return
    # Maximum allowed number of attempts to correct for
    # negative densities, for the first and second MacCormack step.
    max_vacuum_corrections = asarray([1, component.gridsize], dtype=C2np['int'])
    # Step/flux directions for the first MacCormack step
    steps = next(maccormack_steps)
    # The two MacCormack steps
    for mc_step in range(2):
        # Evolve the fluid variables. If this leads to negative
        # densities, attempts are made to correct this.
        for attempt in range(max_vacuum_corrections[mc_step]):
            # Evolve fluid variables. In the first MacCormack step,
            # the variables are re-evolved at each attempt. In the
            # second MacCormack step, the variables should only be
            # evolved once (vacuum correction may still take place
            # multiple times).
            if attempt == 0 or mc_step == 0:
                # Compute starred variables from unstarred variables
                # (first MacCormack step) or vice versa
                # (second MacCormack step).
                evolve_fluid(component, ᔑdt, steps, mc_step)
            # Do vacuum corrections if toogled for this species.
            # If not, check but du not correct for vacuum.
            if 𝔹[vacuum_corrections.get('all') or vacuum_corrections.get(component.name)]:
                # Nullify the Δ buffers, so that they are ready to
                # be used by the following vacuum correction sweep.
                component.nullify_Δ()
                # Check and correct for density values heading dangerously
                # fast towards negative values. If every density value
                # is OK, accept this attempt at a MacCormack step as is.
                if not correct_vacuum(component, mc_step):
                    break
            else:
                check_vacuum(component, mc_step)
                break
        else:
            # None of the attempted MacCormack steps were accepted.
            # If this is the second MacCormack step, this means that
            # we have been unable to correct for negative densities.
            if mc_step == 1:
                abort('Giving up after {} failed attempts to remove negative densities in "{}"'
                      .format(max_vacuum_corrections[mc_step], component.name))
        # Reverse step direction for the second MacCormack step
        for i in range(3):
            steps[i] *= -1
    # The two MacCormack steps leave all values of all fluid variables
    # with double their actual values. All grid values thus need
    # to be halved. Note that no further communication is needed as we
    # also halve the pseudo and ghost points.
    component.scale_fluid_grid(0.5)
    # Nullify the starred grid buffers and the Δ buffers,
    # leaving these with no leftover junk.
    component.nullify_fluid_gridˣ()
    component.nullify_Δ()

# Infinite generator cycling through the 8 triples of
# step/flux directions, used in the maccormack function.
def generate_maccormack_steps():
    steps = []
    for sign in (+1, -1):
        steps.append(sign*asarray((+1, +1, +1), dtype=C2np['Py_ssize_t']))
        steps.append(sign*asarray((-1, +1, -1), dtype=C2np['Py_ssize_t']))
        steps.append(sign*asarray((-1, -1, +1), dtype=C2np['Py_ssize_t']))
        steps.append(sign*asarray((+1, -1, -1), dtype=C2np['Py_ssize_t']))
    yield from itertools.cycle(steps)
maccormack_steps = generate_maccormack_steps()

# Function which evolve the fluid variables of a component,
# disregarding all source terms.
@cython.header(# Arguments
               component='Component',
               ᔑdt='dict',
               steps='Py_ssize_t[::1]',
               mc_step='int',
               # Locals
               J_div='double[:, :, ::1]',
               J_el='double[:, :, ::1]',
               Jˣ_el='double[:, :, ::1]',
               dim_div='int',
               dim_el='int',
               fluidscalar='FluidScalar',
               grid='double*',
               gridˣ='double*',
               h='double',
               indices_local_end='Py_ssize_t[::1]',
               indices_local_start='Py_ssize_t[::1]',
               multi_index='tuple',
               multi_index_list='list',
               view='str',
               viewˣ='str',
               Δ='double',
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               step_i='Py_ssize_t',
               step_j='Py_ssize_t',
               step_k='Py_ssize_t',
               σ_multi_index='double[:, :, ::1]',
               ϱ='double[:, :, ::1]',
               ϱˣ='double[:, :, ::1]',
               )
def evolve_fluid(component, ᔑdt, steps, mc_step):
    """It is assumed that the unstarred and starred grids have
    correctly populated pseudo and ghost points.
    """
    # Comoving grid spacing
    h = boxsize/component.gridsize
    # Arrays of start and end indices for the local part of the
    # fluid grids, meaning disregarding pseudo points and ghost points.
    # We have 2 ghost points in the beginning and 1 pseudo point and
    # 2 ghost points in the end.
    indices_local_start = asarray((2, 2, 2)      , dtype=C2np['Py_ssize_t'])
    indices_local_end   = asarray(component.shape, dtype=C2np['Py_ssize_t']) - 2 - 1
    # At the beginning of the first MacCormack step, the starred buffers
    # should contain a copy of the actual (unstarred) data.
    # At the beginning of the second MacCormack step, the unstarred
    # variables should be updated by adding to them the values in the
    # starred buffers.
    for fluidscalar in component.iterate_fluidscalars(include_disguised_scalar=False):
        grid  = fluidscalar.grid
        gridˣ = fluidscalar.gridˣ
        for i in range(component.size):
            with unswitch:
                if mc_step == 0:
                    gridˣ[i] = grid[i]
                else:  # mc_step == 1
                    grid[i] += gridˣ[i]
    # Attribute names of the data in fluidscalars.
    # In the second MacCormack step, the roles of the
    # starred and the unstarred grids should be swapped.
    view  = 'grid_mv'
    viewˣ = 'gridˣ_mv'
    if mc_step == 1:
        view, viewˣ = viewˣ, view
    # The continuity equation (flux terms only).
    # Δϱ = - ᔑa³ʷ⁻²(1 + w)dt ∇·J    (energy flux)
    # The energy flux
    ϱ  = getattr(component.ϱ, view )
    ϱˣ = getattr(component.ϱ, viewˣ)
    for (dim_div, ), J_div in component.J.iterate(view, multi_indices=True):
        step_i = steps[dim_div] if dim_div == 0 else 0
        step_j = steps[dim_div] if dim_div == 1 else 0
        step_k = steps[dim_div] if dim_div == 2 else 0
        for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
            for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
                for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                    Δ = ℤ[steps[dim_div]]*(  J_div[i + step_i, j + step_j, k + step_k]
                                           - J_div[i         , j         , k         ])
                    ϱˣ[i, j, k] += Δ*ℝ[-ᔑdt['a³ʷ⁻²(1+w)', component]/h]
    # The Euler equation (flux terms only).
    # ΔJᵢ = - c²ᔑa⁻³ʷw/(1 + w)dt (∇ϱ)ᵢ    (pressure term)
    #       - ᔑa³ʷ⁻²dt ∇·(Jᵢ/ϱ J)         (momentum flux)
    #       - ᔑa⁻³ʷdt ∇·(ϱσᵢ)             (stress   flux)
    for dim_el in range(3):  # Loop over elements of J
        J_el  = getattr(component.J[dim_el], view )
        Jˣ_el = getattr(component.J[dim_el], viewˣ)
        # The pressure term
        step_i = steps[dim_el] if dim_el == 0 else 0
        step_j = steps[dim_el] if dim_el == 1 else 0
        step_k = steps[dim_el] if dim_el == 2 else 0
        for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
            for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
                for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                    Δ = ℤ[steps[dim_el]]*(  ϱ[i + step_i, j + step_j, k + step_k]
                                          - ϱ[i         , j         , k         ])
                    Jˣ_el[i, j, k] += Δ*ℝ[-light_speed**2*ᔑdt['a⁻³ʷw/(1+w)', component]/h]
        # The momentum flux
        for dim_div in range(3):  # Loop over dimensions in divergence
            J_div = getattr(component.J[dim_div], view)
            step_i = steps[dim_div] if dim_div == 0 else 0
            step_j = steps[dim_div] if dim_div == 1 else 0
            step_k = steps[dim_div] if dim_div == 2 else 0
            for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
                for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
                    for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                        Δ = ℤ[steps[dim_div]]*(  J_el [i + step_i, j + step_j, k + step_k]
                                                /ϱ    [i + step_i, j + step_j, k + step_k]
                                                *J_div[i + step_i, j + step_j, k + step_k]
                                               - J_el [i         , j         , k         ]
                                                /ϱ    [i         , j         , k         ]
                                                *J_div[i         , j         , k         ])
                        Jˣ_el[i, j, k] += Δ*ℝ[-ᔑdt['a³ʷ⁻²', component]/h]
    if component.N_fluidvars == 2:
        finalize_maccormack_step(component, mc_step)
        return
    # The stress flux
    for multi_index, σ_multi_index in component.σ.iterate(view, multi_indices=True):
        for dim_el in set(multi_index):  # Loop over elements of J affected by σ_multi_index
            Jˣ_el = getattr(component.J[dim_el], viewˣ)
            # The index in multi_index other than the chosen dim_el is
            # the dimension of differentiation by the divergence.
            multi_index_list = list(multi_index)
            multi_index_list.remove(dim_el)
            dim_div = multi_index_list[0]
            step_i = steps[dim_div] if dim_div == 0 else 0
            step_j = steps[dim_div] if dim_div == 1 else 0
            step_k = steps[dim_div] if dim_div == 2 else 0
            for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
                for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
                    for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                        Δ = ℤ[steps[dim_div]]*(  ϱ            [i + step_i, j + step_j, k + step_k]
                                                *σ_multi_index[i + step_i, j + step_j, k + step_k]
                                               - ϱ            [i         , j         , k         ]
                                                *σ_multi_index[i         , j         , k         ])
                        Jˣ_el[i, j, k] += Δ*ℝ[-ᔑdt['a⁻³ʷ', component]/h]
    # The equation for time evolution of stress (flux terms only).
    # Δσᵢⱼ = ???    (??? flux)
    ...
    if component.N_fluidvars == 3:
        finalize_maccormack_step(component, mc_step)
        return

# Function for doing communication of pseudo and ghost points of
# fluid grids after each MacCormack step.
@cython.header(component='Component', mc_step='int')
def finalize_maccormack_step(component, mc_step):
    # Populate the pseudo and ghost points of all fluid variable grids
    # with the updated values. Depedendent on whether we are in the end
    # of the first or the second MacCormack step (mc_step), the updated
    # grids are really the starred grids (first MacCormack step) or the
    # unstarred grids (second MacCormack step).
    if mc_step == 0:
        component.communicate_fluid_gridsˣ(mode='populate')
    else:  # mc_step == 1
        component.communicate_fluid_grids (mode='populate')

# Function which evolve the fluid variables of a component
# due to internal source terms.
@cython.header(# Arguments
               component='Component',
               ᔑdt='dict',
               # Locals
               Jᵢ='FluidScalar',
               Jᵢ_ptr='double*',
               h='double',
               i='Py_ssize_t',
               j='Py_ssize_t',
               multi_index='tuple',
               multi_index_list='list',
               potential='double[:, :, ::1]',
               n='Py_ssize_t',
               source='double[:, :, ::1]',
               source_ptr='double*',
               σᵢⱼ='FluidScalar',
               σᵢⱼ_ptr='double*',
               ϱ_ptr='double*',
               ẇ='double',
               )
def apply_internal_sources(component, ᔑdt):
    """By "internal sources" is meant source terms which do not arise
    due to interactions, such as the pressure term in the time evolution
    for ϱ for a fluid with non-zero w.
    A special kind of such internal source arise when
    component.closure == 'class', in which case one additional fluid
    variable should be realized using CLASS, and then affect its lower
    fluid variable (which will then be the highest dynamic fluid
    variable) through the dynamical fluid equations. The coupling
    between two such fluid variables takes the form of a flux,
    but since one of the variables is not dynamic, here it act just like
    a source term, and should hence be treated as such.
    Because lower fluid variables appear in the source terms of higher
    fluid variables, we need to update the higher fluid variables first.
    """
    ϱ_ptr = component.ϱ.grid
    ẇ = component.ẇ()
    # Physical grid spacing
    h = boxsize/component.gridsize
    # Update σ due to its internal source term
    if component.N_fluidvars >= 3:
        ...
    # Update J due to its internal source term
    # ΔJᵢ = -ẇ/(1+w)Jᵢ
    if component.N_fluidvars >= 2 and ẇ != 0:
        for Jᵢ in component.J.iterate():
            Jᵢ_ptr = Jᵢ.grid
            for n in range(component.size):
                Jᵢ_ptr[n] *= ℝ[1 - ᔑdt['ẇ/(1+w)', component]]
    # If closure of the Boltzmann hierarchy is achieved by continuously
    # realizing σ, do this realization now and update J accordingly.
    # This source term looks like
    # ΔJᵢ = -ᔑa⁻³ʷdt ∇·(ϱσᵢ)
    if component.N_fluidvars == 2 and component.closure == 'class':
        # Loop over all distinct σᵢⱼ and realize them as we go
        for multi_index, σᵢⱼ in component.σ.iterate(multi_indices=True):
            # Transform σᵢⱼ to -ᔑa⁻³ʷdt ϱσᵢⱼ,
            # which is then the potential of the source.
            σᵢⱼ_ptr = σᵢⱼ.grid
            for n in range(component.size):
                σᵢⱼ_ptr[n] *= ℝ[-ᔑdt['a⁻³ʷ', component]]*ϱ_ptr[n]
            potential = σᵢⱼ.grid_mv
            # Loop over elements of J affected by σᵢⱼ
            for i in set(multi_index):
                Jᵢ = component.J[i]
                Jᵢ_ptr = Jᵢ.grid
                # The index in multi_index other than the chosen i is
                # the dimension of differentiation by the divergence, j.
                multi_index_list = list(multi_index)
                multi_index_list.remove(i)
                j = multi_index_list[0]
                # Differentiate the potential and apply the source term
                source = diff_domain(potential, j, h, order=2, noghosts=False)
                source_ptr = cython.address(source[:, :, :])
                for n in range(component.size):
                    Jᵢ_ptr[n] += source_ptr[n]
    # Update ϱ due to its internal source term
    # Δϱ = 3ẇ log(a)ϱ
    if ẇ != 0:
        for n in range(component.size):
            ϱ_ptr[n] *= ℝ[1 + 3*ᔑdt['ẇlog(a)', component]]

# Function which checks and warn about vacuum in a fluid component
@cython.header(# Arguments
               component='Component',
               mc_step='int',
               # Locals
               any_vacuum='bint',
               i='Py_ssize_t',
               ϱ='double*',
               )
def check_vacuum(component, mc_step):
    # Grab pointer to the density. After the first MacCormack step,
    # the starred buffers have been updated from the non-starred
    # buffers, and so it is the starred buffers that should be checked
    # for vacuum values. After the second MacCormack step, the unstarred
    # buffers have been updated from the starred buffers, and so we
    # should check the unstarred buffers.
    if mc_step == 0:
        ϱ = component.ϱ.grid
    else:  # mc_step == 1
        ϱ = component.ϱ.gridˣ
    # Check for vacuum
    any_vacuum = False
    for i in range(component.size):
        if ϱ[i] < ϱ_vacuum:
            any_vacuum = True
            break
    # Show a warning if any vacuum elements were found
    any_vacuum = reduce(any_vacuum, op=MPI.LOR)
    if any_vacuum:
        masterwarn('Vacuum detected in "{}"'.format(component.name))

# Function which checks for imminent vacuum in a fluid component
# and does one sweep of vacuum corrections.
@cython.header(# Arguments
               component='Component',
               mc_step='int',
               # Locals
               Jx='double[:, :, ::1]',
               Jx_correction='double',
               Jx_ptr='double*',
               Jxˣ='double[:, :, ::1]',
               Jy='double[:, :, ::1]',
               Jy_correction='double',
               Jy_ptr='double*',
               Jyˣ='double[:, :, ::1]',
               Jz='double[:, :, ::1]',
               Jz_correction='double',
               Jz_ptr='double*',
               Jzˣ='double[:, :, ::1]',
               dist2='Py_ssize_t',
               fac_smoothing='double',
               fac_time='double',
               i='Py_ssize_t',
               indices_local_start='Py_ssize_t[::1]',
               indices_local_end='Py_ssize_t[::1]',
               j='Py_ssize_t',
               k='Py_ssize_t',
               m='Py_ssize_t',
               mi='Py_ssize_t',
               mj='Py_ssize_t',
               mk='Py_ssize_t',
               n='Py_ssize_t',
               ni='Py_ssize_t',
               nj='Py_ssize_t',
               nk='Py_ssize_t',
               shape='tuple',
               timespan='double',
               vacuum_imminent='bint',
               ΔJx='double[:, :, ::1]',
               ΔJx_ptr='double*',
               ΔJy='double[:, :, ::1]',
               ΔJy_ptr='double*',
               ΔJz='double[:, :, ::1]',
               ΔJz_ptr='double*',
               Δϱ='double[:, :, ::1]',
               Δϱ_ptr='double*',
               ϱ='double[:, :, ::1]',
               ϱ_correction='double',
               ϱ_ijk='double',
               ϱ_ptr='double*',
               ϱˣ='double[:, :, ::1]',
               ϱˣ_ijk='double',
               returns='bint',
               )
def correct_vacuum(component, mc_step):
    """This function will detect and correct for imminent vacuum in a
    fluid component. The vacuum detection is done differently depending
    on the MacCormack step (the passed mc_step). For the first
    MacCormack step, vacuum is considered imminent if a density below
    the vacuum density, ϱ_vacuum, will be reached within 'timespan'
    similiar time steps. For the second MacCormack step, vacuum is
    considered imminent if the density is below the vacuum density.
    The vacuum correction is done by smoothing all fluid variables in
    the 3x3x3 neighbouring cells souronding the vacuum cell.
    The smoothing between each pair of cells, call them (i, j),
    is given by
    ϱi += fac_smoothing*(ϱj - ϱi)/r²,
    ϱj += fac_smoothing*(ϱi - ϱj)/r²,
    (similar for other fluid variables)
    where r is the distance between the cells in grid units.
    Whether or not any vacuum corrections were made is returned
    as the return value.
    Experimentally, it has been found that when
    max_vacuum_corrections[0] == 1,
    the following values give good, stable results:
    timespan = 30
    fac_smoothing = 1.5/(6/1 + 12/2 + 8/3)
    """
    # In the case of the first MacCormack step, consider vacuum to be
    # imminent if a cell will reach the vacuum density after this many
    # similar time steps. Should be at least 1.
    timespan = 30
    # Amount of smoohing to apply when vacuum is detected.
    # A numerator of 0 implies no smoothing.
    # A numerator of 1 implies that in the most extreme case,
    # a vacuum cell will be replaced with a weighted average of its
    # 26 neighbour cells (all of the original cell will be distributed
    # among these neighbors).
    fac_smoothing = ℝ[1.0/(6/1 + 12/2 + 8/3)]
    # Arrays of start and end indices for the local part of the
    # fluid grids, meaning disregarding pseudo points and ghost points.
    # We have 2 ghost points in the beginning and 1 pseudo point and
    # 2 ghost points in the end.
    shape = component.shape
    indices_local_start = asarray([2, 2, 2], dtype=C2np['Py_ssize_t'])
    indices_local_end   = asarray(shape    , dtype=C2np['Py_ssize_t']) - 2 - 1
    # Extract memory views and pointers to the fluid variables
    ϱ       = component.ϱ .grid_mv
    ϱ_ptr   = component.ϱ .grid
    ϱˣ      = component.ϱ .gridˣ_mv
    ϱˣ_ptr  = component.ϱ .gridˣ
    Δϱ      = component.ϱ .Δ_mv
    Δϱ_ptr  = component.ϱ .Δ
    Jx      = component.Jx.grid_mv
    Jx_ptr  = component.Jx.grid
    Jxˣ     = component.Jx.gridˣ_mv
    Jxˣ_ptr = component.Jx.gridˣ
    ΔJx     = component.Jx.Δ_mv
    ΔJx_ptr = component.Jx.Δ
    Jy      = component.Jy.grid_mv
    Jy_ptr  = component.Jy.grid
    Jyˣ     = component.Jy.gridˣ_mv
    Jyˣ_ptr = component.Jy.gridˣ
    ΔJy     = component.Jy.Δ_mv
    ΔJy_ptr = component.Jy.Δ
    Jz      = component.Jz.grid_mv
    Jz_ptr  = component.Jz.grid
    Jzˣ     = component.Jz.gridˣ_mv
    Jzˣ_ptr = component.Jz.gridˣ
    ΔJz     = component.Jz.Δ_mv
    ΔJz_ptr = component.Jz.Δ
    # In the case of the second MacCormack step, the role of the
    # starred and the unstarred variables should be swapped.
    if mc_step == 1:
        ϱ     , ϱˣ      = ϱˣ     , ϱ
        ϱ_ptr , ϱˣ_ptr  = ϱˣ_ptr , ϱ_ptr
        Jx    , Jxˣ     = Jxˣ    , Jx
        Jx_ptr, Jxˣ_ptr = Jxˣ_ptr, Jx_ptr
        Jy    , Jyˣ     = Jyˣ    , Jy
        Jy_ptr, Jyˣ_ptr = Jyˣ_ptr, Jy_ptr
        Jz    , Jzˣ     = Jzˣ    , Jz
        Jz_ptr, Jzˣ_ptr = Jzˣ_ptr, Jz_ptr
    # Loop over the local domain and check and compute
    # corrections for imminent vacuum.
    vacuum_imminent = False
    for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
        for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
            for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                # Unstarred and starred density at this point
                ϱ_ijk  = ϱ [i, j, k]
                ϱˣ_ijk = ϱˣ[i, j, k]
                # Check for imminent vacuum.
                # After the first MacCormack step, vacuum is considered
                # to be imminent if a density below the vacuum density,
                # ϱ_vacuum, will be reached within timespan similiar
                # time steps. That is, vacuum is imminent if
                # ϱ + timespan*dϱ < ϱ_vacuum,
                # where dϱ is the change in ϱ from the first MacCormack
                # step, given by dϱ = ½(ϱˣ - ϱ), where the factor ½ is
                # due to ϱˣ really holding double the change,
                # ϱˣ = ϱ + 2*dϱ. Put together, this means that vacuum
                # is imminent if
                # ϱˣ + ϱ*(2/timespan - 1) < 2/timespan*ϱ_vacuum.
                # After the second MacCormack step, vacuum is considered
                # to be imminent only if the density is lower than the
                # vacuum density, ϱ_vacuum. Because the starred
                # variables hold double their actual values,
                # this corresponds to
                # ϱˣ_ijk < 2*ϱ_vacuum.
                if (   (mc_step == 0 and ϱ_ijk*ℝ[2/timespan - 1] + ϱˣ_ijk < ℝ[2/timespan*ϱ_vacuum])
                    or (mc_step == 1 and                           ϱˣ_ijk < ℝ[2*ϱ_vacuum])
                    ):
                    vacuum_imminent = True
                    # The amount of smoothing to apply depends upon
                    # how far into the future densities below the vacuum
                    # density will be reached.
                    if mc_step == 0:
                        # The number of time steps before densities
                        # lower than the vacuum density is given by
                        # ϱ + timesteps*dϱ == ϱ_vacuum, dϱ = ½(ϱˣ - ϱ).
                        # --> timesteps = 2*(ϱ - ϱ_vacuum)/(ϱ - ϱˣ).
                        fac_time = 0.5*(ϱ_ijk - ϱˣ_ijk)/(ϱ_ijk - ϱ_vacuum)
                    else:  # mc_step == 1
                        # The density is already lower
                        # than the vaccuum density.
                        fac_time = 1
                    # Loop over all cell pairs (m, n) in the 3x3x3 block
                    # souronding the vacuum cell and apply smoothing.
                    for m in range(27):
                        # 3D indices of m'th cell
                        mi = i + relative_neighbour_indices_i[m]
                        mj = j + relative_neighbour_indices_j[m]
                        mk = k + relative_neighbour_indices_k[m]
                        for n in range(m + 1, 27):
                            # 3D indices of n'th cell
                            ni = i + relative_neighbour_indices_i[n]
                            nj = j + relative_neighbour_indices_j[n]
                            nk = k + relative_neighbour_indices_k[n]
                            # Distance squared between the two cells,
                            # in grid units (1 ≤ dist2 ≤ 12).
                            dist2 = (ni - mi)**2 + (nj - mj)**2 + (nk - mk)**2
                            # Compute vacuum corrections
                            ϱ_correction  = (ϱ [ni, nj, nk] - ℝ[ϱ [mi, mj, mk]])*ℝ[ fac_smoothing
                                                                                   *fac_time/dist2]
                            Jx_correction = (Jx[ni, nj, nk] - ℝ[Jx[mi, mj, mk]])*ℝ[ fac_smoothing
                                                                                   *fac_time/dist2]
                            Jy_correction = (Jy[ni, nj, nk] - ℝ[Jy[mi, mj, mk]])*ℝ[ fac_smoothing
                                                                                   *fac_time/dist2]
                            Jz_correction = (Jz[ni, nj, nk] - ℝ[Jz[mi, mj, mk]])*ℝ[ fac_smoothing
                                                                                   *fac_time/dist2]
                            # Store vacuum corrections
                            Δϱ [mi, mj, mk] += ϱ_correction
                            ΔJx[mi, mj, mk] += Jx_correction
                            ΔJy[mi, mj, mk] += Jy_correction
                            ΔJz[mi, mj, mk] += Jz_correction
                            Δϱ [ni, nj, nk] -= ϱ_correction
                            ΔJx[ni, nj, nk] -= Jx_correction
                            ΔJy[ni, nj, nk] -= Jy_correction
                            ΔJz[ni, nj, nk] -= Jz_correction
    # If vacuum is imminent on any process, consider it to be
    # imminent on every process.
    vacuum_imminent = allreduce(vacuum_imminent, op=MPI.LOR)
    if vacuum_imminent:
        # Communicate contributions to local vacuum corrections
        # residing on other processes.
        component.communicate_fluid_Δ(mode='add contributions')
        # Local Δ buffers now store final values. Populate pseudo
        # and ghost points of Δ buffers.
        component.communicate_fluid_Δ(mode='populate')
        # Apply vacuum corrections. Note that no further communication
        # is needed as we also apply vacuum corrections to the
        # pseudo and ghost points.
        for i in range(component.size):
            ϱ_ptr [i] += Δϱ_ptr [i]
        for i in range(component.size):
            Jx_ptr[i] += ΔJx_ptr[i]
        for i in range(component.size):
            Jy_ptr[i] += ΔJy_ptr[i]
        for i in range(component.size):
            Jz_ptr[i] += ΔJz_ptr[i]
    # The return value should indicate whether or not
    # vacuum corrections have been carried out.
    return vacuum_imminent
# 1D memory views of relative indices to the 27 neighbours of a cell
# (itself included). These are thus effectively mappings from
# 1D indices to 3D indices.
cython.declare(relative_neighbour_indices_i='Py_ssize_t[::1]',
               relative_neighbour_indices_j='Py_ssize_t[::1]',
               relative_neighbour_indices_k='Py_ssize_t[::1]',
               )
relative_neighbour_indices = asarray([(i, j, k) for i in range(-1, 2)
                                                for j in range(-1, 2)
                                                for k in range(-1, 2)], dtype=C2np['Py_ssize_t'])
relative_neighbour_indices_i = asarray(relative_neighbour_indices[:, 0]).copy()
relative_neighbour_indices_j = asarray(relative_neighbour_indices[:, 1]).copy()
relative_neighbour_indices_k = asarray(relative_neighbour_indices[:, 2]).copy()
