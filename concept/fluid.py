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
               ρ_ptr='double*',
               ρux_ptr='double*',
               ρuy_ptr='double*',
               ρuz_ptr='double*',
               )
def maccormack(component, ᔑdt):
    # Maximum allowed number of attempts to correct for
    # negative densities, for the first and second MacCormack step.
    max_vacuum_corrections = asarray([1, component.gridsize], dtype=C2np['int'])
    # Extract fluid grid pointers
    ρ_ptr   = component.ρ.grid
    ρux_ptr = component.ρux.grid
    ρuy_ptr = component.ρuy.grid
    ρuz_ptr = component.ρuz.grid
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
            if mc_step == 0 or attempt == 0:
                # Nullify the starred grid buffers,
                # so that they are ready to be populated
                # by the following MacCormack step.
                if mc_step == 0:
                    component.nullify_fluid_gridˣ()
                # Compute starred variables from unstarred variables
                evolve_fluid(component, ᔑdt, steps, mc_step)
            # Nullify the Δ buffers, so that they are ready to
            # be used by the following vacuum correction sweep.
            component.nullify_fluid_Δ()
            # Check and correct for density values heading dangerously
            # fast towards negative values. If every density value
            # is OK, accept this attempt at a MacCormack step as is.
            if not correct_vacuum(component, mc_step):
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
    for i in range(component.size):
        ρ_ptr  [i] *= 0.5
        ρux_ptr[i] *= 0.5
        ρuy_ptr[i] *= 0.5
        ρuz_ptr[i] *= 0.5
    # Nullify the starred grid buffers and the Δ buffers,
    # leaving these with no leftover junk.
    component.nullify_fluid_gridˣ()
    component.nullify_fluid_Δ()

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

# Function which evolve the fluid variables of a component
@cython.header(# Arguments
               component='Component',
               ᔑdt='dict',
               steps='Py_ssize_t[::1]',
               mc_step='int',
               # Locals
               h='double',
               i='Py_ssize_t',
               indices_local_start='Py_ssize_t[::1]',
               indices_local_end='Py_ssize_t[::1]',
               j='Py_ssize_t',
               k='Py_ssize_t',
               shape='tuple',
               step_i='Py_ssize_t',
               step_j='Py_ssize_t',
               step_k='Py_ssize_t',
               ux_ijk='double',
               ux_sjk='double',
               uy_ijk='double',
               uy_isk='double',
               uz_ijk='double',
               uz_ijs='double',
               w='double',
               ρ='double[:, :, ::1]',
               ρ_ijk='double',
               ρ_ijs='double',
               ρ_isk='double',
               ρ_sjk='double',
               ρux='double[:, :, ::1]',
               ρux_ijk='double',
               ρux_ijs='double',
               ρux_isk='double',
               ρux_sjk='double',
               ρuxˣ='double[:, :, ::1]',
               ρuy='double[:, :, ::1]',
               ρuy_ijk='double',
               ρuy_ijs='double',
               ρuy_isk='double',
               ρuy_sjk='double',
               ρuyˣ='double[:, :, ::1]',
               ρuz='double[:, :, ::1]',
               ρuz_ijk='double',
               ρuz_ijs='double',
               ρuz_isk='double',
               ρuz_sjk='double',
               ρuzˣ='double[:, :, ::1]',
               ρˣ='double[:, :, ::1]',
               )
def evolve_fluid(component, ᔑdt, steps, mc_step):
    """It is assumed that the unstarred and starred grids have
    correctly populated pseudo and ghost points.
    """
    # Physical grid spacing
    h = boxsize/component.gridsize
    # Exract steps in each direction
    step_i, step_j, step_k = steps
    # Arrays of start and end indices for the local part of the
    # fluid grids, meaning disregarding pseudo points and ghost points.
    # We have 2 ghost points in the beginning and 1 pseudo point and
    # 2 ghost points in the end.
    shape = component.shape
    indices_local_start = asarray((2, 2, 2), dtype=C2np['Py_ssize_t'])
    indices_local_end   = asarray(shape    , dtype=C2np['Py_ssize_t']) - 2 - 1
    # Extract fluid grids
    ρ   = component.ρ  .grid_mv
    ρux = component.ρux.grid_mv
    ρuy = component.ρuy.grid_mv
    ρuz = component.ρuz.grid_mv
    # Extract starred fluid grids
    ρˣ   = component.ρ  .gridˣ_mv
    ρuxˣ = component.ρux.gridˣ_mv
    ρuyˣ = component.ρuy.gridˣ_mv
    ρuzˣ = component.ρuz.gridˣ_mv
    # Get the equation of state parameter w at this instance in time
    w = component.w()
    # In the case of the second MacCormack step, the role of the
    # starred and the unstarred variables should be swapped.
    if mc_step == 1:
        ρ  , ρˣ   = ρˣ  , ρ
        ρux, ρuxˣ = ρuxˣ, ρux
        ρuy, ρuyˣ = ρuyˣ, ρuy
        ρuz, ρuzˣ = ρuzˣ, ρuz
    # Loop which update the parsed starred variables
    # from the parsed unstarred variables.
    for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
        for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
            for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                # Density at this point
                ρ_ijk = ρ[i, j, k]
                # Momentum density components at this point
                ρux_ijk = ρux[i, j, k]
                ρuy_ijk = ρuy[i, j, k]
                ρuz_ijk = ρuz[i, j, k]
                # Velocity components at this point
                ux_ijk = ρux_ijk/ρ_ijk
                uy_ijk = ρuy_ijk/ρ_ijk
                uz_ijk = ρuz_ijk/ρ_ijk
                # Density at forward (backward) points
                ρ_sjk = ρ[i + step_i, j         , k         ]
                ρ_isk = ρ[i         , j + step_j, k         ]
                ρ_ijs = ρ[i         , j         , k + step_k]
                # Momentum density components at forward (backward) points
                ρux_sjk = ρux[i + step_i, j         , k         ]
                ρux_isk = ρux[i         , j + step_j, k         ]
                ρux_ijs = ρux[i         , j         , k + step_k]
                ρuy_sjk = ρuy[i + step_i, j         , k         ]
                ρuy_isk = ρuy[i         , j + step_j, k         ]
                ρuy_ijs = ρuy[i         , j         , k + step_k]
                ρuz_sjk = ρuz[i + step_i, j         , k         ]
                ρuz_isk = ρuz[i         , j + step_j, k         ]
                ρuz_ijs = ρuz[i         , j         , k + step_k]
                # Velocity components at forward (backward) points
                ux_sjk = ρux_sjk/ρ_sjk
                uy_isk = ρuy_isk/ρ_isk
                uz_ijs = ρuz_ijs/ρ_ijs
                # Flux of ρ (ρ*u)
                ρ_flux = (+ step_i*(ρux_sjk - ρux_ijk)
                          + step_j*(ρuy_isk - ρuy_ijk)
                          + step_k*(ρuz_ijs - ρuz_ijk)
                          )
                # Flux of ρux (ρux*u)
                ρux_flux = (+ step_i*(ρux_sjk*ux_sjk - ρux_ijk*ux_ijk)
                            + step_j*(ρux_isk*uy_isk - ρux_ijk*uy_ijk)
                            + step_k*(ρux_ijs*uz_ijs - ρux_ijk*uz_ijk)
                            # Pressure term
                            + step_i*ℝ[light_speed**2*w/(1 + w)]*(ρ_sjk - ρ_ijk)
                            )
                # Flux of ρuy (ρuy*u)
                ρuy_flux = (+ step_i*(ρuy_sjk*ux_sjk - ρuy_ijk*ux_ijk)
                            + step_j*(ρuy_isk*uy_isk - ρuy_ijk*uy_ijk)
                            + step_k*(ρuy_ijs*uz_ijs - ρuy_ijk*uz_ijk)
                            # Pressure term
                            + step_j*ℝ[light_speed**2*w/(1 + w)]*(ρ_isk - ρ_ijk)
                            )
                # Flux of ρuz (ρuz*u)
                ρuz_flux = (+ step_i*(ρuz_sjk*ux_sjk - ρuz_ijk*ux_ijk)
                            + step_j*(ρuz_isk*uy_isk - ρuz_ijk*uy_ijk)
                            + step_k*(ρuz_ijs*uz_ijs - ρuz_ijk*uz_ijk)
                            # Pressure term
                            + step_k*ℝ[light_speed**2*w/(1 + w)]*(ρ_ijs - ρ_ijk)
                            )
                # Update ρ
                ρˣ[i, j, k] += (# Initial value
                                + ρ_ijk
                                # Flux
                                - ℝ[ᔑdt['a⁻²']/h]*ρ_flux
                                )
                # Update ρux
                ρuxˣ[i, j, k] += (# Initial value
                                  + ρux_ijk
                                  # Flux
                                  - ℝ[ᔑdt['a⁻²']/h]*ρux_flux
                                  )
                # Update ρuy
                ρuyˣ[i, j, k] += (# Initial value
                                  + ρuy_ijk
                                  # Flux
                                  - ℝ[ᔑdt['a⁻²']/h]*ρuy_flux
                                  )
                # Update ρuz
                ρuzˣ[i, j, k] += (# Initial value
                                  + ρuz_ijk
                                  # Flux
                                  - ℝ[ᔑdt['a⁻²']/h]*ρuz_flux
                                  )
    # Populate the pseudo and ghost points with the updated values.
    # Depedendent on whether we are doing the first of second
    # MacCormack step (mc_step), the updated grids are really the
    # starred grids (first MacCormack step) or the
    # unstarred grids (second MacCormack step)
    if mc_step == 0:
        component.communicate_fluid_gridsˣ(mode='populate')
    else:  # mc_step == 1
        component.communicate_fluid_grids(mode='populate')

# Function which checks for imminent vacuum in a fluid component
# and does one sweep of vacuum corrections.
@cython.header(# Arguments
               component='Component',
               mc_step='int',
               # Locals
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
               Δρ='double[:, :, ::1]',
               Δρ_ptr='double*',
               Δρux='double[:, :, ::1]',
               Δρux_ptr='double*',
               Δρuy='double[:, :, ::1]',
               Δρuy_ptr='double*',
               Δρuz='double[:, :, ::1]',
               Δρuz_ptr='double*',
               ρ='double[:, :, ::1]',
               ρ_correction='double',
               ρ_ijk='double',
               ρ_ptr='double*',
               ρux='double[:, :, ::1]',
               ρux_correction='double',
               ρux_ptr='double*',
               ρuxˣ='double[:, :, ::1]',
               ρuy='double[:, :, ::1]',
               ρuy_correction='double',
               ρuy_ptr='double*',
               ρuyˣ='double[:, :, ::1]',
               ρuz='double[:, :, ::1]',
               ρuz_correction='double',
               ρuz_ptr='double*',
               ρuzˣ='double[:, :, ::1]',
               ρˣ='double[:, :, ::1]',
               ρˣ_ijk='double',
               returns='bint',
               )
def correct_vacuum(component, mc_step):
    """This function will detect and correct for imminent vacuum in a
    fluid component. The vacuum detection is done differently depending
    on the MacCormack step (the parsed mc_step). For the first
    MacCormack step, vacuum is considered imminent if a density below
    the vacuum density, ρ_vacuum, will be reached within timespan
    similiar time steps. For the second MacCormack step, vacuum is
    considered imminent if the density is below the vacuum density.
    The vacuum correction is done by smoothing all fluid variables in
    the 3x3x3 neighbouring cells souronding the vacuum cell.
    The smoothing between each pair of cells, call them (ρi, ρj),
    is given by
    ρi += fac_smoothing*(ρj - ρi)/r²,
    ρj += fac_smoothing*(ρi - ρj)/r²,
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
    fac_smoothing = ℝ[1.5/(6/1 + 12/2 + 8/3)]
    # Arrays of start and end indices for the local part of the
    # fluid grids, meaning disregarding pseudo points and ghost points.
    # We have 2 ghost points in the beginning and 1 pseudo point and
    # 2 ghost points in the end.
    shape = component.shape
    indices_local_start = asarray([2, 2, 2], dtype=C2np['Py_ssize_t'])
    indices_local_end   = asarray(shape    , dtype=C2np['Py_ssize_t']) - 2 - 1
    # Extract memory views and pointers to the fluid variables
    ρ        = component.ρ  .grid_mv
    ρ_ptr    = component.ρ  .grid
    ρˣ       = component.ρ  .gridˣ_mv
    ρˣ_ptr   = component.ρ  .gridˣ
    Δρ       = component.ρ  .Δ_mv
    Δρ_ptr   = component.ρ  .Δ
    ρux      = component.ρux.grid_mv
    ρux_ptr  = component.ρux.grid
    ρuxˣ     = component.ρux.gridˣ_mv
    ρuxˣ_ptr = component.ρux.gridˣ
    Δρux     = component.ρux.Δ_mv
    Δρux_ptr = component.ρux.Δ
    ρuy      = component.ρuy.grid_mv
    ρuy_ptr  = component.ρuy.grid
    ρuyˣ     = component.ρuy.gridˣ_mv
    ρuyˣ_ptr = component.ρuy.gridˣ
    Δρuy     = component.ρuy.Δ_mv
    Δρuy_ptr = component.ρuy.Δ
    ρuz      = component.ρuz.grid_mv
    ρuz_ptr  = component.ρuz.grid
    ρuzˣ     = component.ρuz.gridˣ_mv
    ρuzˣ_ptr = component.ρuz.gridˣ
    Δρuz     = component.ρuz.Δ_mv
    Δρuz_ptr = component.ρuz.Δ
    # In the case of the second MacCormack step, the role of the
    # starred and the unstarred variables should be swapped.
    if mc_step == 1:
        ρ      , ρˣ       = ρˣ      , ρ
        ρ_ptr  , ρˣ_ptr   = ρˣ_ptr  , ρ_ptr
        ρux    , ρuxˣ     = ρuxˣ    , ρux
        ρux_ptr, ρuxˣ_ptr = ρuxˣ_ptr, ρux_ptr
        ρuy    , ρuyˣ     = ρuyˣ    , ρuy
        ρuy_ptr, ρuyˣ_ptr = ρuyˣ_ptr, ρuy_ptr
        ρuz    , ρuzˣ     = ρuzˣ    , ρuz
        ρuz_ptr, ρuzˣ_ptr = ρuzˣ_ptr, ρuz_ptr
    # Loop over the local domain and check and compute
    # corrections for imminent vacuum.
    vacuum_imminent = False
    for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
        for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
            for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                # Unstarred and starred density at this point
                ρ_ijk  = ρ [i, j, k]
                ρˣ_ijk = ρˣ[i, j, k]
                # Check for imminent vacuum.
                # After the first MacCormack step, vacuum is considered
                # to be imminent if a density below the vacuum density,
                # ρ_vacuum, will be reached within timespan similiar
                # time steps. That is, vacuum is imminent if
                # ρ + timespan*dρ < ρ_vacuum,
                # where dρ is the change in ρ from the first MacCormack
                # step, given by dρ = ½(ρˣ - ρ), where the factor ½ is
                # due to ρˣ really holding double the change,
                # ρˣ = ρ + 2*dρ. Put together, this means that vacuum
                # is imminent if
                # ρˣ + ρ*(2/timespan - 1) < 2/timespan*ρ_vacuum.
                # After the second MacCormack step, vacuum is considered
                # to be imminent only if the density is lower than the
                # vacuum density, ρ_vacuum. Because the starred
                # variables hold double their actual values,
                # this corresponds to
                # ρˣ_ijk < 2*ρ_vacuum.
                if (   (mc_step == 0 and ρ_ijk*ℝ[2/timespan - 1] + ρˣ_ijk < ℝ[2/timespan*ρ_vacuum])
                    or (mc_step == 1 and                           ρˣ_ijk < ℝ[2*ρ_vacuum])
                    ):
                    vacuum_imminent = True
                    # The amount of smoothing to apply depends upon
                    # how far into the future densities below the vacuum
                    # density will be reached.
                    if mc_step == 0:
                        # The number of time steps before densities
                        # lower than the vacuum density is given by
                        # ρ + timesteps*dρ == ρ_vacuum, dρ = ½(ρˣ - ρ).
                        # --> timesteps = 2*(ρ - ρ_vacuum)/(ρ - ρˣ).
                        fac_time = 0.5*(ρ_ijk - ρˣ_ijk)/(ρ_ijk - ρ_vacuum)
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
                            ρ_correction   = (ρ  [ni, nj, nk] - ℝ[ρ  [mi, mj, mk]])*ℝ[fac_smoothing*fac_time/dist2]
                            ρux_correction = (ρux[ni, nj, nk] - ℝ[ρux[mi, mj, mk]])*ℝ[fac_smoothing*fac_time/dist2]
                            ρuy_correction = (ρuy[ni, nj, nk] - ℝ[ρuy[mi, mj, mk]])*ℝ[fac_smoothing*fac_time/dist2]
                            ρuz_correction = (ρuz[ni, nj, nk] - ℝ[ρuz[mi, mj, mk]])*ℝ[fac_smoothing*fac_time/dist2]
                            # Store vacuum corrections
                            Δρ  [mi, mj, mk] += ρ_correction
                            Δρux[mi, mj, mk] += ρux_correction
                            Δρuy[mi, mj, mk] += ρuy_correction
                            Δρuz[mi, mj, mk] += ρuz_correction
                            Δρ  [ni, nj, nk] -= ρ_correction
                            Δρux[ni, nj, nk] -= ρux_correction
                            Δρuy[ni, nj, nk] -= ρuy_correction
                            Δρuz[ni, nj, nk] -= ρuz_correction
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
            ρ_ptr  [i] += Δρ_ptr  [i]
            ρux_ptr[i] += Δρux_ptr[i]
            ρuy_ptr[i] += Δρuy_ptr[i]
            ρuz_ptr[i] += Δρuz_ptr[i]
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
