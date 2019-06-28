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
import interactions
cimport('from analysis import debug, measure, powerspec')
cimport('from communication import domain_subdivisions')
cimport('from graphics import render2D, render3D')
cimport('from integration import cosmic_time,          '
        '                        expand,               '
        '                        hubble,               '
        '                        initiate_time,        '
        '                        scale_factor,         '
        '                        scalefactor_integral, '
        )
cimport('from snapshot import get_initial_conditions, save')
cimport('from species import Component, get_representation')
cimport('from utilities import delegate')



# Function containing the main time loop of CO𝘕CEPT
@cython.header(
    # Locals
    autosave_filename=str,
    autosave_time='double',
    bottleneck=str,
    component='Component',
    components=list,
    dump_index='Py_ssize_t',
    dump_time=object,  # collections.namedtuple
    dump_times=list,
    integrals='double[::1]',
    output_filenames=dict,
    period_frac='double',
    recompute_Δt_max='bint',
    sync_at_dump='bint',
    sync_time='double',
    time_step='Py_ssize_t',
    time_step_last_sync='Py_ssize_t',
    time_step_type=str,
    timespan='double',
    Δt='double',
    Δt_begin='double',
    Δt_downjump_fac='double',
    Δt_half='double',
    Δt_increase_fac='double',
    Δt_increase_max_fac='double',
    Δt_increase_min_fac='double',
    Δt_initial_fac='double',
    Δt_min='double',
    Δt_max='double',
    Δt_new='double',
    Δt_period='Py_ssize_t',
    Δt_ratio='double',
    Δt_ratio_abort='double',
    Δt_ratio_warn='double',
    Δt_reduce_fac='double',
    Δt_sync='double',
    ᔑdt_substeps=dict,
    returns='void',
)
def timeloop():
    # Do nothing if no dump times exist
    if not (  [nr for val in output_times['a'].values() for nr in val]
            + [nr for val in output_times['t'].values() for nr in val]):
        return
    # Print out domain decomposition
    masterprint(
        f'Domain decomposition: '
        f'{domain_subdivisions[0]}×{domain_subdivisions[1]}×{domain_subdivisions[2]}'
    )
    # Determine and set the correct initial values for the cosmic time
    # universals.t and the scale factor universals.a = a(universals.t).
    initiate_time()
    # Get the dump times and the output filename patterns
    dump_times, output_filenames = prepare_for_output()
    # Get the initial components.
    # These may be loaded from a snapshot or generated from scratch.
    masterprint('Setting up initial conditions ...')
    components = get_initial_conditions()
    if not components:
        masterprint('done')
        return
    # Realize all linear fluid variables of all components
    for component in components:
        component.realize_if_linear(0, specific_multi_index=0)        # ϱ
        component.realize_if_linear(1, specific_multi_index=0)        # J
        component.realize_if_linear(2, specific_multi_index='trace')  # 𝒫
        component.realize_if_linear(2, specific_multi_index=(0, 0))   # ς
    masterprint('done')
    # Possibly output at the beginning of simulation
    if dump_times[0].t == universals.t or dump_times[0].a == universals.a:
        dump(components, output_filenames, dump_times[0])
        dump_times.pop(0)
        # Return now if all dumps lie at the initial time
        if len(dump_times) == 0:
            return
    # The initial time step size Δt will be set to the maximum allowed
    # value times this factor. At early times of almost homogeneity,
    # it is preferable with a small Δt, and so
    # this factor should be below unity.
    Δt_initial_fac = 0.9
    # When reducing Δt, set it to the maximum allowed value
    # times this factor.
    Δt_reduce_fac = 0.95
    # When increasing Δt, we never increase it beyond the maximally
    # alowed value Δt_max. In fact, we never increase Δt beyond
    # Δt + Δt_increase_fac*(Δt_max - Δt).
    # A value of Δt_increase_fac slightly below 1 ensures that the time
    # step size is not set right at the boundary of what is allowed,
    # which would be bad as the time step will then often have to be
    # lowered right after it has been increased.
    Δt_increase_fac = 0.95
    # The maximum allowed fractional increase in Δt
    # after Δt_period time steps with constant time step size.
    Δt_increase_max_fac = 0.25
    # The minimum fractional increase in Δt needed before it is deemed
    # worth it to synchronize drifts/kicks and update Δt.
    Δt_increase_min_fac = 0.01
    # Ratios between old and new Δt, below which the program
    # will show a warning or abort, respectively.
    Δt_ratio_warn  = 0.7
    Δt_ratio_abort = 0.01
    # When using adaptive time stepping (N_rungs > 1), a particle jumps
    # to the rung just above its current rung as soon as its velocity
    # gets too large (according to fac_softening). Before it jumps to
    # the rung just below, its velocity has to be this factor times the
    # minimum velocity of its current rung. This factor should be
    # somewhat below unity, so that a particle only jumps down (gets a
    # larger time step) if its velocity is well below what is absolutely
    # needed. This ensures that particles with velocities right at the
    # border between two rungs stay at a given rung over many time
    # steps, rather than fluctuating back and forth between two rungs,
    # which would degrade the symplecticity.
    Δt_downjump_fac = 0.85
    # The number of time steps before the base time step size Δt is
    # allowed to increase. Choosing a multiple of 8 prevents the
    # formation of spurious anisotropies when evolving fluids with the
    # MacCormack method, as each of the 8 flux directions are then
    # used with the same time step size (in the simple case of no
    # reduction to Δt and no synchronizations due to dumps).
    Δt_period = 1*8
    # Set initial time step size
    if Δt_begin_autosave == -1:
        # Set the initial time step size to the largest allowed value
        # times Δt_initial_fac.
        Δt_max, bottleneck = get_base_timestep_size(components)
        Δt_begin = Δt_initial_fac*Δt_max
        # We always want the simulation time span to be at least
        # one whole Δt_period long.
        timespan = dump_times[len(dump_times) - 1].t - universals.t
        if Δt_begin > timespan/Δt_period:
            Δt_begin = timespan/Δt_period
        # We need at least 1.5 base time steps before the first dump
        if Δt_begin > (dump_times[0].t - universals.t)/1.5:
            Δt_begin = (dump_times[0].t - universals.t)/1.5
        Δt = Δt_begin
    else:
        # Set Δt_begin and Δt to the autosaved values
        Δt_begin = Δt_begin_autosave
        Δt = Δt_autosave
    # Minimum allowed time step size.
    # If Δt needs to be lower than this, the program will terminate.
    Δt_min = 1e-4*Δt_begin
    # Construct initial rung populatation
    for component in components:
        component.assign_rungs(Δt, fac_softening)
    # Dict of arrays which will store the time step integrals for the
    # highest rung (all time step integrals for lower rungs can be build
    # by adding up these).
    ᔑdt_substeps = {
        integrand: zeros(ℤ[2**N_rungs//2*3], dtype=C2np['double'])
        for integrand in (
            # Global integrands
            '1',
            'a**(-1)',
            'a**(-2)',
            'ȧ/a',
            # Single-component integrands
            *[(integrand, component.name) for component, in itertools.product(*[components]*1)
                for integrand in (
                    'a**(-3*w_eff)',
                    'a**(-3*w_eff-1)',
                    'a**(3*w_eff-2)',
                    'a**(-3*w_eff)*Γ/H',
                )
            ],
            # Two-component integrands
            *[(integrand, component_0.name, component_1.name)
                for component_0, component_1 in itertools.product(*[components]*2)
                for integrand in (
                    'a**(-3*w_eff₀-3*w_eff₁-1)',
                )
            ]
        )
    }
    # Record what time it is, for use with autosaving
    autosave_time = time()
    # The main time loop
    masterprint('Beginning of main time loop')
    time_step = initial_time_step
    bottleneck = ''
    time_step_type = 'init'
    sync_time = ထ
    time_step_last_sync = 0
    recompute_Δt_max = True
    for dump_index, dump_time in enumerate(dump_times):
        # Break out of this loop when a dump has been performed
        while True:
            # Print out message at beginning of each time step
            print_timestep_heading(time_step, Δt,
                bottleneck if time_step_type == 'init' else '', components)
            # Analyze and print out debugging information, if required
            with unswitch:
                if enable_debugging:
                    debug(components)
            # Handle the time step.
            # This is either of type "init" or "full".
            if time_step_type == 'init':
                # An init step is always followed by a full step
                time_step_type = 'full'
                # This is not a full base time step. Half a long-range
                # kick will be applied, i.e. fluid interactions,
                # long-range particle interactions and internal
                # sources terms. Each particle rung will be kicked by
                # half a sub-step. It is assumed that the drifting and
                # kicking of all components is synchronized. As this
                # does not count as an actual time step,
                # the universal time will not be updated.
                # The base time step Δt is split into 2**N_rungs half
                # sub-steps for the highest rung (number N_rungs - 1).
                # The short-range kick of the particles at rung 0,
                # as well as fluid kick and long-range particle kick
                # is then 2**N_rungs//2 half sub-steps long.
                # Compute time step integrals for each
                # of these 2**N_rungs//2 half sub-steps.
                compute_time_step_integrals(Δt, components, ᔑdt_substeps, 'init', sync_time)
                # Apply initial half kick to fluids, initial half
                # long-range kick to particles and inital half
                # application of internal sources.
                kick_long(components, ᔑdt_substeps, 'init')
                # Assign a short-range rung to each particle
                for component in components:
                    component.assign_rungs(Δt, fac_softening)
                # Initial half kick of particles on all rungs
                kick_short(components, ᔑdt_substeps)
                # The full step following this init step will reuse all
                # of the 2**N_rungs//2 integrals computed for the half
                # sub-steps at the beginning of the base step. Here it
                # is assumed that the content of ᔑdt_substeps is
                # computed by a previous full step, in which case the
                # integrals to reuse will be the ones at the end, not
                # the beginning. We thus need to shift the 2**N_rungs//2
                # integrals so that they appear at the end.
                for integrals in ᔑdt_substeps.values():
                    integrals[ℤ[2**N_rungs]:] = integrals[:ℤ[2**N_rungs//2]]
                # Check whether next dump is within 1.5*Δt
                if dump_time.t - universals.t <= 1.5*Δt:
                    # Next base step should synchronize at dump time
                    sync_time = dump_time.t
                    continue
                # Check whether the base time step needs to be reduced
                Δt_max, bottleneck = get_base_timestep_size(components)
                if Δt > Δt_max:
                    # Next base step should synchronize.
                    # Thereafter we can lower the base time step size.
                    sync_time = universals.t + 0.5*Δt
                    recompute_Δt_max = False
                    continue
            elif time_step_type == 'full':
                # This is a full base time step of size Δt.
                # All components will be drifted and kicked Δt.
                # The kicks will start and end half a time step ahead
                # of the drifts.
                # The base time step Δt needs to be split
                # into 2**N_rungs half sub-steps for the
                # highest rung (number N_rungs - 1).
                # Compute time step integrals for each half sub-step.
                compute_time_step_integrals(Δt, components, ᔑdt_substeps, 'full', sync_time)
                # Drift fluids
                drift_fluids(components, ᔑdt_substeps)
                # Continually perform interlaced drift and kick
                # operations of the rungs, until the particles are
                # drifted forward to the exact time of the next base
                # time step (Δt away) and kicked half a sub-step
                # (of size Δt/(2*2**i) for run i) into the next
                # base time step.
                driftkick_short(components, Δt, ᔑdt_substeps, Δt_downjump_fac)
                # All drifting is now exactly at the next base time
                # step, while the long-range kicks are lacking behind.
                # Before doing the long-range kicks, set the universal
                # time to match the current position of the long-range
                # kicks, so that various time averages will be over
                # the kick step.
                integrals = ᔑdt_substeps['1']
                Δt_half = sum(integrals[:ℤ[2**N_rungs//2]])
                universals.t += Δt_half
                universals.a = scale_factor(universals.t)
                # Apply full kick to fluids, full long-range kick to
                # particles and fully apply internal sources.
                kick_long(components, ᔑdt_substeps, 'full')
                # Set universal time to match end of this base time
                # step (the location of drifts).
                integrals = ᔑdt_substeps['1']
                Δt_half = sum(integrals[ℤ[2**N_rungs//2]:ℤ[2**N_rungs]])
                universals.t += Δt_half
                universals.a = scale_factor(universals.t)
                # Check whether we are at sync time
                if Δt_half == 0 or sync_time - universals.t <= Δt_reltol*Δt:
                    # We are at sync time. Base time step completed.
                    # Ensure that the universal time
                    # matches exactly with the sync time.
                    universals.t = sync_time
                    # Reset time_step_type and sync_time
                    time_step_type = 'init'
                    sync_time = ထ
                    # Reduce base time step if necessary.
                    # If not, increase it as allowed.
                    if recompute_Δt_max:
                        Δt_max, bottleneck = get_base_timestep_size(components)
                    recompute_Δt_max = True
                    if Δt > Δt_max:
                        # Reduce base time step size
                        Δt_new = Δt_reduce_fac*Δt_max
                        Δt_ratio = Δt_new/Δt
                        if Δt_ratio < Δt_ratio_abort:
                            abort(
                                f'Due to {bottleneck}, the time step size needs to be rescaled '
                                f'by a factor {Δt_ratio:.1g}. This extreme change is unacceptable.'
                            )
                        elif Δt_ratio < Δt_ratio_warn:
                            masterwarn(
                                f'Rescaling time step size by a '
                                f'factor {Δt_ratio:.1g} due to {bottleneck}'
                            )
                        if Δt_new < Δt_min:
                            abort(
                                f'Time evolution effectively halted with a time step size '
                                f'of {Δt_new} {unit_time} (at the start of the simulation '
                                f'the time step size was {Δt_begin} {unit_time})'
                        )
                        Δt = Δt_new
                    else:
                        # The base time step size will be increased,
                        # and so we have no bottleneck.
                        bottleneck = ''
                        # New, bigger base time step size,
                        # according to Δt ∝ a.
                        Δt_new = universals.a*ℝ[Δt_begin/a_begin]
                        if Δt_new < Δt:
                            Δt_new = Δt
                        # Add small, constant contribution to the new
                        # base time step size.
                        Δt_new += ℝ[(1 + Δt_increase_min_fac)*Δt_begin]
                        # Make sure that the relative change
                        # of the base time step size is not too big.
                        if  Δt_new > ℝ[Δt + Δt_increase_fac*(Δt_max - Δt)]:
                            Δt_new = ℝ[Δt + Δt_increase_fac*(Δt_max - Δt)]
                        period_frac = float(time_step + 1 - time_step_last_sync)/Δt_period
                        if period_frac > 1:
                            period_frac = 1
                        elif period_frac < 0:
                            period_frac = 0
                        if  Δt_new > ℝ[1 + period_frac*Δt_increase_max_fac]*Δt:
                            Δt_new = ℝ[1 + period_frac*Δt_increase_max_fac]*Δt
                        Δt = Δt_new
                    # Update time step counters
                    time_step += 1
                    time_step_last_sync = time_step
                    # If it is time, perform autosave
                    with unswitch:
                        if autosave_interval > 0:
                            if bcast(time() - autosave_time > ℝ[autosave_interval/units.s]):
                                autosave(components, time_step, Δt, Δt_begin)
                                autosave_time = time()
                    # Dump output if at dump time
                    if universals.t == dump_time.t:
                        dump(components, output_filenames, dump_time)
                        # Ensure that we have at least 1.5
                        # base time steps before the next dump.
                        if dump_index != len(dump_times) - 1:
                            Δt_max = (dump_times[dump_index + 1].t - universals.t)/1.5
                            if Δt > Δt_max:
                                Δt = Δt_max
                        # Break out of the infinite loop,
                        # proceeding to the next dump time.
                        break
                    # Not at dump time.
                    # Ensure that we have at least 1.5
                    # base time steps before we reach the dump time.
                    Δt_max = (dump_time.t - universals.t)/1.5
                    if Δt > Δt_max:
                        Δt = Δt_max
                    # Go to init step
                    continue
                # Check whether next dump is within 1.5*Δt
                if dump_time.t - universals.t <= 1.5*Δt:
                    # We need to synchronize at dump time
                    sync_time = dump_time.t
                    continue
                # Check whether the base time step needs to be reduced
                Δt_max, bottleneck = get_base_timestep_size(components)
                if Δt > Δt_max:
                    # We should synchronize, whereafter the
                    # base time step size can be lowered.
                    sync_time = universals.t + 0.5*Δt
                    recompute_Δt_max = False
                    continue
                # Check whether the base time step should be increased
                if (Δt_max > ℝ[1 + Δt_increase_min_fac]*Δt
                    and (time_step + 1 - time_step_last_sync) >= Δt_period
                ):
                    # We should synchronize, whereafter the
                    # base time step size can be raised.
                    sync_time = universals.t + 0.5*Δt
                    recompute_Δt_max = False
                    continue
                # Base time step completed
                time_step += 1
    # All dumps completed; end of main time loop
    print_timestep_heading(time_step, Δt, bottleneck, components, end=True)
    # Remove dumped autosave snapshot, if any
    if master:
        autosave_filename = f'{autosave_dir}/autosave_{jobid}.hdf5'
        if os.path.isfile(autosave_filename):
            os.remove(autosave_filename)
# Two times with a difference below Δt_reltol*Δt
# should be treated as indistinguishable.
cython.declare(Δt_reltol='double')
Δt_reltol = 1e-6

# Function for computing the size of the base time step
@cython.header(
    # Arguments
    components=list,
    # Locals
    H='double',
    a='double',
    bottleneck=str,
    component='Component',
    component_lapse='Component',
    extreme_force=str,
    force=str,
    lapse_gridsize='Py_ssize_t',
    method=str,
    resolution='Py_ssize_t',
    scale='double',
    v_max='double',
    v_rms='double',
    Δt='double',
    Δt_courant='double',
    Δt_decay='double',
    Δt_dynamical='double',
    Δt_hubble='double',
    Δt_pm='double',
    Δt_ẇ='double',
    Δx_max='double',
    ρ_bar='double',
    ρ_bar_component='double',
    returns=tuple,
)
def get_base_timestep_size(components):
    """This function computes the maximum allowed size
    of the base time step Δt. The time step limiters come in three
    categories; global limiters, component limiters and
    particle/fluid element limiters. For each limiter, the value of Δt
    should not be exceed a small fraction of the following.
    Global limiters:
    - The dynamical time scale.
    - The Hubble time (≃ present age of the universe)
      if Hubble expansion is enabled.
    Component limiters:
    - 1/abs(ẇ) for every component, so that the transition from
      relativistic to non-relativistic happens smoothly.
    - The reciprocal decay rate of each matter component, wieghted with
      their current total mass (or background density) relative to
      all matter.
    Particle/fluid element limiters:
    - For fluid components (with a Boltzmann hierarchy closed after J
      (velocity)): The time it takes for the fastest fluid element to
      traverse a fluid cell, i.e. the Courant condition.
    - For particle/fluid components using the PM method: The time it
      would take to traverse a PM grid cell for a particle/fluid element
      with the rms velocity of all particles/fluid elements within a
      given component.
    - For particle components using the P³M method: The time it
      would take to traverse the long/short-range force split scale for
      a particle with the rms velocity of all particles within a
      given component.
    The return value is a tuple containing the maximum allowed Δt and a
    str stating which limiter is the bottleneck.
    """
    a = universals.a
    H = hubble(a)
    Δt = ထ
    bottleneck = ''
    # The dynamical time scale
    ρ_bar = 0
    for component in components:
        ρ_bar += a**(-3*(1 + component.w_eff(a=a)))*component.ϱ_bar
    Δt_dynamical = fac_dynamical/sqrt(G_Newton*ρ_bar)
    if Δt_dynamical < Δt:
        Δt = Δt_dynamical
        bottleneck = 'the dynamical timescale'
    # The Hubble time
    if enable_Hubble:
        Δt_hubble = fac_hubble/H
        if Δt_hubble < Δt:
            Δt = Δt_hubble
            bottleneck = 'the Hubble time'
    # 1/abs(ẇ)
    for component in components:
        Δt_ẇ = fac_ẇ/(abs(cast(component.ẇ(a=a), 'double')) + machine_ϵ)
        if Δt_ẇ < Δt:
            Δt = Δt_ẇ
            bottleneck = f'ẇ of {component.name}'
    # Reciprocal decay rate
    for component in components:
        if component.representation == 'fluid' and component.is_linear(0):
            continue
        ρ_bar_component = component.ϱ_bar*a**(-3*(1 + component.w_eff(a=a)))
        Δt_decay = fac_Γ/(abs(component.Γ(a)) + machine_ϵ)*ρ_bar/ρ_bar_component
        if Δt_decay < Δt:
            Δt = Δt_decay
            bottleneck = f'decay rate of {component.name}'
    # Courant condition for fluid elements
    for component in components:
        if component.representation == 'particles':
            continue
        # Find maximum propagation speed of fluid
        v_max = measure(component, 'v_max')
        # In the odd case of a completely static component,
        # set v_max to be just above 0.
        if v_max == 0:
            v_max = machine_ϵ
        # The Courant condition
        Δx_max = boxsize/component.gridsize
        Δt_courant = fac_courant*Δx_max/v_max
        if Δt_courant < Δt:
            Δt = Δt_courant
            bottleneck = f'the Courant condition for {component.name}'
    # PM limiter
    for component in components:
        # Find PM resolution for this component.
        # The PM method is implemented for gravity and the lapse force.
        resolution = 0
        lapse_gridsize = 0
        for force, method in component.forces.items():
            if method != 'pm':
                continue
            if force == 'gravity':
                if φ_gridsize > resolution:
                    resolution = φ_gridsize
                    extreme_force = 'gravity'
            elif force == 'lapse':
                # Find gridsize of the lapse force
                if lapse_gridsize == 0:
                    for component_lapse in components:
                        if component_lapse.species != 'lapse':
                            continue
                        lapse_gridsize = component_lapse.gridsize
                        break
                    else:
                        abort(
                            f'Failed to detect any lapse component, but the "{component.name}" '
                            f'component is assigned the lapse force.'
                        )
                if lapse_gridsize > resolution:
                    resolution = lapse_gridsize
                    extreme_force = 'lapse'
            else:
                abort(f'Unregistered force "{force}" with method "{method}"')
        if resolution == 0:
            continue
        # Find rms bulk velocity, i.e. do not add the sound speed
        v_rms = measure(component, 'v_rms')
        if component.representation == 'fluid':
            v_rms -= light_speed*sqrt(component.w(a=a))/a
        # In the odd case of a completely static component,
        # set v_rms to be just above 0.
        if v_rms < machine_ϵ:
            v_rms = machine_ϵ
        # The PM limiter
        Δx_max = boxsize/resolution
        Δt_pm = fac_pm*Δx_max/v_rms
        if Δt_pm < Δt:
            Δt = Δt_pm
            bottleneck = f'the PM method of the {extreme_force} force for {component.name}'
    # P³M limiter
    for component in components:
        # Find P³M resolution for this component.
        # The P³M method is only implemented for gravity.
        scale = ထ
        for force, method in component.forces.items():
            if method != 'p3m':
                continue
            if force == 'gravity':
                if ℝ[shortrange_params['gravity']['scale']] < scale:
                    scale = ℝ[shortrange_params['gravity']['scale']]
                    extreme_force = 'gravity'
            else:
                abort(f'Unregistered force "{force}" with method "{method}"')
        if scale == ထ:
            continue
        # Find rms velocity
        v_rms = measure(component, 'v_rms')
        # In the odd case of a completely static component,
        # set v_rms to be just above 0.
        if v_rms < machine_ϵ:
            v_rms = machine_ϵ
        # The P³M limiter
        Δx_max = scale
        Δt_p3m = fac_p3m*Δx_max/v_rms
        if Δt_p3m < Δt:
            Δt = Δt_p3m
            bottleneck = f'the P³M method of the {extreme_force} force for {component.name}'
    # Return maximum allowed base time step size and the bottleneck
    return Δt, bottleneck

# Function for tabulating integrals over (sub) time steps
@cython.header(
    # Arguments
    Δt_base='double',
    components=list,
    ᔑdt_substeps=dict,
    step_type=str,
    sync_time='double',
    # Locals
    at_sync_time='bint',
    half_substep='Py_ssize_t',
    index_end='Py_ssize_t',
    index_start='Py_ssize_t',
    integrals='double[::1]',
    integrand=object,  # str or tuple
    t='double',
    t_half_sub_end='double',
    t_half_sub_start='double',
    Δt_half_sub='double',
    returns='void',
)
def compute_time_step_integrals(Δt_base, components, ᔑdt_substeps, step_type, sync_time):
    # Always operate from the current time
    t = universals.t
    # The base time step should be divided
    # into 2**N_rungs equal half sub-steps.
    Δt_half_sub = Δt_base/ℤ[2**N_rungs]
    # Loop over each integrand
    for integrand, integrals in ᔑdt_substeps.items():
        # For each half sub-step in the base step, compute integrals for
        # every integrand contained in ᔑdt_substeps.
        with unswitch:
            if step_type == 'init':
                # The initial half kick is over the first 2**N_rungs//2
                # half sub-steps (for particles on rung 0).
                index_start = 0
                index_end = ℤ[2**N_rungs//2]
            else:  # step_type == 'full'
                # A full step is over the entire base step Δt, but also
                # half of the next base step. We thus need to know the
                # integrals at 3/2*2**N_rungs half sub-steps. The first
                # 2**N_rungs/2 of these half sub-steps were used in the
                # previous step and are thus already known. We reuse
                # these rather than recomputing them.
                integrals[:ℤ[2**N_rungs//2]] = integrals[ℤ[2**N_rungs]:]
                index_start = ℤ[2**N_rungs//2]
                index_end = ℤ[2**N_rungs//2*3]
        at_sync_time = False
        for half_substep in range(index_start, index_end):
            # Start and end time for this half sub-step
            t_half_sub_start = t + half_substep*Δt_half_sub
            t_half_sub_end = t_half_sub_start + Δt_half_sub
            # If the end time is beyond the sync time,
            # set it equal to the sync time.
            if t_half_sub_end > sync_time:
                at_sync_time = True
                t_half_sub_end = sync_time
                # If the end time is extremely near the start time,
                # it means that integrals for all half sub-steps up
                # to the sync time have already been computed.
                # The integral for the current half sub-step, as well as
                # for all later ones, should be nullified.
                if t_half_sub_end - t_half_sub_start <= Δt_reltol*Δt_base:
                    integrals[half_substep:] = 0
                    break
            # When not using the CLASS background,
            # we need to tabulate a(t) over the sub-step.
            with unswitch:
                if not enable_class_background:
                    expand(
                        scale_factor(t_half_sub_start),
                        t_half_sub_start,
                        t_half_sub_end - t_half_sub_start,
                    )
            # Compute and store the integral
            integrals[half_substep] = scalefactor_integral(
                integrand, t_half_sub_start, t_half_sub_end - t_half_sub_start, components,
            )
            # When at the sync time, integrals for all later
            # half sub-steps should be nullified.
            if at_sync_time:
                integrals[half_substep+1:] = 0
                break

# Function which perform long-range kicks on all components
@cython.header(
    # Arguments
    components=list,
    ᔑdt_substeps=dict,
    step_type=str,
    # Locals
    a='double',
    a_next='double',
    component='Component',
    force=str,
    integrals='double[::1]',
    integrand=object,  # str or tuple
    interactions_list=list,
    method=str,
    printout='bint',
    receivers=list,
    suppliers=list,
    t='double',
    ᔑdt=dict,
    returns='void',
)
def kick_long(components, ᔑdt_substeps, step_type):
    """We take into account three different cases of long-range kicks:
    - Internal source terms (fluid and particle components).
    - Interactions acting on fluids (only PM implemented).
    - Long-range interactions acting on particle components,
      i.e. PM and the long-range part of P³M.
    This function can operate in two separate modes:
    - step_type == 'init':
      The kick is over the first half of the base time step of size Δt.
      This interval consists of the first 2**N_rungs//4 sub-steps
      or equivalently the first 2**N_rungs//2 half sub-steps.
    - step_type == 'full':
      The kick is over the second half of the base time step of size Δt
      as well as over an equally sized portion of the next time step.
      This interval consists of the last 2**N_rungs//2 sub-steps
      or equivalently the last 2**N_rungs half sub-steps,
      i.e. every sub-step but the first 2**N_rungs//2.
    """
    # Construct local dict ᔑdt, mapping each integral to a single
    # number, based on the time step type.
    ᔑdt = {}
    for integrand, integrals in ᔑdt_substeps.items():
        with unswitch:
            if 𝔹[step_type == 'init']:
                ᔑdt[integrand] = sum(integrals[:ℤ[2**N_rungs//2]])
            else:  # step_type == 'full'
                ᔑdt[integrand] = sum(integrals[ℤ[2**N_rungs//2]:])
    # If the time step size is zero, meaning that we are already
    # at a sync time, return now.
    if ᔑdt['1'] == 0:
        return
    # Set t and a to match the time at the beginning of the kick.
    # Note that when the drifts and kicks are out of sync, you should
    # manually set universals.t and universals.a to match the current
    # time for the kicks, prior to calling this function.
    t = universals.t
    a = universals.a
    # Realize all linear fluid scalars which are not components
    # of a tensor. This comes down to ϱ and 𝒫.
    a_next = scale_factor(t + ᔑdt['1'])
    for component in components:
        component.realize_if_linear(0,  # ϱ
            specific_multi_index=0, a=a, a_next=a_next
        )
        component.realize_if_linear(2,  # 𝒫
            specific_multi_index='trace', a=a, a_next=a_next,
        )
    # Apply the effect of all internal source terms
    for component in components:
        component.apply_internal_sources(ᔑdt, a_next)
    # Find all long-range interactions
    interactions_list = interactions.find_interactions(components, 'long-range')
    # Invoke each long-range interaction sequentially
    printout = True
    for force, method, receivers, suppliers in interactions_list:
        getattr(interactions, force)(method, receivers, suppliers, ᔑdt, 'long-range', printout)

# Function which kicks all short-range rungs a single time
@cython.header(
    # Arguments
    components=list,
    ᔑdt_substeps=dict,
    # Locals
    component='Component',
    force=str,
    index_end='Py_ssize_t',
    index_start='Py_ssize_t',
    integrals='double[::1]',
    integrand=object,  # str or tuple
    interactions_list=list,
    lowest_populated_rung='signed char',
    method=str,
    printout='bint',
    receiver='Component',
    receivers=list,
    rung_index='signed char',
    rung_integrals='double[::1]',
    suppliers=list,
    ᔑdt=dict,
    returns='void',
)
def kick_short(components, ᔑdt_substeps):
    """The kick is over the first half of the sub-step for each rung.
    A sub-step for rung i is 1/2**i as long as the base step
    of size Δt, and so half a sub-step is 1/(2*2**i) of the base step.
    """
    # Find all short-range interactions
    interactions_list = interactions.find_interactions(components, 'short-range')
    if not interactions_list:
        return
    # As we only do a single, simultaneous interaction for all rungs,
    # we must flag all rungs as active. For performance reasons, we
    # choose the lowest active rung as the lowest populated rung.
    lowest_populated_rung = ℤ[N_rungs - 1]
    for component in components:
        # Set lowest active rung
        component.lowest_active_rung = component.lowest_populated_rung
        # Lowest populated rung among all components
        if component.lowest_populated_rung < lowest_populated_rung:
            lowest_populated_rung = component.lowest_populated_rung
    lowest_populated_rung = allreduce(lowest_populated_rung, op=MPI.MIN)
    # Though the size of the time interval over which to kick is
    # different for each rung, we only perform a single interaction
    # for each pair of components and short-range forces.
    # We then need to know all of the N_rungs time step integrals for
    # each integrand simultaneously. Here we store these as
    # ᔑdt[integrand][rung_index]. We reuse the global ᔑdt_rungs as a
    # container (no content will be reused). If this is the first call,
    # we first populate this with arrays for each integrand.
    ᔑdt = ᔑdt_rungs
    if not ᔑdt:
        for integrand in ᔑdt_substeps:
            # We need a value of the integral for each rung,
            # of which there are N_rungs. Additionally, we need a value
            # of the integral for jumping down/up a rung, for each rung,
            # meaning that we need 3*N_rungs integrals. The integral
            # for a normal kick of rung rung_index is then stored in
            # ᔑdt[integrand][rung_index], while the integral for jumping
            # down from rung rung_index to rung_index - 1 is stored in
            # ᔑdt[integrand][rung_index + N_rungs], while the integral
            # for jumping up from rung_index to rung_index + 1 is stored
            # in ᔑdt[integrand][rung_index + 2*N_rungs]. Since a
            # particle at rung 0 cannot jump down and a particle at rung
            # N_rungs - 1 cannot jump up, indices 0 + N_rungs = N_rungs
            # and N_rungs - 1 + 2*N_rungs = 3*N_rungs - 1 are unused.
            # We allocate 3*N_rungs - 1 integrals, leaving the unused
            # index N_rungs be, while the unused index 3*N_rungs - 1
            # will be out of bounce.
            ᔑdt[integrand] = zeros(3*N_rungs - 1, dtype=C2np['double'])
    for integrand, integrals in ᔑdt_substeps.items():
        rung_integrals = ᔑdt[integrand]
        for rung_index in range(ℤ[N_rungs - 1], lowest_populated_rung - 1, -1):
            index_end = 2**(ℤ[N_rungs - 1] - rung_index)
            index_start = index_end//2
            rung_integrals[rung_index] = sum(integrals[index_start:index_end])
            if rung_index != ℤ[N_rungs - 1]:
                rung_integrals[rung_index] += rung_integrals[rung_index + 1]
    # Invoke short-range interactions
    printout = True
    for force, method, receivers, suppliers in interactions_list:
        getattr(interactions, force)(method, receivers, suppliers, ᔑdt, 'short-range', printout)
# Dict storing time step integrals used by the kick_short()
# and driftkick_short() functions. This holds time step integrals for
# each rung and is indexed by ᔑdt_rungs[integrand][rung_index].
cython.declare(ᔑdt_rungs=dict)
ᔑdt_rungs = {}

# Function which drifts all fluid components
@cython.header(
    # Arguments
    components=list,
    ᔑdt_substeps=dict,
    # Locals
    a_next='double',
    component='Component',
    fluid_components=list,
    integrals='double[::1]',
    integrand=object,  # str or tuple
    t='double',
    ᔑdt=dict,
    returns='void',
)
def drift_fluids(components, ᔑdt_substeps):
    """This function always drift over a full base time step,
    consisting of 2**N_rungs half sub-steps. If you wish to e.g. only
    drift over the first half, you should nullify the last 2**N_rungs
    integrals in ᔑdt_substeps before calling this function.
    """
    # Collect all fluid components. Do nothing if none exists.
    fluid_components = [
        component for component in components if component.representation == 'fluid'
    ]
    if not fluid_components:
        return
    # Construct local dict ᔑdt,
    # mapping each integral to a single number.
    ᔑdt = {}
    for integrand, integrals in ᔑdt_substeps.items():
        ᔑdt[integrand] = sum(integrals[:ℤ[2**N_rungs]])
    # If the time step size is zero, meaning that we are already
    # at a sync time, return now.
    if ᔑdt['1'] == 0:
        return
    # Drift all fluid components sequentially
    t = universals.t
    a_next = scale_factor(t + ᔑdt['1'])
    for component in fluid_components:
        component.drift(ᔑdt, a_next)

# Function which perform interlaced drift and kick operations
# on the short-range rungs.
@cython.header(
    # Arguments
    components=list,
    Δt='double',
    ᔑdt_substeps=dict,
    Δt_downjump_fac='double',
    # Locals
    any_kicks='bint',
    any_rung_jumps='bint',
    any_rung_jumps_list=list,
    component='Component',
    driftkick_index='Py_ssize_t',
    force=str,
    i='Py_ssize_t',
    index_end='Py_ssize_t',
    index_start='Py_ssize_t',
    integrals='double[::1]',
    integrand=object,  # str or tuple
    interactions_list=list,
    lowest_active_rung='signed char',
    message=list,
    method=str,
    particle_components=list,
    printout='bint',
    receivers=list,
    rung_index='signed char',
    rung_integrals='double[::1]',
    suppliers=list,
    text=str,
    ᔑdt=dict,
    returns='void',
)
def driftkick_short(components, Δt, ᔑdt_substeps, Δt_downjump_fac):
    """Every rung is fully drifted and kicked over a complete base time
    step of size Δt. Rung i will be kicked 2**i times.
    All rungs will be drifted synchronously in steps
    of Δt/(2**N_rungs//2), i.e. each drift is over two half sub-steps.
    The first drift will start at the beginning of the base step.
    The kicks will vary in size for the different rungs. Rung i will
    be kicked Δt/(2**i) in each kick operation, i.e. a whole sub-step
    for the highest rung (N_rungs - 1), two sub-steps for the rung
    below, four sub-steps for the rung below that, and so on.
    It as assumed that all rungs have already been kicked so that
    these are half a kick-sized step ahead of the drifts. Thus, the
    kick position of the highest rung is already half a sub-step into
    the base time step, the rung below is two half sub-steps into the
    base time step, the rung below that is four half sub-steps into the
    base step, and so on.
    The drifts and kicks follow this rhythm:
      - drift all
      - kick rung  (N_rungs - 1)
      - drift all
      - kick rungs (N_rungs - 1), (N_rungs - 2)
      - drift all
      - kick rung  (N_rungs - 1)
      - drift all
      - kick rungs (N_rungs - 1), (N_rungs - 2), (N_rungs - 3)
      - ...
    Thus the highest rung participates in all kicks, the one below only
    in every other kick, the one below that only in every fourth kick,
    and so on.
    """
    # Collect all particle components. Do nothing if none exists.
    particle_components = [
        component for component in components if component.representation == 'particles'
    ]
    if not particle_components:
        return
    # Find all short-range interactions
    interactions_list = interactions.find_interactions(components, 'short-range')
    # In case of no short-range interactions among the particles at all,
    # we may drift the particles in one go, after which we are done
    # within this function, as the long-range kicks
    # are handled elsewhere.
    if not interactions_list:
        # Construct local dict ᔑdt,
        # mapping each integral to a single number.
        ᔑdt = {}
        for integrand, integrals in ᔑdt_substeps.items():
            ᔑdt[integrand] = sum(integrals[:ℤ[2**N_rungs]])
        # If the time step size is zero, meaning that we are already
        # at a sync time, return now.
        if ᔑdt['1'] == 0:
            return
        for component in particle_components:
            masterprint(f'Drifting {component.name} ...')
            component.drift(ᔑdt)
            masterprint('done')
        return
    # We have short-range interactions.
    # Prepare progress message.
    message = [
        f'Intertwining drifts of {particle_components[0].name} with '
        f'the following particle interactions:'
        if len(particle_components) == 1 else (
           'Intertwining drifts of {{{}}} with the following particle interactions:'
            .format(', '.join([component.name for component in particle_components]))
        )
    ]
    for force, method, receivers, suppliers in interactions_list:
        text = interactions.shortrange_progress_messages(force, method, receivers)
        message.append(text[0].upper() + text[1:])
    printout = True
    # Perform the interlaced drifts and kicks
    any_kicks = True
    for driftkick_index in range(ℤ[2**N_rungs//2]):
        # Fill in local dict ᔑdt, mapping each integral over the drift
        # to a single number.
        if any_kicks:
            # We nullify ᔑdt only after a kick.
            # In this way, successive drifts with no kicks in between
            # can be performed in one go.
            ᔑdt = {}
            for integrand in ᔑdt_substeps:
                ᔑdt[integrand] = 0
        index_start = 2*driftkick_index
        index_end = index_start + 2
        for integrand, integrals in ᔑdt_substeps.items():
            ᔑdt[integrand] += sum(integrals[index_start:index_end])
        # Determine the lowest active rung
        # (the lowest rung which should receive a kick).
        # All rungs above this should be kicked as well.
        for rung_index in range(N_rungs):
            if ℤ[driftkick_index + 1] % 2**(ℤ[N_rungs - 1] - rung_index) == 0:
                lowest_active_rung = rung_index
                break
        # Set lowest active rung for each component
        # and check if any kicks are to be performed.
        any_kicks = False
        for component in components:
            # There is no need to have the lowest active rung
            # be below the lowest populated rung.
            if lowest_active_rung < component.lowest_populated_rung:
                component.lowest_active_rung = component.lowest_populated_rung
            else:
                component.lowest_active_rung = lowest_active_rung
            # Flag if any particles exist on active rungs
            if component.highest_populated_rung >= component.lowest_active_rung:
                any_kicks = True
        any_kicks = allreduce(any_kicks, op=MPI.LOR)
        # Skip the kick if no particles at all occupy active rungs.
        # The drift is not skipped, as we do not overwrite the values
        # in ᔑdt, but add to them.
        if not any_kicks:
            continue
        # A kick is to be performed. First do the drift.
        # If the time step size is zero, meaning that we are already
        # at a sync time regarding the drifts, we skip the drift but
        # do not return, as the kicks may still not be at the sync time.
        if ᔑdt['1'] != 0:
            for component in particle_components:
                component.drift(ᔑdt)
        # Though the size of the time interval over which to kick is
        # different for each rung, we perform the kicks of the
        # N_rungs - lowest_active_rung lowest rungs using a single
        # interaction for each pair of components and
        # short-range forces. We then need to know all of the
        # N_rungs - lowest_active_rung time step integrals for each
        # integrand simultaneously. Here we store these
        # as ᔑdt[integrand][rung_index]. We reuse the global ᔑdt_rungs
        # as a container (no content will be reused).
        ᔑdt = ᔑdt_rungs
        for integrand, integrals in ᔑdt_substeps.items():
            rung_integrals = ᔑdt[integrand]
            for rung_index in range(lowest_active_rung, N_rungs):
                index_start = (
                    ℤ[2**(N_rungs - 1 - rung_index)]
                    + (driftkick_index//ℤ[2**(N_rungs - 1 - rung_index)]
                        )*ℤ[2**(N_rungs - rung_index)]
                )
                index_end = index_start + ℤ[2**(N_rungs - rung_index)]
                rung_integrals[rung_index] = sum(integrals[index_start:index_end])
                # We additionally need the integral for jumping down
                # from rung_index to rung_index - 1. We store this using
                # index (rung_index + N_rungs). For any given rung, such
                # a down-jump is only allowed every second kick. When
                # disallowed, we store -1.
                if rung_index > 0 and (
                    (ℤ[driftkick_index + 1] - ℤ[2**(N_rungs - 1 - rung_index)]
                        ) % 2**(N_rungs - rung_index) == 0
                ):
                    index_end = index_start + ℤ[2**(N_rungs - 1 - rung_index)]
                    rung_integrals[rung_index + N_rungs] = sum(integrals[index_start:index_end])
                else:
                    rung_integrals[rung_index + N_rungs] = -1
                # We additionally need the integral for jumping up
                # from rung_index to rung_index + 1.
                if rung_index < ℤ[N_rungs - 1]:
                    index_end = index_start + 3*2**(ℤ[N_rungs - 2] - rung_index)
                    rung_integrals[rung_index + ℤ[2*N_rungs]] = sum(
                        integrals[index_start:index_end])
        # Perform short-range kicks, unless the time step size is zero
        # for all active rungs (i.e. they are all at a sync time),
        # in wich case we go to the next (drift) sub-step.  We cannot
        # just return, as the kicks may still not be at the sync time.
        rung_integrals = ᔑdt['1']
        if sum(rung_integrals[lowest_active_rung:N_rungs]) == 0:
            continue
        # A short-range kick is to be performed
        any_rung_jumps_list = []
        for component in particle_components:
            # Flag inter-rung jumps for each particle.
            # The list any_rung_jumps_list will store booleans telling
            # whether or not any local particles of this component
            # jump rung.
            any_rung_jumps_list.append(
                component.flag_rung_jumps(Δt, rung_integrals, fac_softening, Δt_downjump_fac)
            )
        if printout:
            # This is the first kick. Print out progress message.
            masterprint(message[0])
            for text in message[1:]:
                masterprint(text, indent=4)
            masterprint('...', indent=4, wrap=False)
            printout = False
        for force, method, receivers, suppliers in interactions_list:
            # Perform interaction
            getattr(interactions, force)(
                method, receivers, suppliers, ᔑdt, 'short-range', printout)
        # Apply inter-rung jumps
        for component, any_rung_jumps in zip(particle_components, any_rung_jumps_list):
            if any_rung_jumps:
                component.apply_rung_jumps()
    # Finalize the progress message. If printout is True, no message
    # was ever printed (because there were no kicks).
    if not printout:
        masterprint('done')


# Function which dump all types of output
@cython.header(
    # Arguments
    components=list,
    output_filenames=dict,
    dump_time=object,  # collections.namedtuple
    # Locals
    filename=str,
    time_param=str,
    time_value='double',
    returns='void',
)
def dump(components, output_filenames, dump_time):
    time_param = dump_time.time_param
    time_value = {'t': dump_time.t, 'a': dump_time.a}[time_param]
    # Dump render2D
    if time_value in render2D_times[time_param]:
        filename = output_filenames['render2D'].format(time_param, time_value)
        if time_param == 't':
            filename += unit_time
        render2D(components, filename)
    # Dump snapshot
    if time_value in snapshot_times[time_param]:
        filename = output_filenames['snapshot'].format(time_param, time_value)
        if time_param == 't':
            filename += unit_time
        save(components, filename)
    # Dump power spectrum
    if time_value in powerspec_times[time_param]:
        filename = output_filenames['powerspec'].format(time_param, time_value)
        if time_param == 't':
            filename += unit_time
        powerspec(components, filename)
    # Dump render3D
    if time_value in render3D_times[time_param]:
        filename = output_filenames['render3D'].format(time_param, time_value)
        if time_param == 't':
            filename += unit_time
        render3D(components, filename)

# Function which dump all types of output
@cython.header(
    # Arguments
    components=list,
    time_step='Py_ssize_t',
    Δt='double',
    Δt_begin='double',
    # Locals
    autosave_params_filename=str,
    autosave_filename=str,
    remaining_output_times=dict,
    param_lines=list,
    present='double',
    time_param=str,
    returns='void',
)
def autosave(components, time_step, Δt, Δt_begin):
    masterprint('Autosaving ...')
    autosave_filename        = f'{autosave_dir}/autosave_{jobid}.hdf5'
    autosave_params_filename = f'{paths["params_dir"]}/autosave_{jobid}.params'
    # Save parameter file corresponding to the snapshot
    if master:
        masterprint(f'Writing parameter file "{autosave_params_filename}" ...')
        with disable_numpy_summarization():
            param_lines = []
            # Header
            param_lines += [
                f'# This parameter file is the result of an autosave of job {jobid},',
                f'# which uses the parameter file "{paths["params"]}".',
                f'# The autosave was carried out {datetime.datetime.now()}.',
                f'# The following is a copy of this original parameter file.',
            ]
            param_lines += ['']*2
            # Original parameter file
            param_lines += params_file_content.split('\n')
            param_lines += ['']*2
            # IC snapshot
            param_lines += [
                f'# The autosaved snapshot file was saved to',
                f'initial_conditions = "{autosave_filename}"',
            ]
            # Present time
            param_lines.append(f'# The autosave happened at time')
            if enable_Hubble:
                param_lines.append(f'a_begin = {universals.a:.16e}')
            else:
                param_lines.append(f't_begin = {universals.t:.16e}*{unit_time}')
            # Time step, current and original time step size
            param_lines += [
                f'# The time step and time step size was',
                f'initial_time_step = {time_step + 1}',
                f'{unicode("Δt_autosave")} = {Δt:.16e}*{unit_time}',
                f'# The time step size at the beginning of the simulation was',
                f'{unicode("Δt_begin_autosave")} = {Δt_begin:.16e}*{unit_time}',
            ]
            # All output times
            param_lines += [
                f'# All output times',
                f'output_times_full = {output_times}',
            ]
            # Remaining output times
            remaining_output_times = {'a': {}, 't': {}}
            for time_param, present in zip(('a', 't'), (universals.a, universals.t)):
                for output_kind, output_time in output_times[time_param].items():
                    remaining_output_times[time_param][output_kind] = [
                        ot for ot in output_time if ot >= present
                    ]
            param_lines += [
                f'# Remaining output times',
                f'output_times = {remaining_output_times}',
            ]
        # Write to parameter file
        with open(autosave_params_filename, 'w', encoding='utf-8') as autosave_params_file:
            print('\n'.join(param_lines), file=autosave_params_file)
        masterprint('done')
    # Save standard snapshot. Include all components regardless
    # of the snapshot_select user parameter.
    save(components, autosave_filename, snapshot_type='standard', save_all_components=True)
    # If this simulation run was started from an autosave snapshot
    # with a different name from the one just saved, remove this
    # now superfluous autosave snapshot.
    if master:
        if (    isinstance(initial_conditions, str)
            and re.search(r'^autosave_\d+\.hdf5$', os.path.basename(initial_conditions))
            and os.path.abspath(initial_conditions) != os.path.abspath(autosave_filename)
            and os.path.isfile(initial_conditions)
        ):
            os.remove(initial_conditions)
    masterprint('done')

# Function which prints out basic information
# about the current time step.
@cython.header(
    # Arguments
    time_step='Py_ssize_t',
    Δt='double',
    bottleneck=str,
    components=list,
    end='bint',
    # Locals
    component='Component',
    i='Py_ssize_t',
    last_populated_rung='signed char',
    part=str,
    parts=list,
    rung_index='signed char',
    rung_N='Py_ssize_t',
    width='Py_ssize_t',
    width_max='Py_ssize_t',
    returns='void',
)
def print_timestep_heading(time_step, Δt, bottleneck, components, end=False):
    global heading_ljust, timestep_heading_last_time_step
    if timestep_heading_last_time_step == time_step:
        return
    timestep_heading_last_time_step = time_step
    # Create list of text pieces. Left justify the first column
    # according to the global heading_ljust.
    parts = []
    parts.append('\nEnd of main time loop' if end else terminal.bold(f'\nTime step {time_step}'))
    if enable_Hubble:
        parts.append('\nScale factor:'.ljust(heading_ljust))
        parts.append(significant_figures(universals.a, 4, fmt='unicode'))
    if enable_Hubble:
        parts.append('\nCosmic time:'.ljust(heading_ljust))
    else:
        parts.append('\nTime:'.ljust(heading_ljust))
    parts.append(f'{{}} {unit_time}'.format(significant_figures(universals.t, 4, fmt='unicode')))
    if not end:
        parts.append('\nStep size:'.ljust(heading_ljust))
        parts.append(f'{{}} {unit_time}'.format(significant_figures(Δt, 4, fmt='unicode')))
        if bottleneck:
            parts.append(f' (limited by {bottleneck})')
    # Equation of state of each component
    for component in components:
        if (component.w_type != 'constant'
            and 'metric' not in component.class_species
            and 'lapse'  not in component.class_species
        ):
            parts.append(f'\nEoS w ({component.name}): ')
            parts.append(significant_figures(component.w(), 4, fmt='unicode'))
    # Rung population for each component
    for component in components:
        if not component.use_rungs:
            continue
        parts.append(f'\nRung population ({component.name}): ')
        rung_population = []
        last_populated_rung = 0
        for rung_index in range(N_rungs):
            rung_N = allreduce(component.rungs_N[rung_index], op=MPI.SUM)
            rung_population.append(str(rung_N))
            if rung_N > 0:
                last_populated_rung = rung_index
        parts.append(', '.join(rung_population[:last_populated_rung+1]))
    # Find the maximum width of the first column and left justify
    # the entire first colum to match this maximum width.
    if heading_ljust == 0:
        width_max = 0
        for part in parts:
            if (   'Scale factor:' in part
                or 'Cosmic time:'  in part
                or 'Time:'         in part
                or 'Step size:'    in part
            ):
                width = len(part)
                if width > width_max:
                    width_max = width
        heading_ljust = width_max + 1
        for i, part in enumerate(parts):
            if part.endswith(':'):
                parts[i] = part.ljust(heading_ljust)
    # Print out the combined heading
    masterprint(''.join(parts))
cython.declare(
    heading_ljust='Py_ssize_t',
    timestep_heading_last_time_step='Py_ssize_t',
)
heading_ljust = 0
timestep_heading_last_time_step = -1

# Function which checks the sanity of the user supplied output times,
# creates output directories and defines the output filename patterns.
# A Python function is used because it contains a closure
# (a lambda function).
def prepare_for_output():
    """As this function uses universals.t and universals.a as the
    initial values of the cosmic time and the scale factor, you must
    initialize these properly before calling this function.
    """
    # Check that the output times are legal
    for time_param, at_begin in zip(('a', 't'), (universals.a, universals.t)):
        for output_kind, output_time in output_times[time_param].items():
            if output_time and np.min(output_time) < at_begin:
                message = [
                    f'Cannot produce a {output_kind} at {time_param} '
                    f'= {np.min(output_time):.6g}'
                ]
                if time_param == 't':
                    message.append(f' {unit_time}')
                message.append(f', as the simulation starts at {time_param} = {at_begin:.6g}')
                if time_param == 't':
                    message.append(f' {unit_time}')
                message.append('.')
                abort(''.join(message))
    # Create output directories if necessary
    if master:
        for time_param in ('a', 't'):
            for output_kind, output_time in output_times[time_param].items():
                # Do not create directory if this kind of output
                # should never be dumped to the disk.
                if not output_time or not output_kind in output_dirs:
                    continue
                # Create directory
                output_dir = output_dirs[output_kind]
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
    Barrier()
    # Construct the patterns for the output file names. This involves
    # determining the number of digits of the scalefactor in the output
    # filenames. There should be enough digits so that adjacent dumps do
    # not overwrite each other, and so that the name of the first dump
    # differs from that of the IC, should it use the same
    # naming convention.
    output_filenames = {}
    for time_param, at_begin in zip(('a', 't'), (universals.a, universals.t)):
        # Here the output_times_full dict is used rather than just the
        # output_times dict. These dicts are equal, except after
        # starting from an autosave, where output_times will contain
        # the remaining dump times only, whereas output_times_full
        # will contain all the original dump times.
        # We use output_times_full so as to stick to the original naming
        # format used before restarting from the autosave.
        for output_kind, output_time in output_times_full[time_param].items():
            # This kind of output does not matter if
            # it should never be dumped to the disk.
            if not output_time or not output_kind in output_dirs:
                continue
            # Compute number of digits
            times = sorted(set((at_begin, ) + output_time))
            ndigits = 0
            while True:
                fmt = f'{{:.{ndigits}f}}'
                if (len(set([fmt.format(ot) for ot in times])) == len(times)
                    and (fmt.format(times[0]) != fmt.format(0) or not times[0])):
                    break
                ndigits += 1
            fmt = f'{{}}={fmt}'
            # Use the format (that is, either the format from the a
            # output times or the t output times) with the largest
            # number of digits.
            if output_kind in output_filenames:
                if int(re.search(
                    '[0-9]+',
                    re.search('{.+?}', output_filenames[output_kind]).group(),
                ).group()) >= ndigits:
                    continue
            # Store output name patterns
            output_dir = output_dirs[output_kind]
            output_base = output_bases[output_kind]
            sep = '_' if output_base else ''
            output_filenames[output_kind] = f'{output_dir}/{output_base}{sep}{fmt}'
    # Lists of sorted dump times of both kinds
    a_dumps = sorted(set([nr for val in output_times['a'].values() for nr in val]))
    t_dumps = sorted(set([nr for val in output_times['t'].values() for nr in val]))
    # Combine a_dumps and t_dumps into a single list of named tuples
    Dump_time = collections.namedtuple(
        'Dump_time', ('time_param', 't', 'a')
    )
    dump_times =  [Dump_time('t', t=t_dump, a=None) for t_dump in t_dumps]
    dump_times += [Dump_time('a', a=a_dump, t=None) for a_dump in a_dumps]
    if enable_Hubble:
        a_lower = t_lower = machine_ϵ
        for i, dump_time in enumerate(dump_times):
            if dump_time.time_param == 't' and dump_time.a is None:
                a = scale_factor(dump_time.t)
                dump_time = Dump_time('t', t=dump_time.t, a=a)
            elif dump_time.time_param == 'a' and dump_time.t is None:
                t = cosmic_time(dump_time.a, a_lower, t_lower)
                dump_time = Dump_time('a', a=dump_time.a, t=t)
                a_lower, t_lower = dump_time.a, dump_time.t
            dump_times[i] = dump_time
    # Sort the list according to the cosmic time
    dump_times = sorted(dump_times, key=(lambda dump_time: dump_time.t))
    # Two dump times at the same or very near the same time
    # should count as one.
    if len(dump_times) > 1:
        dump_time = dump_times[0]
        dump_times_unique = [dump_time]
        t_previous = dump_time.t
        for dump_time in dump_times[1:]:
            if not np.isclose(dump_time.t, t_previous, rtol=1e-6, atol=0):
                dump_times_unique.append(dump_time)
                t_previous = dump_time.t
        dump_times = dump_times_unique
    return dump_times, output_filenames



# Here we set the values for the various factors used when determining
# the time step size. The values given below has been tuned by hand as
# to achieve a matter power spectrum at a = 1 that has converged to
# within 1% on all scales, for Δt_base_factor = Δt_rung_factor = 1.
# For further specification of each factor,
# consult the get_base_timestep_size() function.
cython.declare(
    fac_dynamical='double',
    fac_hubble='double',
    fac_ẇ='double',
    fac_Γ='double',
    fac_courant='double',
    fac_pm='double',
    fac_p3m='double',
    fac_softening='double',
)
# The base time step should be below the dynamic time scale
# times this factor.
fac_dynamical = 0.042*Δt_base_factor
# The base time step should be below the current Hubble time scale
# times this factor.
fac_hubble = 0.12*Δt_base_factor
# The base time step should be below |ẇ|⁻¹ times this factor,
# for all components. Here w is the equation of state parameter.
fac_ẇ = 0.0017*Δt_base_factor
# The base time step should be below |Γ|⁻¹ times this factor,
# for all components. Here Γ is the decay rate.
fac_Γ = 0.0028*Δt_base_factor
# The base time step should be below that set by the 1D Courant
# condition times this factor, for all fluid components.
fac_courant = 0.14*Δt_base_factor
# The base time step should be small enough so that particles
# participating in interactions using the PM method do not drift further
# than the size of one PM grid cell times this factor in a single
# time step. The same condition is applied to fluids, where the bulk
# velocity is what counts (i.e. we ignore the sound speed).
fac_pm = 0.032*Δt_base_factor
# The base time step should be small enough so that particles
# participating in interactions using the P³M method do not drift
# further than the long/short-range force split scale times this factor
# in a single time step.
fac_p3m = 0.027*Δt_base_factor
# When using adaptive time stepping (N_rungs > 1), the individual time
# step size for a given particle must not be so large that it drifts
# further than its softening length times this factor. If it does become
# large enough for this, the particle jumps to the rung just above
# its current rung.
fac_softening = 2.0*Δt_rung_factor

# If this module is run properly (detected by jobid being set),
# launch the CO𝘕CEPT run.
if jobid != -1:
    if 'special' in special_params:
        # Instead of running a simulation, run some utility
        # as defined by the special_params dict.
        delegate()
    else:
        # Run the time loop
        timeloop()
        # Simulation done
        universals.any_warnings = allreduce(universals.any_warnings, op=MPI.LOR)
        if universals.any_warnings:
            masterprint(f'CO𝘕CEPT run {jobid} finished')
        else:
            masterprint(f'CO𝘕CEPT run {jobid} finished successfully', fun=terminal.bold_green)
    # Shutdown CO𝘕CEPT properly
    abort(exit_code=0)
