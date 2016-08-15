# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2016 Jeppe Mosgaard Dakin.
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
from mesh import diff
from snapshot import load
cimport('from analysis import debug, powerspec')
cimport('from graphics import render, terminal_render')
cimport('from gravity import build_φ')
cimport('from integration import cosmic_time, expand, initiate_time, scalefactor_integral')
cimport('from utilities import delegate')
cimport('from snapshot import save')



# Function that computes several time integrals with integrands having
# to do with the scale factor (e.g. ∫dta⁻¹, ∫dtȧ/a)
# The result is stored in ᔑdt_steps[integrand][index],
# where index == 0 corresponds to step == 'first half' and
# index == 1 corresponds to step == 'second half'. 
@cython.header(# Arguments
               step='str',
               # Locals
               a_next='double',
               index='int',
               integrand='str',
               t_next='double',
               )
def scalefactor_integrals(step):
    global ᔑdt_steps, Δt
    # Update the scale factor and the cosmic time. This also
    # tabulates a(t), needed for the scalefactor integrals.
    a_next = expand(universals.a, universals.t, 0.5*Δt)
    t_next = universals.t + 0.5*Δt
    if t_next + 1e-3*Δt > next_dump[1]:
        # Case 1: Dump time reached and exceeded.
        # A smaller time step than
        # 0.5*Δt is needed to hit dump time exactly. 
        # Case 2: Dump time very nearly reached.
        # Go directly to dump time (otherwize the next time step wilĺ
        # be very small).
        t_next = next_dump[1]
        # Find a_next = a(t_next) and tabulate a(t)
        a_next = expand(universals.a, universals.t, t_next - universals.t)
        if next_dump[0] == 'a':
            # This should be the same as the result above,
            # but this is included to ensure agreement of future
            # floating point comparisons.
            a_next = next_dump[2]
    # Update the universal scale factor and cosmic time
    universals.a, universals.t = a_next, t_next
    # Map the step string to the index integer
    if step == 'first half':
        index = 0
    elif step == 'second half':
        index = 1
    elif master:
        abort('The value "{}" was given for the step'.format(step))
    # Do the scalefactor integrals
    for integrand in ᔑdt_steps:
        ᔑdt_steps[integrand][index] = scalefactor_integral(integrand)

# Function which dump all types of output. The return value signifies
# whether or not something has been dumped.
@cython.header(# Arguments
               components='list',
               output_filenames='dict',
               final_render='tuple',
               op='str',
               # Locals
               filename='str',
               time_param='str',
               time_val='double',
               returns='bint',
               )
def dump(components, output_filenames, final_render, op=None):
    global i_dump, dumps, next_dump
    # Do nothing if not at dump time
    if universals.t != next_dump[1]:
        if next_dump[0] == 'a':
            if universals.a != next_dump[2]:
                return False
        else:
            return False
    # Synchronize positions and momenta before dumping
    if op == 'drift':
        drift(components, 'first half')
    elif op == 'kick':
        kick(components, 'second half')
    # Dump terminal render
    for time_val, time_param in zip((universals.a, universals.t), ('a', 't')):
        if time_val in terminal_render_times[time_param]:
            terminal_render(components)
    # Dump snapshot
    for time_val, time_param in zip((universals.a, universals.t), ('a', 't')):
        if time_val in snapshot_times[time_param]:
            filename = output_filenames['snapshot'].format(time_param, time_val)
            if time_param == 't':
                filename += unit_time
            save(components, filename)
    # Dump powerspectrum
    for time_val, time_param in zip((universals.a, universals.t), ('a', 't')):
        if time_val in powerspec_times[time_param]:
            filename = output_filenames['powerspec'].format(time_param, time_val)
            if time_param == 't':
                filename += unit_time
            powerspec(components, filename)
    # Dump render
    for time_val, time_param in zip((universals.a, universals.t), ('a', 't')):
        if time_val in render_times[time_param]:
            filename = output_filenames['render'].format(time_param, time_val)
            if time_param == 't':
                filename += unit_time
            render(components, filename,
                   cleanup=((time_param, time_val) == final_render))
    # Increment dump time
    i_dump += 1
    if i_dump < len(dumps):
        next_dump = dumps[i_dump]
    return True

@cython.header(# Locals
               integrand='str',
               index='int',
               )
def nullify_ᔑdt_steps():
    # Reset (nullify) the ᔑdt_steps, making the next kick operation
    # apply for only half a step, even though 'whole' is used.
    for integrand in ᔑdt_steps:
        for index in range(2):
            ᔑdt_steps[integrand][index] = 0

# Function which kick all of the components
@cython.header(# Arguments
               components='list',
               step='str',
               # Locals
               ᔑdt='dict',
               component='Component',
               component_group='list',
               component_groups='object',  # collections.defaultdict
               dim='int',
               meshbuf_mv='double[:, :, ::1]',
               h='double',
               integrand='str',
               key='str',
               φ='double[:, :, ::1]',
               )
def kick(components, step):
    # Regardless of species, a 'kick' is always defined as the
    # gravitational interaction.
    if not enable_gravity:
        return
    # Construct the local dict ᔑdt,
    # based on which type of step is to be performed.
    ᔑdt = {}
    for integrand in ᔑdt_steps:
        if step == 'first half':
            ᔑdt[integrand] = ᔑdt_steps[integrand][0]
        elif step == 'second half':
            ᔑdt[integrand] = ᔑdt_steps[integrand][1]
        elif step == 'whole':
            ᔑdt[integrand] = np.sum(ᔑdt_steps[integrand])
        elif master:
            abort('The value "{}" was given for the step'.format(step))
    # Group the components based on assigned kick algorithms
    # (for particles). Group all fluids together.
    component_groups = collections.defaultdict(list)
    for component in components:
        if component.representation == 'particles':
            if master and component.species not in kick_algorithms:
                abort('Species "{}" do not have an assigned kick algorithm!'.format(component.species))
            component_groups[kick_algorithms[component.species]].append(component)
        elif component.representation == 'fluid':
            component_groups['fluid'].append(component)
    # First let the components (that needs to) interact
    # with the gravitationak potential.
    if 'PM' in component_groups or 'fluid' in component_groups:
        # Construct the potential φ due to all components
        φ = build_φ(components)
        # Print combined progress message, as all these kicks are done
        # simultaneously for all the components.
        if 'PM' in component_groups and not 'fluid' in component_groups:
            masterprint('Kicking (PM) {} ...'
                        .format(', '.join([component.name
                                           for component in component_groups['PM']])
                                )
                        )
        elif 'PM' not in component_groups and 'fluid' in component_groups:
            masterprint('Kicking (potential only) {} ...'
                        .format(', '.join([component.name
                                           for component in component_groups['fluid']])
                                )
                        )
        else:
            masterprint('Kicking (PM) {} and (potential only) {} ...'
                        .format(', '.join([component.name
                                           for component in component_groups['PM']]),
                                ', '.join([component.name
                                           for component in component_groups['fluid']])
                                )
                        )
        # For each dimension, differentiate φ and apply the force to
        # all components which interact with φ (particles using the PM
        # method and all fluids).
        h = boxsize/φ_gridsize  # Physical grid spacing of φ
        for dim in range(3):
            # Do the differentiation of φ
            meshbuf_mv = diff(φ, dim, h, order=4)
            # Apply PM kick
            for component in component_groups['PM']:
                component.kick(ᔑdt, meshbuf_mv, dim)
            # Apply kick to fluids
            for component in component_groups['fluid']:
                component.kick(ᔑdt, meshbuf_mv, dim)
        # Done with potential interactions
        masterprint('done')
    # Now kick all other components sequentially
    for key, component_group in component_groups.items():
        if key in ('PM', 'fluid'):
            continue
        for component in component_group:
            component.kick(ᔑdt)

# Function which drift all of the components
@cython.header(# Arguments
               components='list',
               step='str',
               # Locals
               ᔑdt='dict',
               component='Component',
               )
def drift(components, step):
    # Construct the local dict ᔑdt,
    # based on which type of step is to be performed.
    ᔑdt = {}
    for integrand in ᔑdt_steps:
        if step == 'first half':
            ᔑdt[integrand] = ᔑdt_steps[integrand][0]
        elif step == 'second half':
            ᔑdt[integrand] = ᔑdt_steps[integrand][1]
        elif step == 'whole':
            ᔑdt[integrand] = np.sum(ᔑdt_steps[integrand])
        elif master:
            abort('The value "{}" was given for the step'.format(step))
    # Drift all components sequentially
    for component in components:
        component.drift(ᔑdt)

# Function containing the main time loop of CO𝘕CEPT
@cython.header(# Locals
               component='Component',
               components='list',
               dim='int',
               final_render='tuple',
               kick_algorithm='str',
               output_filenames='dict',
               timestep='Py_ssize_t',
               Δt_update_freq='Py_ssize_t',
               )
def timeloop():
    global ᔑdt_steps, i_dump, next_dump, Δt
    # Get the output filename patterns
    # and create the global list "dumps".
    # Avoid sanity checks by setting a = t = -∞.
    universals.a = universals.t = -inf
    output_filenames, final_render = prepare_output_times()
    # Do nothing if no dump times exist
    if len(dumps) == 0: 
        return
    # Determine the correct initial values for the cosmic time
    # universals.t and the scale factor a(universals.t) = universals.a.
    initiate_time()
    # Get the output filename patterns once again
    # and recreate the global list "dumps", this time with proper
    # values of universals.a and universals.t.
    output_filenames, final_render = prepare_output_times()
    # Load initial conditions
    components = load(IC_file, only_components=True)
    # The number of time steps before Δt is updated
    Δt_update_freq = 10
    # The time step size
    if enable_Hubble:
        # The time step size should be a
        # small fraction of the age of the universe. 
        Δt = Δt_factor*universals.t
    else:
        # Simply divide the simulation time span into equal chunks
        Δt = Δt_factor*(dumps[len(dumps) - 1][1] - universals.t)
    # Reduce time step size if it is too large for any component
    Δt = reduce_Δt(components, Δt, give_notice=False)
    # Arrays containing the factors ∫_t^(t + Δt/2) integrand(a) dt
    # for different integrands. The two elements in each variable are
    # the first and second half of the factor for the entire time step.
    ᔑdt_steps = {'a⁻¹': zeros(2, dtype=C2np['double']),
                 'a⁻²': zeros(2, dtype=C2np['double']),
                 'ȧ/a': zeros(2, dtype=C2np['double']),
                 }
    # Specification of next dump and a corresponding index
    i_dump = 0
    next_dump = dumps[i_dump]
    # Possible output at the beginning of simulation
    dump(components, output_filenames, final_render)
    # Before beginning time stepping,
    # communicate pseudo and ghost points on all fluid grids
    # and nullify all fluid buffers of every component.
    # For particle components, these are no-op's.
    for component in components:
        component.communicate_fluid_grids()
        component.nullify_fluid_buffers()
    # The main time loop
    masterprint('Beginning of main time loop')
    timestep = -1
    while i_dump < len(dumps):
        timestep += 1
        # Print out message at beginning of each time step
        masterprint(terminal.bold('\nTime step {}'.format(timestep))
                    + ('{:<' + ('14' if enable_Hubble else '13') + '} {} {}')
                      .format('\nCosmic time:',
                              significant_figures(universals.t, 4, fmt='Unicode'),
                              unit_time)
                    + ('{:<14} {}'.format('\nScale factor:',
                                          significant_figures(universals.a, 4, fmt='Unicode'))
                       if enable_Hubble else '')
                    )
        # Analyze and print out debugging information, if required
        debug(components)
        # Kick
        # (even though 'whole' is used, the first kick (and the first
        # kick after a dump) is really only half a step (the first
        # half), as ᔑdt_steps[integrand][1] == 0 for every integrand).
        scalefactor_integrals('first half')
        kick(components, 'whole')
        if dump(components, output_filenames, final_render, 'drift'):
            # Reset the ᔑdt_steps, starting the leapfrog cycle anew
            nullify_ᔑdt_steps()
            continue
        # Update Δt every Δt_update_freq time step
        if enable_Hubble and not (timestep % Δt_update_freq):
            # Let the positions catch up to the momenta
            drift(components, 'first half')
            # Update the time step size
            # (increase it according to Δt_factor, but not above what
            # any of the components allow for).
            Δt = reduce_Δt(components, Δt_factor*universals.t, give_notice=False)
            # Reset the ᔑdt_steps, starting the leapfrog cycle anew
            nullify_ᔑdt_steps()
            continue
        # Drift
        scalefactor_integrals('second half')
        drift(components, 'whole')
        if dump(components, output_filenames, final_render, 'kick'):
            # Reset the ᔑdt_steps, starting the leapfrog cycle anew
            nullify_ᔑdt_steps()
            continue
        # Reduce time step size if it is too large for any component
        Δt = reduce_Δt(components, Δt)
    # All dumps completed; end of time loop
    masterprint('\nEnd of main time loop'
                + ('{:<' + ('14' if enable_Hubble else '13') + '} {} {}')
                   .format('\nCosmic time:',
                           significant_figures(universals.t, 4, fmt='Unicode'),
                           unit_time)
                + ('{:<14} {}'.format('\nScale factor:',
                                      significant_figures(universals.a, 4, fmt='Unicode'))
                   if enable_Hubble else '')
                )

@cython.header(# Arguments
               components='list',
               Δt='double',
               give_notice='bint',
               # Locals
               cell_size='double',
               component='Component',
               dim='int',
               fac_reduce='double',
               fac_stability='double',
               fastest_component='Component',
               ϱ='double[:, :, :]',
               ϱux='double[:, :, :]',
               ϱuy='double[:, :, :]',
               ϱuz='double[:, :, :]',
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               u_max='double',
               ϱux_ijk='double',
               ϱuy_ijk='double',
               ϱuz_ijk='double',
               u2_ijk='double',
               u2_max='double',
               ẋ_max='double',
               Δt_max='double',
               Δt_max_component='double',
               ϱ_ijk='double',
               )
def reduce_Δt(components, Δt, give_notice=True):
    # Safety factor. A value of 1 allows for a maximum velocity
    # corresponding to one grid cell per time step. To be on the safe
    # side, this value should be smaller than 1.
    fac_stability = 0.9
    # When the current time step size Δt is too large, it will be
    # reduced to the maximally allowed time step times this factor.
    fac_reduce = 0.9
    # Find the maximum velocity of fluid components
    # and compute the maximum allowed time step size Δt_max
    # based on this maximum velocity.
    Δt_max = inf
    fastest_component = None
    for component in components:
        if component.representation != 'fluid':
            continue
        # Find maximum, local velocity for this component
        u2_max = 0
        ϱ   = component.fluidvars['ϱ'].grid_noghosts
        ϱux = component.fluidvars['ϱux'].grid_noghosts
        ϱuy = component.fluidvars['ϱux'].grid_noghosts
        ϱuz = component.fluidvars['ϱux'].grid_noghosts
        size_i = ϱux.shape[0] - 1
        size_j = ϱux.shape[1] - 1
        size_k = ϱux.shape[2] - 1
        for i in range(size_i):
            for j in range(size_j):
                for k in range(size_k):
                    ϱ_ijk = ϱ[i, j, k]
                    ϱux_ijk = ϱux[i, j, k]
                    ϱuy_ijk = ϱuy[i, j, k]
                    ϱuz_ijk = ϱuz[i, j, k]
                    u2_ijk = (ϱux_ijk**2 + ϱuy_ijk**2 + ϱuz_ijk**2)/ϱ_ijk**2
                    if u2_ijk > u2_max:
                        u2_max = u2_ijk
        u_max = sqrt(u2_max)
        # Communicate maximum global velocity of this component
        # to all processes.
        u_max = allreduce(u_max, op=MPI.MAX)
        # In the odd case of a completely static fluid,
        # set u_max to be just above 0.
        if u_max == 0:
            u_max = machine_ϵ 
        # Compute maximum allowed timestep size Δt for this component.
        # Imortantly, because Δt is an interval of cosmic time t,
        # free of any scaling by the scale factor a, the cosmic time
        # derivative of the comoving coordinates, ẋ = u/a,
        # should be used in stead of the peculiar velocity u directly.
        # The sqrt(3) is because the simulation is in 3D. With sqrt(3)
        # included and fac_stability == 1, the below is the general
        # 3-dimensional Courant condition.
        cell_size = boxsize/component.gridsize
        ẋ_max = u_max/universals.a
        Δt_max_component = ℝ[fac_stability/sqrt(3)]*cell_size/ẋ_max
        # The component with the lowest value of the maximally allowed
        # time step size determines the global maximally allowed
        # time step size.
        if Δt_max_component < Δt_max:
            Δt_max = Δt_max_component
            fastest_component = component
    # Adjust the current time step size Δt if it greater than the
    # largest allowed value Δt_max.
    if Δt > Δt_max:
        if give_notice:
            masterwarn('Rescaling time step size by a factor {:.1g} due to large velocities of {}'
                        .format(Δt_max/Δt, fastest_component.name))
        Δt = fac_reduce*Δt_max
    return Δt

# Function which checks the sanity of the user supplied output times,
# creates output directories and defines the output filename patterns.
# A Python function is used because it contains a closure
# (a lambda function).
def prepare_output_times():
    """As this function uses universals.t and universals.a as the
    initial values of the cosmic time and the scale factor, you must
    initialize these properly before calling this function.
    """
    global dumps
    # Check that the output times are legal
    if master:
        for time_param, at_begin in zip(('a', 't'), (universals.a, universals.t)):
            for output_kind, output_time in output_times[time_param].items():
                if output_time and np.min(output_time) < at_begin:
                    msg = ('Cannot produce a {} at {} = {:.6g}{}, '
                           'as the simulation starts at {} = {:.6g}{}.'
                           ).format(output_kind, time_param, np.min(output_time),
                                    (' ' + unit_time) if time_param == 't' else '',
                                    time_param, at_begin,
                                    (' ' + unit_time) if time_param == 't' else '')
                    abort(msg)
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
        for output_kind, output_time in output_times[time_param].items():
            # This kind of output does not matter if
            # it should never be dumped to the disk.
            if not output_time or not output_kind in output_dirs:
                continue
            # Compute number of digits
            times = sorted(set((at_begin, ) + output_time))
            ndigits = 0
            while True:
                fmt = '{{:.{}f}}'.format(ndigits)
                if (len(set([fmt.format(ot) for ot in times])) == len(times)
                    and (fmt.format(times[0]) != fmt.format(0) or not times[0])):
                    break
                ndigits += 1
            fmt = '{{}}={}'.format(fmt)
            # Use the format (that is, either the format from the a
            # output times or the t output times) with the largest
            # number of digits.
            if output_kind in output_filenames:
                if int(re.search('[0-9]+',
                                 re.search('{.+?}',
                                           output_filenames[output_kind])
                                 .group()).group()) >= ndigits:
                    continue
            # Store output name patterns
            output_dir = output_dirs[output_kind]
            output_base = output_bases[output_kind]
            output_filenames[output_kind] = ('{}/{}{}'.format(output_dir,
                                                              output_base,
                                                              '_' if output_base else '')
                                             + fmt)
    # Lists of sorted dump times of both kinds
    a_dumps = sorted(set([nr for val in output_times['a'].values() for nr in val]))
    t_dumps = sorted(set([nr for val in output_times['t'].values() for nr in val]))
    # Both lists combined into one list of lists, the first ([1])
    # element of which are the cosmic time in both cases.
    dumps = [['a', -1, a] for a in a_dumps]
    a_lower = t_lower = machine_ϵ
    for i, d in enumerate(dumps):
        d[1] = cosmic_time(d[2], a_lower, t_lower)
        a_lower, t_lower = d[2], d[1]
    dumps += [['t', t] for t in t_dumps]
    # Sort the list according to the cosmic time
    dumps = sorted(dumps, key=(lambda d: d[1]))
    # It is possible for an a-time to have the same cosmic time value
    # as a t-time. This case should count as only a single dump time.
    for i, d in enumerate(dumps):
        if i + 1 < len(dumps) and d[1] == dumps[i + 1][1]:
            # Remove the t-time, leaving the a-time
            dumps.pop(i + 1)
    # Determine the final render time (scalefactor or cosmic time).
    # Place the result in a tuple (eg. ('a', 1) or ('t', 13.7)).
    final_render = ()
    if render_times['t']:
        final_render_t = render_times['t'][len(render_times['t']) - 1]
        final_render = ('t', final_render_t)
    if render_times['a']:
        final_render_a = render_times['a'][len(render_times['a']) - 1]
        final_render_t = cosmic_time(final_render_a)
        if not final_render or (final_render and final_render_t > final_render[1]):
            final_render = ('a', final_render_t)
    return output_filenames, final_render

# If anything special should happen, rather than starting the timeloop
if special_params:
    delegate()
    Barrier()
    sys.exit()

# Declare global variables used in above functions
cython.declare(ᔑdt_steps='dict',
               i_dump='Py_ssize_t',
               dumps='list',
               next_dump='list',
               Δt='double',
               )
# Run the time loop
timeloop()
# Simulation done
if universals.any_warnings:
    masterprint('\nCO𝘕CEPT run finished')
else:
    masterprint(terminal.bold_green('\nCO𝘕CEPT run finished successfully'))
# Due to an error having to do with the Python -m switch,
# the program must explicitly be told to exit.
Barrier()
sys.exit()
