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
from snapshot import load
cimport('from analysis import powerspec')
cimport('from graphics import render, terminal_render')
cimport('from gravity import PM, build_φ, fluid_φkick')
cimport('from mesh import diff')
cimport('from integration import expand, cosmic_time, scalefactor_integral')
cimport('from utilities import delegate')
cimport('from snapshot import save')



# Function that computes the kick and drift factors (integrals).
# The result is stored in a_integrals[integrand][index],
# where index is the argument and can be either 0 or 1. 
@cython.header(# Arguments
               index='int',
               # Locals
               a_next='double',
               t_next='double',
               )
def do_kick_drift_integrals(index):
    global a, a_integrals, a_dump, t, Δt
    # Update the scale factor and the cosmic time. This also
    # tabulates a(t), needed for the scalefactor integrals.
    a_next = expand(a, t, 0.5*Δt)
    t_next = t + 0.5*Δt
    if a_next >= a_dump:
        # Dump time reached. A smaller time step than
        # 0.5*Δt is needed to hit a_dump exactly.
        a_next = a_dump
        t_next = cosmic_time(a_dump, a, t, t_next)
        expand(a, t, t_next - t)
    a = a_next
    t = t_next
    # Do the scalefactor integrals
    # ∫_t^(t + Δt/2)a⁻¹dt,
    # ∫_t^(t + Δt/2)a⁻²dt.
    a_integrals['a⁻¹'][index] = scalefactor_integral(-1)
    a_integrals['a⁻²'][index] = scalefactor_integral(-2)

# Function which dump all types of output. The return value signifies
# whether or not something has been dumped.
@cython.header(# Arguments
               components='list',
               output_filenames='dict',
               op='str',
               # Locals
               index='int',
               integrand='str',
               returns='bint',
               )
def dump(components, output_filenames, op=None):
    global a, a_dump, i_dump
    # Do nothing if not at dump time
    if a != a_dump:
        return False
    # Synchronize positions and momenta before dumping
    if op == 'drift':
        drift(components, 'first half')
    elif op == 'kick':
        kick(components, 'second half')
    # Dump terminal render
    if a in terminal_render_times:
        terminal_render(components)
    # Dump snapshot
    if a in snapshot_times:
        save(components, a, output_filenames['snapshot'].format(a))
    # Dump powerspectrum
    if a in powerspec_times:
        powerspec(components, a, output_filenames['powerspec'].format(a))
    # Dump render
    if a in render_times:
        render(components, a, output_filenames['render'].format(a),
               cleanup=(a == render_times[len(render_times) - 1]))
    # Increment dump time
    i_dump += 1
    if i_dump < len(a_dumps):
        a_dump = a_dumps[i_dump]
    # Reset (nullify) the a_integrals, making the next kick operation
    # apply for only half a step, even though 'whole' is used.
    for integrand in a_integrals:
        for index in range(2):
            a_integrals[integrand][index] = 0
    return True

# Function which kick all of the components
@cython.header(# Arguments
               components='list',
               step='str',
               # Locals
               a_integrals_step='dict',
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
    # Construct the local dict a_integrals_step,
    # based on which type of step is to be performed.
    a_integrals_step = {}
    for integrand in a_integrals:
        if step == 'first half':
            a_integrals_step[integrand] = a_integrals[integrand][0]
        elif step == 'second half':
            a_integrals_step[integrand] = a_integrals[integrand][1]
        elif step == 'whole':
            a_integrals_step[integrand] = np.sum(a_integrals[integrand])
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
            masterprint('Kicking (PM) {} ...'.format(', '.join(
                        [component.name for component in component_groups['PM']])))
        elif 'PM' not in component_groups and 'fluid' in component_groups:
            masterprint('Kicking (potential only) {} ...'.format(', '.join(
                        [component.name for component in component_groups['fluid']])))
        else:
            masterprint('Kicking (PM) {} and (potential only) {} ...'.format(', '.join(
                          [component.name for component in component_groups['PM']]
                        + [component.name for component in component_groups['fluid']]))) 
        # For each dimension, differentiate φ and apply the force to
        # all components which interact with φ (particles using the PM
        # method and all fluids).
        h = boxsize/φ_gridsize  # Physical grid spacing of φ
        for dim in range(3):
            # Do the differentiation of φ
            meshbuf_mv = diff(φ, dim, h)
            # Apply PM kick
            for component in component_groups['PM']:
                PM(component, a_integrals_step['a⁻¹'], meshbuf_mv, dim)
            # Apply the potential part of the kick to fluids
            for component in component_groups['fluid']:
                fluid_φkick(component, a_integrals_step['a⁻²'], meshbuf_mv, dim)
        # Done with potential interactions
        masterprint('done')
    # Now kick all other components sequentially
    for key, component_group in component_groups.items():
        if key in ('PM', 'fluid'):
            continue
        for component in component_group:
            component.kick(a_integrals_step['a⁻¹'])

# Function which drift all of the components
@cython.header(# Arguments
               components='list',
               step='str',
               # Locals
               a_integrals_step='dict',
               component='Component',
               )
def drift(components, step):
    # Construct the local dict a_integrals_step,
    # based on which type of step is to be performed.
    a_integrals_step = {}
    for integrand in a_integrals:
        if step == 'first half':
            a_integrals_step[integrand] = a_integrals[integrand][0]
        elif step == 'second half':
            a_integrals_step[integrand] = a_integrals[integrand][1]
        elif step == 'whole':
            a_integrals_step[integrand] = np.sum(a_integrals[integrand])
    # Simply drift the components sequentially
    for component in components:
        component.drift(a_integrals_step['a⁻²'])

# Function containing the main time loop of CO𝘕CEPT
@cython.header(# Locals
               component='Component',
               components='list',
               dim='int',
               kick_algorithm='str',
               output_filenames='dict',
               timestep='Py_ssize_t',
               Δt_update_freq='Py_ssize_t',
               )
def timeloop():
    global a, a_dump, a_integrals, i_dump, t, Δt
    # Do nothing if no dump times exist
    if len(a_dumps) == 0:
        return
    # Get the output filename patterns
    output_filenames = prepare_output_times()
    # Load initial conditions
    components = load(IC_file, only_components=True)
    # The number of time steps before Δt is updated
    Δt_update_freq = 10
    # Initial cosmic time t, where a(t) = a_begin
    a = a_begin
    t = cosmic_time(a)
    # The time step size should be a
    # small fraction of the age of the universe.
    Δt = Δt_factor*t
    # Arrays containing the drift and kick factors ∫_t^(t + Δt/2)dt/a
    # and ∫_t^(t + Δt/2)dt/a**2. The two elements in each variable are
    # the first and second half of the factor for the entire time step.
    a_integrals = {'a⁻¹': zeros(2, dtype=C2np['double']),
                   'a⁻²': zeros(2, dtype=C2np['double']),
                   }
    # Scalefactor at next dump and a corresponding index
    i_dump = 0
    a_dump = a_dumps[i_dump]
    # Possible output at a == a_begin
    dump(components, output_filenames)
    # The main time loop
    masterprint('Begin main time loop')
    timestep = -1
    while i_dump < len(a_dumps):
        timestep += 1
        # Print out message at beginning of each time step
        masterprint(terminal.bold('\nTime step {}'.format(timestep))
                    + '{:<14} {}'    .format('\nScale factor:',
                                             significant_figures(a, 4, fmt='Unicode'))
                    + '{:<14} {} Gyr'.format('\nCosmic time:',
                                             significant_figures(t/units.Gyr, 4, fmt='Unicode'))
                    )
        # Kick
        # (even though 'whole' is used, the first kick (and the first
        # kick after a dump) is really only half a step (the first
        # half), as a_integrals[integrand][1] == 0 for every integrand).
        do_kick_drift_integrals(0)
        kick(components, 'whole')
        if dump(components, output_filenames, 'drift'):
            continue
        # Update Δt every Δt_update_freq time step
        if not (timestep % Δt_update_freq):
            # Let the positions catch up to the momenta
            drift(components, 'first half')
            Δt = Δt_factor*t
            # Reset the second kick factor,
            # making the next operation a half kick.
            a_integrals['a⁻¹'][1] = 0
            continue
        # Drift
        do_kick_drift_integrals(1)
        drift(components, 'whole')
        if dump(components, output_filenames, 'kick'):
            continue

# Function which checks the sanity of the user supplied output times,
# creates output directories and defines the output filename patterns.
@cython.header(# Locals
               fmt='str',
               msg='str',
               ndigits='int',
               ot='double',
               output_base='str',
               output_dir='str',
               output_kind='str',
               output_filenames='dict',
               output_time='tuple',
               times='list',
               returns='dict',
               )
def prepare_output_times():
    # Check that the output times are legal
    if master:
        for output_kind, output_time in output_times.items():
            if output_time and np.min(output_time) < a_begin:
                msg = ('Cannot produce a {} at time a = {}, as the simulation starts at a = {}.'
                       ).format(output_kind, np.min(output_time), a_begin)
                abort(msg)
    # Create output directories if necessary
    if master:
        for output_kind, output_time in output_times.items():
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
    # filenames. There should be enogh digits so that adjacent dumps do
    # not overwrite each other, and so that the name of the first dump
    # differs from that of the IC, should it use the same
    # naming convention.
    output_filenames = {}
    for output_kind, output_time in output_times.items():
        # This kind of output does not matter if
        # it should never be dumped to the disk.
        if not output_time or not output_kind in output_dirs:
            continue
        # Compute number of digits 
        times = sorted(set((a_begin, ) + output_time))
        ndigits = 0
        while True:
            fmt = '{:.' + str(ndigits) + 'f}'
            if (len(set([fmt.format(ot) for ot in times])) == len(times)
                and (fmt.format(times[0]) != fmt.format(0) or not times[0])):
                break
            ndigits += 1    
        # Store output name patterns                   
        output_dir = output_dirs[output_kind]
        output_base = output_bases[output_kind]
        output_filenames[output_kind] = ('{}/{}{}a='.format(output_dir,
                                                            output_base,
                                                            '_' if output_base else '')
                                         + fmt)
    return output_filenames

# If anything special should happen, rather than starting the timeloop
if special_params:
    delegate()
    Barrier()
    sys.exit()

# Declare global variables used in above functions
cython.declare(a='double',
               a_dump='double',
               a_integrals='dict',
               i_dump='Py_ssize_t',
               t='double',
               Δt='double',
               )
# Run the time loop
timeloop()
# Simulation done
if any_warnings[0]:
    masterprint('\nCO𝘕CEPT run finished')
else:
    masterprint(terminal.bold_green('\nCO𝘕CEPT run finished successfully'))
# Due to an error having to do with the Python -m switch,
# the program must explicitly be told to exit.
Barrier()
sys.exit()

