# This file is part of CONCEPT, the cosmological N-body code in Python.
# Copyright (C) 2015 Jeppe Mosgard Dakin.
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CONCEPT. If not, see http://www.gnu.org/licenses/
#
# The auther of CONCEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CONCEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from analysis import powerspec
    from species import construct, construct_random
    from IO import load, save
    from special import delegate
    from integration import expand, cosmic_time, scalefactor_integral
    from graphics import render, significant_figures, terminal_render
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from analysis cimport powerspec
    from species cimport construct, construct_random
    from special cimport delegate
    from IO cimport load, save
    from integration cimport expand, cosmic_time, scalefactor_integral
    from graphics cimport render, significant_figures, terminal_render
    """

# Imports and definitions common to pure Python and Cython
from os.path import basename

# Function that computes the kick and drift factors (integrals).
# The result is stored in drift_fac[index] and kick_fac[index],
# where index is the argument and can be either 0 or 1. 
@cython.header(# Arguments
               index='int',
               # Locals
               a_next='double',
               t_next='double',
               )
def do_kick_drift_integrals(index):
    global a, a_dump, drift_fac, kick_fac, t, Δt
    # Update the scale factor and the cosmic time. This also
    # tabulates a(t), needed for the kick and drift integrals.
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
    # Do the kick and drift integrals
    # ∫_t^(t + Δt/2)dt/a and ∫_t^(t + Δt/2)dt/a**2.
    kick_fac[index]  = scalefactor_integral(-1)
    drift_fac[index] = scalefactor_integral(-2)

# Function which dump all types of output. The return value signifies
# whether or not something has been dumped.
@cython.header(# Arguments
               particles='Particles',
               op='str',
               # Locals
               returns='bint',
               )
def dump(particles, op=None):
    global a, a_dump, drift_fac, i_dump, kick_fac
    # Do nothing if not at dump time
    if a != a_dump:
        return False
    # Synchronize positions and momenta before dumping
    if op == 'drift':
        particles.drift(drift_fac[0])
    elif op == 'kick':
        particles.kick(kick_fac[1])
    # Dump powerspectrum
    if a in powerspec_times:
        powerspec(particles, output_filenames['powerspec'].format(a))
    # Dump render
    if a in render_times:
        render(particles, a, output_filenames['render'].format(a))
    # Dump snapshot
    if a in snapshot_times:
        save(particles, a, output_filenames['snapshot'].format(a))
    # Dump terminal render
    if a in terminal_render_times:
        terminal_render(particles)
    # Increment dump time
    i_dump += 1
    if i_dump < len(a_dumps):
        a_dump = a_dumps[i_dump]
    # Reset the second kick factor,
    # making the next operation a half kick.
    kick_fac[1] = 0
    return True

# Function containing the main time loop of CONCEPT
@cython.header(# Locals
               timestep='ptrdiff_t',
               particles='Particles',
               Δt_update_freq='Py_ssize_t',
               )
def timeloop():
    global a, a_dump, drift_fac, i_dump, kick_fac, t, Δt
    # Do nothing if no dump times exist
    if len(a_dumps) == 0:
        return
    # Load initial conditions
    particles = load(IC_file)
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
    drift_fac = zeros(2, dtype='float64')
    kick_fac = zeros(2, dtype='float64')
    # Scalefactor at next dump and a corresponding index
    i_dump = 0
    a_dump = a_dumps[i_dump]
    # Possible output at a == a_begin
    dump(particles)
    # The main time loop
    masterprint('Begin main time loop')
    timestep = -1
    while i_dump < len(a_dumps):
        timestep += 1
        # Print out message at beginning of each time step
        masterprint(terminal.bold('\nTime step ' + str(timestep))
                    + '\nScale factor: '
                    + significant_figures(a, 4, just=3)
                    + '\nCosmic time:  '
                    + significant_figures(t/units.Gyr, 4, just=3)
                    + ' Gyr')
        # Kick (first time is only half a kick, as kick_fac[1] == 0)
        do_kick_drift_integrals(0)
        particles.kick(kick_fac[0] + kick_fac[1])
        if dump(particles, 'drift'):
            continue
        # Update Δt every Δt_update_freq time step
        if not (timestep % Δt_update_freq):
            # Let the positions catch up to the momenta
            particles.drift(drift_fac[0])
            Δt = Δt_factor*t
            # Reset the second kick factor,
            # making the next operation a half kick.
            kick_fac[1] = 0
            continue
        # Drift
        do_kick_drift_integrals(1)
        particles.drift(drift_fac[0] + drift_fac[1])
        if dump(particles, 'kick'):
            continue

# Declare global variables used in above functions,
# as well as variables in the code below.
cython.declare(# Globals
               a='double',
               a_dump='double',
               drift_fac='double[::1]',
               i_dump='Py_ssize_t',
               kick_fac='double[::1]',
               t='double',
               Δt='double',
               # Module level variables
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
               )
# Check that the output times are legal
if master:
    for output_kind, output_time in output_times.items():
        if output_time and np.min(output_time) < a_begin:
            msg = ('Cannot produce a {} at time a = {}, '
                   + 'as the simulation starts at a = {}.'
                   ).format(output_kind, np.min(output_time), a_begin)
            raise Exception(msg)
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
# determining the number of scalefactor digits in the output filenames.
# There should be enogh digits so that adjacent dumps do not overwrite
# each other, and so that the name of the first dump differs from that
# of the IC, should it use the same naming convention.
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
        if (len(set(fmt.format(ot) for ot in times)) == len(times)
            and (fmt.format(times[0]) != fmt.format(0) or not times[0])):
            break
        ndigits += 1    
    # Store output name patterns                   
    output_dir = output_dirs[output_kind]
    output_base = output_bases[output_kind]
    output_filenames[output_kind] = ('{}/{}{}a='.format(output_dir,
                                                        output_base,
                                                        ('_' if output_base
                                                             else ''))
                                     + fmt)
# If anything special should happen, rather than starting the timeloop
if special_params:
    delegate()
    Barrier()
    sys.exit()

# Run the time loop
timeloop()
# Simulation done
masterprint(terminal.bold_green(terminal.CONCEPT + ' run finished'))
# Due to an error having to do with the Python -m switch,
# the program must explicitly be told to exit.
Barrier()
sys.exit()
