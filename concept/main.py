# Copyright (C) 2015 Jeppe Mosgard Dakin
#
# This file is part of CONCEPT, the cosmological N-body code in Python
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.



# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from analysis import powerspectrum
    from species import construct, construct_random
    from IO import save, load
    from integration import expand, cosmic_time, scalefactor_integral
    from graphics import render, significant_figures
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from analysis cimport powerspectrum
    from species cimport construct, construct_random
    from IO cimport load, save, load_gadget, save_gadget
    from integration cimport expand, cosmic_time, scalefactor_integral
    from graphics cimport render, significant_figures
    """

# Imports and definitions common to pure Python and Cython
from os.path import basename

# Function that computes the kick and drift factors (integrals).
# The result is stored in drift_fac[index] and kick_fac[index],
# where index is the argument and can be either 0 or 1. 
@cython.header(# Arguments
               index='int',
               # Locals
               a_end='double',
               t_end='double',
               )
def do_kick_drift_integrals(index):
    global a, a_dump, drift_fac, kick_fac, t, Δt
    # Update the scale factor and the cosmic time. This also
    # tabulates a(t), needed for the kick and drift integrals.
    a_end = expand(a, t, 0.5*Δt)
    t += 0.5*Δt
    if a_end >= a_dump:
        # Dump time reached. A smaller time step than
        # Δt/2 is needed to hit a_dump exactly.
        t -= 0.5*Δt
        t_end = cosmic_time(a_dump, a, t, t + 0.5*Δt)
        expand(a, t, t_end - t)
        a_end = a_dump
        t = t_end
    a = a_end
    # Do the kick and drift integrals
    # ∫_t^(t + Δt/2)dt/a and ∫_t^(t + Δt/2)dt/a**2.
    kick_fac[index]  = scalefactor_integral(-1)
    drift_fac[index] = scalefactor_integral(-2)

# Function which dump all types of output. The return value signifies
# whether or not something has been dumped.
@cython.header(# Arguments
               op='str',
               # Locals
               powerspec_filename='str',
               snapshot_filename='str',
               render_filename='str',
               returns='bint',
               )
def dump(op):
    global a, a_dump, drift_fac, i_dump, kick_fac
    # Do nothing if not at dump time
    if a != a_dump:
        return False
    # Synchronize positions and momenta before dumping
    if op == 'drift':
        particles.drift(drift_fac[0])
    elif op == 'kick':
        particles.kick(kick_fac[1])
    # Dump snapshot
    if a in snapshot_times:
        snapshot_filename = (snapshot_dir + '/' + snapshot_base
                             + '_a={:.3f}'.format(a))
        save(particles, a, snapshot_filename)
    # Dump powerspectrum
    if a in powerspec_times:
        powerspec_filename = (powerspec_dir + '/' + powerspec_base
                              + '_a={:.3f}'.format(a))
        powerspectrum(particles, powerspec_filename)
    # Dump render
    if a in render_times:
        render_filename = (render_dir + '/' + render_base
                           + '_a={:.3f}.png'.format(a))
        render(particles, a, render_filename)
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
               Δt_update_freq='size_t',
               )
def timeloop():
    global a, a_dump, drift_fac, i_dump, kick_fac, t, Δt
    # Do nothing if no dump times exist
    if len(a_dumps) == 0:
        return
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
        if dump('drift'):
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
        if dump('kick'):
            continue

# Declare global variables used in above functions
cython.declare(a='double',
               a_dump='double',
               i_dump='size_t',
               drift_fac='double[::1]',
               kick_fac='double[::1]',
               t='double',
               Δt='double',
               )

# Check that the output times are legal
for times, kind in zip((snapshot_times, powerspec_times, render_times),
                       ('snapshot', 'powerspectrum', 'render')):
    if not times:
        continue
    if len(times) > len(set(times)):
        masterwarn(kind.capitalize() + ' output times are not unique. '
                                     + 'Extra values will be ignored.')
    if np.min(times) < a_begin and master:
        raise Exception('Cannot produce a ' + kind + ' at time a = '
                         + str(np.min(times)) + ', as the simulation '
                         + 'starts at a = ' + str(a_begin) + '.')

# If the output directories do not exist, create them
if master:
    for times, output_dir in zip((snapshot_times,
                                  powerspec_times,
                                  render_times),
                                 (snapshot_dir,
                                  powerspec_dir,
                                  render_dir)):
        if len(times) > 0:
            os.makedirs(output_dir, exist_ok=True)

# If anything special should happen, rather than starting the timeloop
cython.declare(particles='Particles')
if special:
    # Load initial conditions
    particles = load(IC_file, write_msg=False)
    if special == 'powerspectrum':
        # Load initial conditions
        powerspectrum(particles, powerspec_dir + '/' + powerspec_base + '_'
                                 + basename(IC_file))
    else:
        masterwarn('Special flag "' + special + '" not recognized!')
    Barrier()
    sys.exit()

# Create list of dump times
cython.declare(a_dumps='list')
a_dumps = sorted(set(snapshot_times + powerspec_times + render_times))
# Load initial conditions
particles = load(IC_file)
# Run the time loop
timeloop()
# Simulation done
masterprint(terminal.bold_green(terminal.CONCEPT + ' ran successfully'))
# Due to an error having to do with the Python -m switch,
# the program must explicitly be told to exit.
Barrier()
sys.exit()
