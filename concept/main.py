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
    from graphics import animate, significant_figures
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from analysis cimport powerspectrum
    from species cimport construct, construct_random
    from IO cimport load, save, load_gadget, save_gadget
    from integration cimport expand, cosmic_time, scalefactor_integral
    from graphics cimport animate, significant_figures
    """

# Imports and definitions common to pure Python and Cython
from os.path import basename


@cython.header(# Arguments
               times='tuple',
               item='str',
               at_beginning='bint',
               )
def check_outputtimes(times, item, at_beginning=False):
    if len(times) == 0:
        return
    if ((at_beginning and np.min(times) < a_begin)
        or (not at_beginning and np.min(times) <= a_begin)):
            raise Exception('The first ' + item + ' is set at a = '
                            + str(np.min(times))
                            + ',\nbut the simulation starts at a = '
                            + str(a_begin) + '.')
    if len(times) > len(set(times)):
        masterwarn(item.capitalize() + ' output times are not unique.\n'
                                     + 'Extra values will be ignored.')

@cython.header(# Locals
               a='double',
               a_dump='double',
               a_next='double',
               drift_fac='double[::1]',
               itg_drift0='double',
               itg_drift1='double',
               itg_kick0='double',
               itg_kick1='double',
               kick_drift_index='int',
               kick_fac='double[::1]',
               powerspec_filename='str',
               snapshot_filename='str',
               t='double',
               timer='double',
               timestep='size_t',
               Δt='double',
               )
def timeloop():
    # The number of time steps before Δt is updated
    Δt_update_freq = 10
    # Initial cosmic time t, where a(t) = a_begin
    a = a_begin
    t = cosmic_time(a)
    # Plot the initial configuration
    if len(snapshot_times) > 0:
        animate(particles, 0, a, np.min(snapshot_times))
    # The time step size should be a
    # small fraction of the age of the universe.
    Δt = Δt_factor*t
    # Arrays containing the drift and kick factors ∫_t^(t + Δt/2)dt/a
    # and ∫_t^(t + Δt/2)dt/a**2.
    drift_fac = empty(2, dtype='float64')
    kick_fac = empty(2, dtype='float64')
    # The main time loop (in actuality two nested loops)
    masterprint('Begin main time loop')
    timestep = 1
    timer = time()
    # Loop over all output times
    for a_dump in dump_times:
        #
        masterprint(terminal.bold('\nTime step ' + str(timestep))
                    + '\nScale factor: '
                    + significant_figures(a, 4, just=3)
                    + '\nCosmic time:  '
                    + significant_figures(t/units.Gyr, 4, just=3)
                    + ' Gyr')
        # The filename of the current snapshot
        snapshot_filename = (snapshot_dir + '/' + snapshot_base
                             + '_a=' + '{:.3f}'.format(a_dump))
        # The filename of the current power spectrum
        powerspec_filename = (powerspec_dir + '/' + powerspec_base
                             + '_a=' + '{:.3f}'.format(a_dump))
        # Do the kick and drift integrals
        # ∫_t^(t + Δt/2)dt/a and ∫_t^(t + Δt/2)dt/a**2.
        a = expand(a, t, 0.5*Δt)
        if a > a_dump:
            raise Exception('Finished time integration within a single step!')
        t += 0.5*Δt
        # This variable flips between 0 and 1, telling whether
        # a kick or a drift should be performed, respectively.
        kick_drift_index = 0
        # Do the kick and drift integrals
        # ∫_t^(t + Δt/2)dt/a and ∫_t^(t + Δt/2)dt/a**2.
        kick_fac[kick_drift_index] = scalefactor_integral(-1)
        drift_fac[kick_drift_index] = scalefactor_integral(-2)
        # The first, half kick
        particles.kick(kick_fac[kick_drift_index])
        # Leapfrog until a == a_dump
        while a < a_dump:
            #
            if kick_drift_index:
                masterprint(terminal.bold('\nTime step ' + str(timestep))
                            + '\nScale factor: '
                            + significant_figures(a, 4, just=3)
                            + '\nCosmic time:  '
                            + significant_figures(t/units.Gyr, 4, just=3)
                            + ' Gyr')
            # Flip the state of kick_drift_index
            kick_drift_index = 0 if kick_drift_index == 1 else 1
            #

            # Update the scale factor and the cosmic time. This also
            # tabulates a(t), needed for the kick and drift integrals.
            a_next = expand(a, t, 0.5*Δt)
            t += 0.5*Δt
            if a_next >= a_dump:
                # Final step reached. A smaller time step than
                # Δt/2 is needed to hit a_dump exactly.
                t -= 0.5*Δt
                t_end = cosmic_time(a_dump, a, t, t + 0.5*Δt)
                expand(a, t, t_end - t)
                a_next = a_dump
                t = t_end
            a = a_next
            # Do the kick and drift integrals
            # ∫_t^(t + Δt/2)dt/a and ∫_t^(t + Δt/2)dt/a**2.
            kick_fac[kick_drift_index] = scalefactor_integral(-1)
            drift_fac[kick_drift_index] = scalefactor_integral(-2)
            # Perform drift or kick
            if kick_drift_index:
                # Drift a complete step, overtaking the kicks
                particles.drift(drift_fac[0] + drift_fac[1])
            else:
                # Kick a complete step, overtaking the drifts
                particles.kick(kick_fac[0] + kick_fac[1])
            # Dump output
            if a == a_dump:
                # Synchronize positions and momenta before dumping
                if kick_drift_index:
                    particles.kick(kick_fac[kick_drift_index])
                else:
                    particles.drift(drift_fac[kick_drift_index])
                # Dump snapshot
                if a in snapshot_times:
                    save(particles, a, snapshot_filename)
                # Dump powerspectrum
                if a in powerspec_times:
                    powerspectrum(particles, powerspec_filename)
            # After every second iteration (every whole time step):
            if kick_drift_index:
                # Render particle configuration
                # and print timestep message.
                animate(particles, timestep, a, a_dump)
                # Refresh timer and update the time step nr
                timer = time()
                timestep += 1
                # Update Δt every Δt_update_freq time step
                if not (timestep % Δt_update_freq):
                    Δt_prev = Δt
                    Δt = Δt_factor*t
                    # Due to the new (and increased) Δt, the drifting is
                    # no longer Δt/2 ahead of the kicking.
                    # Drift the missing distance.
                    a_next = expand(a, t, 0.5*(Δt - Δt_prev))
                    if a_next < a_dump:
                        # Only drift if a_dump is not reached by it
                        masterprint('Updating time step size ... ', end='')
                        a = a_next
                        # Do the kick and drift integrals
                        # ∫_t^(t + Δt/2)dt/a and ∫_t^(t + Δt/2)dt/a**2
                        # and drift the remaining distance.
                        kick_fac[kick_drift_index] += scalefactor_integral(-1)
                        particles.drift(scalefactor_integral(-2))
                        masterprint('done')
                    else:
                        # Do not alter Δt just before (or just after,
                        # in the case of a == a_dump) dumps.
                        Δt = Δt_prev
            # Always render particle configuration when at snapshot time
            elif a == a_dump and a in snapshot_times:
                animate(particles, timestep, a, a_dump)

# If anything special should happen, rather than starting the timeloop
cython.declare(particles='Particles')
if ((cython.compiled and special is not None)
    or (not cython.compiled and 'special' in locals())):
    particles = load(IC_file, write_msg=False)
    if special == 'powerspectrum':
        powerspectrum(particles, powerspec_dir + '/' + powerspec_base + '_'
                                 + basename(IC_file))
    Barrier()
    sys.exit()

# Check that the snapshot and powerspectrum times are legal
check_outputtimes(snapshot_times, 'snapshot')
check_outputtimes(powerspec_times, 'power spectrum', at_beginning=True)
# Create list of dump times
cython.declare(dump_times='list')
dump_times = sorted(set(snapshot_times + powerspec_times))
# Power spectrum of the IC?
if len(dump_times) > 0 and dump_times[0] == a_begin:
    dump_times = dump_times[1:]
    if np.min(powerspec_times) == a_begin:
        powerspec_filename = (powerspec_dir + '/' + powerspec_base
                              + '_a=' + '{:.3f}'.format(a_begin))
        powerspectrum(particles, powerspec_filename)
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
