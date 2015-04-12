# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from species import construct, construct_random
    from IO import save, load
    from integration import expand, cosmic_time, scalefactor_integral
    from graphics import animate, timestep_message
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from species cimport construct, construct_random
    from IO cimport load, save, load_gadget, save_gadget
    from integration cimport expand, cosmic_time, scalefactor_integral, ȧ
    from graphics cimport animate, timestep_message
    """

# Exit the program if called with the --exit option
if int(sys.argv[2]):
    if master:
        os.system('printf "\033[1m\033[92mCO\033[3mN\033[0m\033[1m\033[92mCEPT'
                  + ' ran successfully\033[0m\n"')
    sys.exit(0)
# Load initial conditions
cython.declare(particles='Particles',
               a_max='double',
               )
particles = load(IC_file)
# Check that the values in outputtimes are legal
if np.min(outputtimes) <= a_begin:
    raise Exception('The first snapshot is set at a = '
                    + str(min(outputtimes))
                    + ',\nbut the simulation starts at a = '
                    + str(a_begin) + '.')
if len(outputtimes) > len(set(outputtimes)):
    warn('Values in outputtimes are not unique.\n'
         + 'Extra values will be ignored.')
a_max = np.max(outputtimes)


"""
particles = construct_random('hm', 'dark matter', 256)
posx = []
posy = []
posz = []
if rank == nprocs - 1:
    for i in range(2, 4):
        for j in range(4):
            for k in range(4):
                posx.append((i + 0.5)/4*boxsize)
                posy.append((j + 0.5)/4*boxsize)
                posz.append((k + 0.5)/4*boxsize)
posx = array(posx)
posy = array(posy)
posz = array(posz)
momx = zeros(posx.size)
momy = zeros(posx.size)
momz = zeros(posx.size)
particles.populate(posx, 'posx')
particles.populate(posy, 'posy')
particles.populate(posz, 'posz')
particles.populate(momx, 'momx')
particles.populate(momy, 'momy')
particles.populate(momz, 'momz')
"""

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Locals
               a='double',
               a_next='double',
               a_snapshot='double',
               drift_fac='double[::1]',
               i_snapshot='int',
               itg_drift0='double',
               itg_drift1='double',
               itg_kick0='double',
               itg_kick1='double',
               kick_drift_index='int',
               kick_fac='double[::1]',
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
    animate(particles, 0, a, min(outputtimes))
    # The time step size should be a small fraction of the age of the universe
    Δt = Δt_factor*t
    # Arrays containing the drift and kick factors ∫_t^(t + Δt/2)dt/a
    # and ∫_t^(t + Δt/2)dt/a**2.
    drift_fac = empty(2)
    kick_fac = empty(2)
    # The main time loop (in actuality two nested loops)
    if master:
        print('Begin main time loop')
    timestep = 0
    timer = time()
    # Loop over all output snapshots
    for i_snapshot, a_snapshot in enumerate(sorted(set(outputtimes))):
        # The filename of the current snapshot
        snapshot_filename = (output_dir + '/' + snapshot_base
                             + '_' + str(i_snapshot))
        # Do the kick and drift intetrals
        # ∫_t^(t + Δt/2)dt/a and ∫_t^(t + Δt/2)dt/a**2.
        a = expand(a, t, 0.5*Δt)
        if a > a_snapshot:
            raise Exception('Finished time integration within a single step!')
        t += 0.5*Δt
        # This variable flip between 0 and 1, telling whether a kick or a drift
        # should be performed, respectively.
        kick_drift_index = 0
        # Do the kick and drift integrals
        # ∫_t^(t + Δt/2)dt/a and ∫_t^(t + Δt/2)dt/a**2.
        kick_fac[kick_drift_index] = scalefactor_integral(-1)
        drift_fac[kick_drift_index] = scalefactor_integral(-2)
        # The first, half kick
        particles.kick(kick_fac[kick_drift_index])
        # Leapfrog until a == a_snapshot
        while a < a_snapshot:
            # Flip the state of kick_drift_index
            kick_drift_index = 0 if kick_drift_index == 1 else 1
            # Update the scale factor and the cosmic time. This also tabulates
            # a(t), needed for the kick and drift integrals.
            a_next = expand(a, t, 0.5*Δt)
            t += 0.5*Δt
            if a_next >= a_snapshot:
                # Final step reached. A smaller time step than
                # Δt/2 is needed to hit a_snapshot exactly.
                t -= 0.5*Δt
                t_end = cosmic_time(a_snapshot, a, t, t + 0.5*Δt)
                expand(a, t, t_end - t)
                a_next = a_snapshot
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
            # Dump snapshot if a == a_snapshot
            if a == a_snapshot:
                # Synchronize positions and momenta before dumping snapshot
                if kick_drift_index:
                    particles.kick(kick_fac[kick_drift_index])
                else:
                    particles.drift(drift_fac[kick_drift_index])
                # Dump snapshot
                save(particles, a, snapshot_filename)
            # After every second iteration (every whole time step):
            if kick_drift_index:
                # Render particle configuration and print timestep message
                animate(particles, timestep, a, a_snapshot)
                timestep_message(timestep, timer, a, t)
                # Refresh timer and update the time step nr
                timer = time()
                timestep += 1
                # Update Δt every Δt_update_freq time step
                if not (timestep % Δt_update_freq):
                    Δt_prev = Δt
                    Δt = Δt_factor*t
                    # Due to the new (and increased) Δt, the drifting is no
                    # longer Δt/2 ahead of the kicking. Drift the missing
                    # distance.
                    a_next = expand(a, t, 0.5*(Δt - Δt_prev))
                    if a_next < a_snapshot:
                        # Only drift if a_snapshot is not reached by it
                        if master:
                            print('Updating time step size')
                        a = a_next
                        # Do the kick and drift integrals
                        # ∫_t^(t + Δt/2)dt/a and ∫_t^(t + Δt/2)dt/a**2 and
                        # drift the remaining distance.
                        kick_fac[kick_drift_index] += scalefactor_integral(-1)
                        particles.drift(scalefactor_integral(-2))
                    else:
                        # Do not alter Δt just before (or just after, in the
                        # case of a == a_snapshot) snapshot dump.
                        Δt = Δt_prev
            # Always render particle configuration when at snapshot time
            elif a == a_snapshot:
                animate(particles, timestep, a, a_snapshot)
                if a == a_max:
                    timestep_message(timestep, timer, a, t)



# Run the time loop at import time
timeloop()
