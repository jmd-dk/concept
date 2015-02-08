# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from species import construct, construct_random
    from IO import load, save
    from integration import expand, cosmic_time, scalefactor_integral
    from graphics import animate, timestep_message
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from species cimport construct, construct_random
    from IO cimport load, save
    from integration cimport expand, cosmic_time, scalefactor_integral
    from graphics cimport animate, timestep_message
    """

# Construct
cython.declare(particles='Particles')
particles = construct_random('some typename', 'dark matter', N=5000)
particles.mass = 3*H0**2/(8*pi*G_Newton)*boxsize**3/particles.N
# Save
#save(particles, 'ICs/test_0vel')
# Load (and thereby order them correctly)
#particles = load('ICs/test_0vel')

cython.declare(a='double',
               a_next='double',
               itg_drift0='double',
               itg_drift1='double',
               itg_kick0='double',
               itg_kick1='double',
               t='double',
               t_next='double',
               t_iter='double',
               timestep='size_t',
               Δt='double',
               )
# Plot the initial configuration and print message
animate(particles, 0, 0)
if master:
    print('Begin main time loop')
# Initial cosmic time t, where a(t) = a_begin
a = a_begin
t = cosmic_time(a)
# DETERMINE THE TIME STEP SIZE SOMEHOW  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Δt = 100*units.Myr
# The first, half leapfrog kick
a = expand(a, t, Δt/2)
if a > a_end:
    raise Exception('Finished time integration within the first step.')
t += Δt/2
itg_kick0 = scalefactor_integral(-1)
itg_drift0 = scalefactor_integral(-2)
particles.kick(itg_kick0)
# Main time loop
timestep = 0
t_iter = time()
while True:
    # Leapfrog drift
    if a == a_end:
        # Last step reached in previous half kick step.
        # Drift the remaining little bit.
        particles.drift(itg_drift0)
        break
    else:
        # Take half a step
        a_next = expand(a, t, Δt/2)
        t += Δt/2
        if a_next >= a_end:
            # Last step reached
            t -= Δt/2
            t_end = cosmic_time(a_end, a, t, t + Δt/2)
            expand(a, t, t_end - t)
            a_next = a_end
            t = t_end
        a = a_next    
        itg_kick1 = scalefactor_integral(-1)
        itg_drift1 = scalefactor_integral(-2)   # -2 !!!!!!!!!!!!!!!!!!!!!!!
        particles.drift(itg_drift0 + itg_drift1)
    # Leapfrog kick
    if a == a_end:
        # Last step reached in previous half drift step.
        # Kick the remaining little bit.
        particles.kick(itg_kick1)
        break
    else:
        # Take half a step
        a_next = expand(a, t, Δt/2)
        t += Δt/2
        if a_next >= a_end:
            # Last step reached
            t -= Δt/2
            t_end = cosmic_time(a_end, a, t, t + Δt/2)
            expand(a, t, t_end - t)
            a_next = a_end
            t = t_end
        a = a_next
        itg_kick0 = scalefactor_integral(-1)
        itg_drift0 = scalefactor_integral(-2)  # -2 !!!!!!!!!!!!!!!!!!!!!
        particles.kick(itg_kick0 + itg_kick1)
    # Animate and print out time step message
    animate(particles, timestep, a)
    timestep_message(timestep, t_iter, a, t)
    # Update iteration timestamp and number
    t_iter = time()
    timestep += 1
# Plot the final configuration and print the final time step message
animate(particles, timestep, a)
timestep_message(timestep, t_iter, a, t)



# p = m*a**2*xdot
# u = a*xdot = p/(m*a)
# p/m * dt/a**2 = u * dt/a

# m*F * dt/a --> F * dt/a   (tror jeg nok)

#save(particles, 'ICs/mom')

import numpy as np
import matplotlib.pyplot as plt
from numpy import ones

# Read in particles computed via momentum
cython.declare(particles_mom='Particles',
               N_local='size_t',
               N_locals='size_t[::1]',
               X_mom='double[::1]',
               X='double[::1]',
               )
particles_mom = load('ICs/mom')
N_local = particles_mom.N_local
N_locals = empty(nprocs if master else 0, dtype='uintp')
Gather(array(N_local, dtype='uintp'), N_locals)
X_mom = (-30000.0)*ones(particles_mom.N if master else 0)
sendbuf = particles_mom.posx_mw[:N_local]
Gatherv(sendbuf=sendbuf, recvbuf=(X_mom, N_locals))


# X for this run
N_local = particles.N_local
N_locals = empty(nprocs if master else 0, dtype='uintp')
Gather(array(N_local, dtype='uintp'), N_locals)
X = (-30000.0)*ones(particles.N if master else 0)
sendbuf = particles.posx_mw[:N_local]
Gatherv(sendbuf=sendbuf, recvbuf=(X, N_locals))
if master:
    plt.close()
    plt.plot((array(X) - array(X_mom))/boxsize, 'b.-')
    plt.show()


######################################################
# NOW MAKE A BACKUP AND TRY TO REPLACE MOM WITH u
######################################################

