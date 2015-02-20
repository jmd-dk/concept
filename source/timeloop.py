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
particles = construct_random('some typename', 'dark matter', N=2000)
particles.mass = ϱ*boxsize**3/particles.N
# Save
#save(particles, 'ICs/test')
# Load
#particles = load('ICs/test')



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
Δt = 10*units.Myr
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
        itg_drift1 = scalefactor_integral(-2)
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
        itg_drift0 = scalefactor_integral(-2)
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

