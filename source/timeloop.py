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
particles = construct_random('some typename', 'dark matter', N=500)
particles.mass = 3*H0**2/(8*pi*G_Newton)*boxsize**3/particles.N
# Save
save(particles, 'ICs/test')
# Load (and thereby order them correctly)
particles = load('ICs/test')


cython.declare(a='double',
               a_before='double',
               a_next='double',
               leap='double',
               t='double',
               t_iter='double',
               t_before='double',
               timestep='size_t',
               Δt='double',
               )
# Plot the initial configuration
animate(particles, 0, 0)
# Compute initial cosmic time t, where a(t) = a_begin
a = a_begin
t = cosmic_time(a)
# DETERMINE THE TIME STEP SIZE SOMEHOW  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Δt = 100*units.Myr
# First leapfrog kick
a_next = expand(a, t, Δt/2)
leap = Δt/2
particles.kick(scalefactor_integral(-1))
# Main time loop
timestep = 0
t_iter = time()
while a < a_end:
    # Update the scale factor, the time step size and the cosmic time
    a_next = expand(a, t, Δt)
    if a_next >= a_end:
        a_before = a
        t_before = t
        t_next = cosmic_time(a_end, a, t, t + Δt)
        Δt = t_next - t
        a_next = a_end
    a = a_next
    t += Δt
    # Leapfrog integration
    particles.drift(scalefactor_integral(-2))
    if a < a_end:
        particles.kick(scalefactor_integral(-1))
    else:
        a_next = expand(a_before, t_before, Δt - leap)
        particles.kick(scalefactor_integral(-1))
    # Animate
    animate(particles, timestep, a)
    # Print out message
    timestep_message(timestep, t_iter, a, t)
    # Update iteration timestamp and number
    t_iter = time()
    timestep += 1
# Plot the final configuration
animate(particles, timestep, a)
