# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Plotting
from matplotlib import use as matplotlib_backend
matplotlib_backend('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
from matplotlib.pyplot import figure, draw, show, savefig
import matplotlib.cm as cm
from matplotlib import animation

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from species import construct, construct_random
    from IO import load, save
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from species cimport construct, construct_random
    from IO cimport load, save
    """


# Construct
cython.declare(particles='Particles')
particles = construct_random('some typename', 'dark matter', N=20000)


H0 = 70*units.km/units.s/units.Mpc
particles.mass = 3*H0**2/(8*pi*G_Newton)*boxsize**3/particles.N


"""
particles.mass = 1
particles.posx[0] = 5 - 0.1
particles.posx[1] = 5 + 0.1
particles.posy[0] = 5
particles.posy[1] = 5
particles.posz[0] = 5
particles.posz[1] = 5
particles.velx[0] = 0
particles.velx[1] = 0
particles.vely[0] = 0.21
particles.vely[1] = -0.21
particles.velz[0] = 0
particles.velz[1] = 0
"""




# Save
save(particles, 'ICs/test')

# Load (and thereby order them correctly)
particles = load('ICs/test')


# Setting up figure and plot the particles
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               # Locals
               N='size_t',
               N_local='size_t',
               N_locals='size_t[::1]',
               X='double[::1]',
               Y='double[::1]',
               Z='double[::1]',
               i='int',
               j='int',
               )
def animate(particles, artist=None):
    N = particles.N
    N_local = particles.N_local
    # The master process gathers N_local from all processes
    N_locals = empty(nprocs if master else 0, dtype='uintp')
    Gather(array(N_local, dtype='uintp'), N_locals)
    # The master process gathers all particle data
    X = empty(N if master else 0)
    Y = empty(N if master else 0)
    Z = empty(N if master else 0)
    sendbuf = particles.posx_mw[:N_local]
    Gatherv(sendbuf=sendbuf, recvbuf=(X, N_locals))
    sendbuf = particles.posy_mw[:N_local]
    Gatherv(sendbuf=sendbuf, recvbuf=(Y, N_locals))
    sendbuf = particles.posz_mw[:N_local] 
    Gatherv(sendbuf=sendbuf, recvbuf=(Z, N_locals))
    # The master process plots the particle data
    if master:
        if artist is None:
            # Set up figure
            fig = figure()
            ax = fig.add_subplot(111, projection='3d', axisbg='black')
            colors = ['purple']*nprocs#['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray'][:nprocs]
            artist = ax.scatter(X, Y, Z,
                                lw=0,
                                alpha=0.05,
                                c=[colors[(i%nprocs)]
                                   for i in range(nprocs)
                                   for j in range(N_locals[i])],
                                s=20,
                                )
            ax.set_xlim3d(0, boxsize)#(5-3, 5+3)
            ax.set_ylim3d(0, boxsize)#(5-3, 5+3)
            ax.set_zlim3d(0, boxsize)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlabel(r'$z$')
            #ax.view_init(90, 0)
            ax.grid(False)
            ax.set_axis_off()
            fig.patch.set_facecolor('black')
            show(block=False)
        else:
            # Update figure
            artist._offsets3d = juggle_axes(X, Y, Z, zdir='z')
            draw()
    return artist

# Set up animation
visualize = True
frameskip = 3
if visualize:
    artist = animate(particles)




cython.declare(i='size_t',
               dt='double',
               a='double',
               )
dt = 100*units.Myr
a = 0
i = 0

# First leapfrog kick
particles.kick(dt/2)
# Main time loop
while True:
    i += 1
    if master:
        t0 = time()

    # Update the scale factor   FIND OUT HOW TO DO THIS CORRECTLY! FIRST IMPLEMENT UNITS THOUGH
    a += 0.000001*dt    *0

    # Leapfrog integration step
    particles.drift(dt)
    if a < 1:
        particles.kick(dt)
    else:
        # Last leapfrog kick
        particles.kick(dt/2)
        break
    
    if master and True:
        t1 = time()
        print('Computing time:', t1 - t0)

    # Animate
    if visualize and not (i%frameskip):
        artist = animate(particles, artist)
        if master and True:
            savefig('frames/' + str(i) + '.png', bbox_inches='tight')
            print('Plot time:', time() - t1)
