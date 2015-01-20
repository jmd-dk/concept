# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Plotting
from matplotlib import use as matplotlib_backend
matplotlib_backend('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
from matplotlib.pyplot import figure, draw, show
import matplotlib.cm as cm
from matplotlib import animation

# For timing
from time import time, sleep

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
particles = construct_random('some typename', 'dark matter', N=100)



# Save
save(particles, 'ICs/test')

# Load (and thereby order them correctly)
particles = load('ICs/test')


# Set up animation
visualize = True
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
            ax = fig.add_subplot(111, projection='3d')
            if nprocs == 1:
                colors = zeros(N_local)
            else:
                colors = [i/(nprocs - 1.0) for i in range(nprocs) for j in range(N_locals[i])]
            artist = ax.scatter(X, Y, Z, c=colors, cmap=cm.nipy_spectral, alpha=0.1)
            ax.set_xlim3d(0, boxsize)
            ax.set_ylim3d(0, boxsize)
            ax.set_zlim3d(0, boxsize)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlabel(r'$z$')
            #ax.view_init(90, 0)
            show(block=False)
        else:
            # Update figure
            artist._offsets3d = juggle_axes(X, Y, Z, zdir='z')
            draw()
    return artist
if visualize:
    artist = animate(particles)

# Run main loop
cython.declare(i='size_t')
for i in range(10000000):
    if master:
        t0 = time()
    particles.kick()
    particles.drift()
    if master:
        t1 = time()
        print('Computing time:', t1 - t0)
    # Animate
    if visualize:
        artist = animate(particles, artist)
        if master:
            print('Plot time:', time() - t1)
