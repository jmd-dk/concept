# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Imports for plotting
from matplotlib import use as matplotlib_backend
matplotlib_backend('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
from matplotlib.pyplot import figure, draw, show, savefig

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    pass
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    """

# Imports and definitions common to pure Python and Cython
import os

# Setting up figure and plot the particles
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               timestep='size_t',
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
def animate(particles, timestep):
    global artist
    if not visualize or (timestep%framespace):
        return
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
            artist = ax.scatter(X, Y, Z,
                                lw=0,
                                alpha=0.2,
                                c='purple',
                                s=10,
                                )
            ax.set_xlim3d(0, boxsize)
            ax.set_ylim3d(0, boxsize)
            ax.set_zlim3d(0, boxsize)
            #ax.view_init(90, 0)
            ax.set_axis_off()
            show(block=False)
        else:
            # Update figure
            artist._offsets3d = juggle_axes(X, Y, Z, zdir='z')
            draw()
            if saveframes:
                savefig(framefolder + str(timestep) + '.png',
                        bbox_inches='tight', pad_inches=0, dpi=160)
            if liveframes:
                # Save the live frame
                savefig(liveframe,
                        bbox_inches='tight', pad_inches=0, dpi=160)

# This function formats a floating point number f to have the length n
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               f='double',
               n='int',
               )
@cython.returns('str')
def format_number(f, n):
    return ('{:.' + str(n - len(str(int(f)))) + 'f}').format(f).ljust(n + 1)

# This function pretty prints information gathered through a time step
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               timestep='int',
               t_iter='double',
               a='double',
               t='double',
               )
def timestep_message(timestep, t_iter, a, t):
    if master:
        print('Time step ' + str(timestep) + ':',
              'Computation time: ' + format_number(time() - t_iter, 5) + ' s',
              'Scale factor:     ' + format_number(a, 5),
              'Cosmic time:      ' + format_number(t/units.Gyr, 5) + ' Gyr',
              sep='\n    ')

# Set the artist as uninitialized at import time
artist = None
# Check whether frames should be stored and create the
# framefolder folder at import time
cython.declare(saveframes='bint')
if framefolder == '':
    saveframes = False
else:
    saveframes = True
    if not os.path.exists(framefolder):
        os.makedirs(framefolder)
    if framefolder[-1] != '/':
        framefolder += '/'
# Check whether the latest frame should be broadcasted live
cython.declare(liveframes='bint')
if liveframe == '':
    liveframes = False
else:
    liveframes = True
