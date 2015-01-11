# Import everything from the _params_active module (including cython).
# In the .pyx file, this line will be replaced by the content of _params_active.py itself 
from commons import *

# Use a Matplotlib backend compatible with interactive plotting
#from matplotlib import use as matplotlib_backend
#matplotlib_backend('TkAgg')

# Imports
# Scientific libraries (Numpy, Scipy, Matplotlib)
from numpy.random import random
from numpy import transpose
#from matplotlib.pyplot import figure, gca, draw, show

# 3D plotting
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import juggle_axes
# For timing
from time import time



# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from species import Particles
    from gravity import PP, PM
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    # Particles get cimported from species in the .pxd
    from gravity cimport PP, PM
    """

# Create initial particles
cython.declare(particles='Particles')
particles = Particles(pos=random((3, N))*boxsize, vel=0.7*(2*random((3, N))-1), mass=1)

# Setting up figure and plot the particles
#if master:
#    fig = figure()
#    ax = fig.add_subplot(111, projection='3d')
#    h = ax.scatter(*particles.pos, color='purple', alpha=0.25)
#    gca().set_xlim3d(0, boxsize)
#    gca().set_ylim3d(0, boxsize)
#    gca().set_zlim3d(0, boxsize)
#    gca().set_xlabel(r'$x$')
#    gca().set_ylabel(r'$y$')
#    gca().set_zlabel(r'$z$')
    #ax.view_init(90, 0)
#    show(block=False)
#
# Run main loop
cython.declare(i='size_t')
for i in range(10):
    t0 = time()
    PP(particles)  # This should be a method on particles! It should be able to choose between PP and PM when it is instantiated
    print(time() - t0)
    particles.drift()
    #if master:
    #    h._offsets3d = juggle_axes(*particles.pos, zdir='z')
    #    draw()
