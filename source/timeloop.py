# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Use a Matplotlib backend compatible with interactive plotting
from matplotlib import use as matplotlib_backend
matplotlib_backend('TkAgg')

# Imports
# Scientific libraries (Numpy, Scipy, Matplotlib)
from numpy.random import random
from numpy import transpose
from matplotlib.pyplot import figure, gca, draw, show

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
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
cython.declare(particles_ori='Particles')
particles_ori = construct_random('some typename', 'dark matter', N=1717)

# Save
save(particles_ori, 'ICs/test')

# Load (and thereby order them correctly)
cython.declare(particles='Particles')
particles = load('ICs/test')


# Setting up figure and plot the particles
if True:
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    h = ax.scatter(*particles.pos, color='purple', alpha=0.25)
    gca().set_xlim3d(0, boxsize)
    gca().set_ylim3d(0, boxsize)
    gca().set_zlim3d(0, boxsize)
    gca().set_xlabel(r'$x$')
    gca().set_ylabel(r'$y$')
    gca().set_zlabel(r'$z$')
    #ax.view_init(90, 0)
    show(block=False)

# Run main loop
cython.declare(i='size_t')
for i in range(100):
    t0 = time()
    particles.kick()
    #print(time() - t0)
    particles.drift()
    if True:
        h._offsets3d = juggle_axes(*particles.pos, zdir='z')
        draw()

