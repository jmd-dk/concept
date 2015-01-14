# Import everything from the _params_active module (including cython).
# In the .pyx file, this line will be replaced by the content of _params_active.py itself 
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
    from species import construct
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from species cimport construct
    """

# Create random initial particles and save to disk
cython.declare(particles='Particles')
if False:
    N = 1717
    particles = construct('dark matter', pos=random((3, N))*boxsize, vel=0.7*(2*random((3, N))-1), mass=1)
    if master:
        with h5py.File('ICs/test_snapshot', mode='w') as hdf5_file:
            particles_group = hdf5_file.create_group('particles/dark_matter')
            pos = particles_group.create_dataset('pos', (3, particles.N), dtype='float64')
            vel = particles_group.create_dataset('vel', (3, particles.N), dtype='float64')
            pos[...] = particles.pos
            vel[...] = particles.vel
            particles_group.attrs['mass'] = particles.mass
            particles_group.attrs['species'] = 'dark matter'
#  Load particles from disk
with h5py.File('ICs/test_snapshot', mode='r') as hdf5_file:
    # Load all particles
    all_particles = hdf5_file['particles']
    for particle_type in all_particles:
        particles_group = all_particles[particle_type]
        pos = particles_group['pos']
        vel = particles_group['vel']
        # Compute a fair distribution of particle data to the processes
        N = pos.shape[1]
        distribution_sizes = ([N//nprocs]*(nprocs - (N % nprocs))
                              + [N//nprocs + 1]*(N % nprocs))
        distribution_local_size = distribution_sizes[rank]
        distribution_local_start = sum(distribution_sizes[:rank])
        distribution_local_end = distribution_local_start + distribution_local_size
        particles = construct(particles_group.attrs['species'],  # ONLY 'dark matter' FOR NOW (the particles variable should be a tuple of species (or rather particle types) or something)
                              pos=pos[:, distribution_local_start:distribution_local_end],
                              vel=vel[:, distribution_local_start:distribution_local_end],
                              mass=particles_group.attrs['mass'],
                              )


##########################################
#                 TEST
##########################################
print(particles.N)
"""
if rank == 0:
    sendbuf = array([1, 2, 3])
    recvbuf = array([4, 6, 7, -1, -2])
elif rank == 1:
    sendbuf = array([8, 9, 10])
    recvbuf = array([11, 12, 13])
# 1 sender, 0 modtager
if rank == 0:
    Sendrecv(sendbuf, dest=1, recvbuf=recvbuf, source=1)
# 0 sender, 1 modtager
if rank == 1:
    Sendrecv(sendbuf, dest=0, recvbuf=recvbuf, source=0)

print('rank', rank, 'is done', recvbuf)
"""



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
for i in range(25):
    t0 = time()
    particles.kick()
    print(time() - t0)
    particles.drift()
    if True:
        h._offsets3d = juggle_axes(*particles.pos, zdir='z')
        draw()
