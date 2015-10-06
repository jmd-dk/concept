# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CO𝘕CEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CO𝘕CEPT. If not, see http://www.gnu.org/licenses/
#
# The auther of CO𝘕CEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# This file has to be run in pure Python mode!

# Include the concept_dir in the searched paths and get directory of this file
import sys, os
sys.path.append(os.environ['concept_dir'])
this_dir = os.path.dirname(os.path.realpath(__file__))

# Imports from the CO𝘕CEPT code
from commons import *
from IO import load

# Determine the number of snapshots from the outputlist file
N_snapshots = 1

# Read in data from the CO𝘕CEPT snapshots
particles_cython = []
for i in (1, 2, 4):
    particles_cython.append(load(this_dir + '/output/snapshot_cython_' + str(i), write_msg=False))
particles_python = []
for i in (1, 2, 4):
    particles_python.append(load(this_dir + '/output/snapshot_python_' + str(i), write_msg=False))

# Using the particle order of the 0'th snapshot as the standard, find the corresponding
# ID's in the snapshots and order these particles accoringly.
N = particles_python[0].N
D2 = zeros(N)
ID = zeros(N, dtype='int')
for i in range(N_snapshots):
    for j in range(3):
        x = particles_cython[j].posx
        y = particles_cython[j].posy
        z = particles_cython[j].posz
        x_procs = particles_python[j].posx
        y_procs = particles_python[j].posy
        z_procs = particles_python[j].posz
        for l in range(N):
            for k in range(N):
                dx = x[l] - x_procs[k]
                if dx > half_boxsize:
                    dx -= boxsize
                elif dx < -half_boxsize:
                    dx += boxsize
                dy = y[l] - y_procs[k]
                if dy > half_boxsize:
                    dy -= boxsize
                elif dy < -half_boxsize:
                    dy += boxsize
                dz = z[l] - z_procs[k]
                if dz > half_boxsize:
                    dz -= boxsize
                elif dz < -half_boxsize:
                    dz += boxsize
                D2[k] = dx**2 + dy**2 + dz**2
            ID[l] = np.argmin(D2)
        particles_python[j].posx = particles_python[j].posx[ID]
        particles_python[j].posy = particles_python[j].posy[ID]
        particles_python[j].posz = particles_python[j].posz[ID]
        particles_python[j].momx = particles_python[j].momx[ID]
        particles_python[j].momy = particles_python[j].momy[ID]
        particles_python[j].momz = particles_python[j].momz[ID]

# Compute distance between particles in the snapshot pairs
fig_file = this_dir + '/result.pdf'
x_python = [particles_python[j].posx for j in range(3)]
y_python = [particles_python[j].posx for j in range(3)]
z_python = [particles_python[j].posx for j in range(3)]
x_cython = [particles_cython[j].posx for j in range(3)]
y_cython = [particles_cython[j].posx for j in range(3)]
z_cython = [particles_cython[j].posx for j in range(3)]
dist = [sqrt(array([min([(x_cython[j][i] - x_python[j][i] + xsgn*boxsize)**2 + (y_cython[j][i] - y_python[j][i] + ysgn*boxsize)**2 + (z_cython[j][i] - z_python[j][i] + zsgn*boxsize)**2 for xsgn in (-1, 0, +1) for ysgn in (-1, 0, 1) for zsgn in (-1, 0, 1)]) for i in range(N)])) for j in range(3)]

# Plot
fig, ax = plt.subplots(3, sharex=True, sharey=True)
for i, a, d in zip((1, 2, 4), ax, dist):
    a.plot(d/boxsize, 'sr')
    a.set_ylabel('$|\mathbf{x}_{\mathrm{pp}' + str(i) + '} - \mathbf{x}_{\mathrm{c}' + str(i) + '}|/\mathrm{boxsize}$')
ax[-1].set_xlabel('Particle number')
plt.xlim(0, N - 1)
plt.ylim(0, 1)
fig.subplots_adjust(hspace=0)
plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)
plt.savefig(fig_file)

# Printout error message for unsuccessful test
tol = 1e-2
if any(np.mean(dist[j]/boxsize) > tol for j in range(3)):
    masterwarn('Some or all pure Python runs with nprocs = {1, 2, 4} yielded results\n'
               + 'different from the compiled run!\n'
               + 'See "{}" for a visualization.'.format(fig_file))
    sys.exit(1)

