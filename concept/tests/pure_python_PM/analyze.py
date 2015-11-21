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
from snapshot import load_particles

# Determine the number of snapshots from the outputlist file
N_snapshots = np.loadtxt(this_dir + '/outputlist').size

# Read in data from the CO𝘕CEPT snapshots
particles = [load_particles(this_dir + '/output/snapshot_cython', compare_params=False)]
for i in (1, 2, 4):
    particles.append(load_particles(this_dir + '/output/snapshot_python_' + str(i), compare_params=False))

# Using the particle order of the 0'th snapshot as the standard, find the corresponding
# ID's in the snapshots and order these particles accoringly.
N = particles[0].N
D2 = zeros(N)
ID = zeros(N, dtype='int')
for i in range(N_snapshots):
    x = particles[0].posx
    y = particles[0].posy
    z = particles[0].posz
    for j in range(1, 4):
        x_procs = particles[j].posx
        y_procs = particles[j].posy
        z_procs = particles[j].posz
        for l in range(N):
            for k in range(N):
                dx = x[l] - x_procs[k]
                if dx > 0.5*boxsize:
                    dx -= boxsize
                elif dx < -0.5*boxsize:
                    dx += boxsize
                dy = y[l] - y_procs[k]
                if dy > 0.5*boxsize:
                    dy -= boxsize
                elif dy < -0.5*boxsize:
                    dy += boxsize
                dz = z[l] - z_procs[k]
                if dz > 0.5*boxsize:
                    dz -= boxsize
                elif dz < -0.5*boxsize:
                    dz += boxsize
                D2[k] = dx**2 + dy**2 + dz**2
            ID[l] = np.argmin(D2)
        particles[j].posx = particles[j].posx[ID]
        particles[j].posy = particles[j].posy[ID]
        particles[j].posz = particles[j].posz[ID]
        particles[j].momx = particles[j].momx[ID]
        particles[j].momy = particles[j].momy[ID]
        particles[j].momz = particles[j].momz[ID]

# Compute distance between particles in the two snapshots
fig_file = this_dir + '/result.png'
x = [particles[j].posx for j in range(4)]
y = [particles[j].posx for j in range(4)]
z = [particles[j].posx for j in range(4)]
dist = [sqrt(np.array([min([(x[0][i] - x[j][i] + xsgn*boxsize)**2 + (y[0][i] - y[j][i] + ysgn*boxsize)**2 + (z[0][i] - z[j][i] + zsgn*boxsize)**2 for xsgn in (-1, 0, +1) for ysgn in (-1, 0, 1) for zsgn in (-1, 0, 1)]) for i in range(N)])) for j in range(4)]

# Plot
fig, ax = plt.subplots(3, sharex=True, sharey=True)
for i, a, d in zip((1, 2, 4), ax, dist[1:]):
    a.plot(d/boxsize, 'sr')
    a.set_ylabel('$|\mathbf{x}_{\mathrm{pp}' + str(i) + '} - \mathbf{x}_{\mathrm{c}}|/\mathrm{boxsize}$')
ax[-1].set_xlabel('Particle number')
plt.xlim(0, N - 1)
plt.ylim(0, 1)
fig.subplots_adjust(hspace=0)
plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)
plt.savefig(fig_file)

# Printout error message for unsuccessful test
tol = 1e-3
if any(np.mean(dist[j]/boxsize) > tol for j in range(4)):
    masterwarn('Some or all pure Python runs with nprocs = {1, 2, 4} yielded results\n'
               + 'different from the compiled run!\n'
               + 'See "{}" for a visualization.'.format(fig_file))
    sys.exit(1)

