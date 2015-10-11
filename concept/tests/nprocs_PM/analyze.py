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
particles = []
for i in range(N_snapshots):
    fname = 'snapshot_a={:.2f}'.format(np.loadtxt(this_dir + '/outputlist')[i])
    particles.append([load_particles(this_dir + '/output_' + str(j) + '/' + fname, compare_params=False) for j in (1, 2, 4, 8)])

# Using the particle order of the 0'th snapshot as the standard, find the corresponding
# ID's in the snapshots and order these particles accoringly.
N = particles[0][0].N
D2 = zeros(N)
ID = zeros(N, dtype='int')
for i in range(N_snapshots):
    x = particles[i][0].posx
    y = particles[i][0].posy
    z = particles[i][0].posz
    for j in (1, 2, 3):
        x_procs = particles[i][j].posx
        y_procs = particles[i][j].posy
        z_procs = particles[i][j].posz
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
        particles[i][j].posx = particles[i][j].posx[ID]
        particles[i][j].posy = particles[i][j].posy[ID]
        particles[i][j].posz = particles[i][j].posz[ID]
        particles[i][j].momx = particles[i][j].momx[ID]
        particles[i][j].momy = particles[i][j].momy[ID]
        particles[i][j].momz = particles[i][j].momz[ID]

# Compute distance between particles in the two snapshots
fig_file = this_dir + '/result.png'
x = [particles[-1][j].posx for j in range(4)]
y = [particles[-1][j].posx for j in range(4)]
z = [particles[-1][j].posx for j in range(4)]
dist = [sqrt(array([min([(x[0][i] - x[j][i] + xsgn*boxsize)**2 + (y[0][i] - y[j][i] + ysgn*boxsize)**2 + (z[0][i] - z[j][i] + zsgn*boxsize)**2 for xsgn in (-1, 0, +1) for ysgn in (-1, 0, 1) for zsgn in (-1, 0, 1)]) for i in range(N)])) for j in range(4)]

# Plot
fig, ax = plt.subplots(3, sharex=True, sharey=True)
for i, a, d in zip((2, 4, 8), ax, dist[1:]):
    a.plot(d/boxsize, 'sr')
    a.set_ylabel('$|\mathbf{x}_{' + str(i) + '} - \mathbf{x}_1|/\mathrm{boxsize}$')
ax[-1].set_xlabel('Particle number')
plt.xlim(0, N - 1)
plt.ylim(0, 1)
fig.subplots_adjust(hspace=0)
plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)
plt.savefig(fig_file)

# Printout error message for unsuccessful test
tol = 1e-9
if any(np.mean(dist[j]/boxsize) > tol for j in range(4)):
    masterwarn('Runs with different numbers of processes yield different results!\n'
          + 'See "{}" for a visualization.'.format(fig_file))
    sys.exit(1)

