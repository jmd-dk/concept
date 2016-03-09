# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2016 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
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
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# This file has to be run in pure Python mode!

# Include the concept_dir in the searched paths and get directory of this file
import glob, sys, os
sys.path.append(os.environ['concept_dir'])
this_dir = os.path.dirname(os.path.realpath(__file__))

# The name of this test
this_test = os.path.basename(this_dir)

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load_into_standard

# Read in data from the CO𝘕CEPT snapshots
a = []
particles = {1: [], 2: [], 4: [], 8: []}
for n in (1, 2, 4, 8):
    for fname in sorted(glob.glob('{}/output_{}/snapshot_a=*'.format(this_dir, n)),
                        key=lambda s: s[(s.index('=') + 1):]):
        snapshot = load_into_standard(fname, compare_params=False)
        if n == 1:
            a.append(snapshot.params['a'])
        particles[n].append(snapshot.particles_list[0])
N_snapshots = len(a)

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Using the particle order of the n=1 snapshot as the standard, find the corresponding
# ID's in the snapshots and order these particles accordingly.
N = particles[1][0].N
D2 = zeros(N)
ID = zeros(N, dtype='int')
for i in range(N_snapshots):
    x = particles[1][i].posx
    y = particles[1][i].posy
    z = particles[1][i].posz
    for n in (2, 4, 8):
        x_procs = particles[n][i].posx
        y_procs = particles[n][i].posy
        z_procs = particles[n][i].posz
        for j in range(N):
            for k in range(N):
                dx = x[j] - x_procs[k]
                if dx > 0.5*boxsize:
                    dx -= boxsize
                elif dx < -0.5*boxsize:
                    dx += boxsize
                dy = y[j] - y_procs[k]
                if dy > 0.5*boxsize:
                    dy -= boxsize
                elif dy < -0.5*boxsize:
                    dy += boxsize
                dz = z[j] - z_procs[k]
                if dz > 0.5*boxsize:
                    dz -= boxsize
                elif dz < -0.5*boxsize:
                    dz += boxsize
                D2[k] = dx**2 + dy**2 + dz**2
            ID[j] = np.argmin(D2)
        particles[n][i].posx = particles[n][i].posx[ID]
        particles[n][i].posy = particles[n][i].posy[ID]
        particles[n][i].posz = particles[n][i].posz[ID]
        particles[n][i].momx = particles[n][i].momx[ID]
        particles[n][i].momy = particles[n][i].momy[ID]
        particles[n][i].momz = particles[n][i].momz[ID]

# Compute distance between particles in the two snapshots
dist = collections.OrderedDict([(2, []), (4, []), (8, [])])
for i in range(N_snapshots):
    x = {n: particles[n][i].posx for n in (1, 2, 4, 8)}
    y = {n: particles[n][i].posy for n in (1, 2, 4, 8)}
    z = {n: particles[n][i].posz for n in (1, 2, 4, 8)}
    for n in (2, 4, 8):
        dist[n].append(sqrt(np.array([min([(x[1][j] - x[n][j] + xsgn*boxsize)**2 + (y[1][j] - y[n][j] + ysgn*boxsize)**2 + (z[1][j] - z[n][j] + zsgn*boxsize)**2 for xsgn in (-1, 0, +1) for ysgn in (-1, 0, +1) for zsgn in (-1, 0, +1)]) for j in range(N)])))

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(3, sharex=True, sharey=True)
for n, d, ax_i in zip(dist.keys(), dist.values(), ax):
    for i in range(N_snapshots):
        ax_i.plot(np.array(d[i])/boxsize, '.', alpha=.7, label='$a={}$'.format(a[i]), zorder=-i)
    ax_i.set_ylabel('$|\mathbf{x}_{' + str(n) + '} - \mathbf{x}_1|/\mathrm{boxsize}$')
ax[-1].set_xlabel('Particle number')
plt.xlim(0, N - 1)
fig.subplots_adjust(hspace=0)
plt.setp([ax_i.get_xticklabels() for ax_i in ax[:-1]], visible=False)
ax[0].legend(loc='best').get_frame().set_alpha(0.3)
plt.tight_layout()
plt.savefig(fig_file)

# Done analyzing
masterprint('done')

# Printout error message for unsuccessful test
tol = 1e-9
if any(np.mean(np.array(dist[n])/boxsize) > tol for n in (2, 4, 8)):
    masterwarn('Runs with different numbers of processes yield different results!\n'
               + 'See "{}" for a visualization.'.format(fig_file))
    sys.exit(1)

