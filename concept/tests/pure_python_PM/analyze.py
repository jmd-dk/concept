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
particles = {'cython': {1: [], 2: [], 4: []},
             'python': {1: [], 2: [], 4: []}}
for cp in particles.keys():
    for n in (1, 2, 4):
        for fname in sorted(glob.glob('{}/output_{}_{}/snapshot_a=*'.format(this_dir, cp, n)),
                            key=lambda s: s[(s.index('=') + 1):]):
            snapshot = load_into_standard(fname, compare_params=False)
            if cp == 'cython' and n == 1:
                a.append(snapshot.params['a'])
            particles[cp][n].append(snapshot.particles_list[0])
N_snapshots = len(a)

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Using the particle order of the cython snapshot as the standard, find the corresponding
# ID's in the python snapshots and order these particles accoringly.
N = particles['cython'][1][0].N
D2 = zeros(N)
ID = zeros(N, dtype='int')
for i in range(N_snapshots):
    for n in (1, 2, 4):
        x_cython = particles['cython'][n][i].posx
        y_cython = particles['cython'][n][i].posy
        z_cython = particles['cython'][n][i].posz
        x_python = particles['python'][n][i].posx
        y_python = particles['python'][n][i].posy
        z_python = particles['python'][n][i].posz
        for j in range(N):
            for k in range(N):
                dx = x_cython[j] - x_python[k]
                if dx > 0.5*boxsize:
                    dx -= boxsize
                elif dx < -0.5*boxsize:
                    dx += boxsize
                dy = y_cython[j] - y_python[k]
                if dy > 0.5*boxsize:
                    dy -= boxsize
                elif dy < -0.5*boxsize:
                    dy += boxsize
                dz = z_cython[j] - z_python[k]
                if dz > 0.5*boxsize:
                    dz -= boxsize
                elif dz < -0.5*boxsize:
                    dz += boxsize
                D2[k] = dx**2 + dy**2 + dz**2
            ID[j] = np.argmin(D2)
        particles['python'][n][i].posx = particles['python'][n][i].posx[ID]
        particles['python'][n][i].posy = particles['python'][n][i].posy[ID]
        particles['python'][n][i].posz = particles['python'][n][i].posz[ID]
        particles['python'][n][i].momx = particles['python'][n][i].momx[ID]
        particles['python'][n][i].momy = particles['python'][n][i].momy[ID]
        particles['python'][n][i].momz = particles['python'][n][i].momz[ID]

# Compute distance between particles in the two snapshots
dist = collections.OrderedDict([(1, []), (2, []), (4, [])])
for i in range(N_snapshots):
    x = {(cp, n): particles[cp][n][i].posx for cp in ('cython', 'python') for n in (1, 2, 4)}
    y = {(cp, n): particles[cp][n][i].posy for cp in ('cython', 'python') for n in (1, 2, 4)}
    z = {(cp, n): particles[cp][n][i].posz for cp in ('cython', 'python') for n in (1, 2, 4)}
    for n in (1, 2, 4):
        dist[n].append(sqrt(np.array([min([(x['cython', n][j] - x['python', n][j] + xsgn*boxsize)**2 + (y['cython', n][j] - y['python', n][j] + ysgn*boxsize)**2 + (z['cython', n][j] - z['python', n][j] + zsgn*boxsize)**2 for xsgn in (-1, 0, +1) for ysgn in (-1, 0, +1) for zsgn in (-1, 0, +1)]) for j in range(N)])))

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(3, sharex=True, sharey=True)
for n, d, ax_i in zip(dist.keys(), dist.values(), ax):
    for i in range(N_snapshots):
        ax_i.plot(np.array(d[i])/boxsize, '.', alpha=.7, label='$a={}$'.format(a[i]), zorder=-i)
    ax_i.set_ylabel('$|\mathbf{x}_{\mathrm{pp}' + str(n) + '} - \mathbf{x}_{\mathrm{c}' + str(n) + '}|/\mathrm{boxsize}$')
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
tol = 1e-5
if any(np.mean(np.array(d)/boxsize) > tol for d in dist.values()):
    masterwarn('Some or all pure Python runs with nprocs = {1, 2, 4} yielded results\n'
               + 'different from their compiled counterparts!\n'
               + 'See "{}" for a visualization.'.format(fig_file))
    sys.exit(1)

