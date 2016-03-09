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
from snapshot import load_into_standard, load_particles

# Read in data from the CO𝘕CEPT snapshots
a = []
particles = []
for fname in sorted(glob.glob(this_dir + '/output/snapshot_a=*'),
                    key=lambda s: s[(s.index('=') + 1):]):
    snapshot = load_into_standard(fname, compare_params=False)
    a.append(snapshot.params['a'])
    particles.append(snapshot.particles_list[0])
N_snapshots = len(a)

# Read in data from the GADGET snapshots
particles_gadget = []
for fname in sorted(glob.glob(this_dir + '/output/snapshot_gadget_*'),
                    key=lambda s: s[(s.index('gadget_') + 7):])[:N_snapshots]:
    particles_gadget.append(load_particles(fname, compare_params=False)[0])

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Using the particle order of CO𝘕CEPT as the standard, find the corresponding
# ID's in the GADGET snapshots and order these particles accoringly.
N = particles[0].N
D2 = zeros(N)
ID = zeros(N, dtype='int')
for i in range(N_snapshots):
    x = particles[i].posx
    y = particles[i].posy
    z = particles[i].posz
    x_gadget = particles_gadget[i].posx
    y_gadget = particles_gadget[i].posy
    z_gadget = particles_gadget[i].posz
    for j in range(N):
        for k in range(N):
            dx = x[j] - x_gadget[k]
            if dx > 0.5*boxsize:
                dx -= boxsize
            elif dx < -0.5*boxsize:
                dx += boxsize
            dy = y[j] - y_gadget[k]
            if dy > 0.5*boxsize:
                dy -= boxsize
            elif dy < -0.5*boxsize:
                dy += boxsize
            dz = z[j] - z_gadget[k]
            if dz > 0.5*boxsize:
                dz -= boxsize
            elif dz < -0.5*boxsize:
                dz += boxsize
            D2[k] = dx**2 + dy**2 + dz**2
        ID[j] = np.argmin(D2)
    particles_gadget[i].posx = particles_gadget[i].posx[ID]
    particles_gadget[i].posy = particles_gadget[i].posy[ID]
    particles_gadget[i].posz = particles_gadget[i].posz[ID]
    particles_gadget[i].momx = particles_gadget[i].momx[ID]
    particles_gadget[i].momy = particles_gadget[i].momy[ID]
    particles_gadget[i].momz = particles_gadget[i].momz[ID]

# Compute distance between particles in the two snapshots
dist = []
for i in range(N_snapshots):
    x = particles[i].posx
    y = particles[i].posy
    z = particles[i].posz
    x_gadget = particles_gadget[i].posx
    y_gadget = particles_gadget[i].posy
    z_gadget = particles_gadget[i].posz
    dist.append(sqrt(np.array([min([(x[j] - x_gadget[j] + xsgn*boxsize)**2 + (y[j] - y_gadget[j] + ysgn*boxsize)**2 + (z[j] - z_gadget[j] + zsgn*boxsize)**2 for xsgn in (-1, 0, +1) for ysgn in (-1, 0, +1) for zsgn in (-1, 0, +1)]) for j in range(N)])))
    # Plot
    plt.plot(dist[i]/boxsize, '.', alpha=.7, label='$a={}$'.format(a[i]), zorder=-i)

# Finalize plot
fig_file = this_dir + '/result.png'
plt.xlabel('Particle number')
plt.ylabel('$|\mathbf{x}_{\mathrm{CO}N\mathrm{CEPT}} - \mathbf{x}_{\mathrm{GADGET}}|/\mathrm{boxsize}$')
plt.xlim(0, N - 1)
plt.legend(loc='best').get_frame().set_alpha(0.3)
plt.tight_layout()
plt.savefig(fig_file)

# Done analyzing
masterprint('done')

# Printout error message for unsuccessful test
tol = 1e-3
if any(np.mean(d/boxsize) > tol for d in dist):
    masterwarn('The results from {} disagree with those from GADGET.\n'.format(terminal.CONCEPT)
               + 'See "{}" for a visualization.'.format(fig_file))
    sys.exit(1)

