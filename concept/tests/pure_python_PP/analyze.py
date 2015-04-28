# This file has to be run in pure Python mode!

# Include the code directory in the searched paths
import sys, os
concept_dir = os.path.realpath(__file__)
this_dir = os.path.dirname(concept_dir)
while True:
    if concept_dir == '/':
        raise Exception('Cannot find the .paths file!')
    if '.paths' in os.listdir(os.path.dirname(concept_dir)):
        break
    concept_dir = os.path.dirname(concept_dir)
sys.path.append(concept_dir)

# Imports from the CONCEPT code
from commons import *
from IO import load
from graphics import animate

# Use a matplotlib backend that does not require a running X-server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Determine the number of snapshots from the outputlist file
N_snapshots = np.loadtxt(this_dir + '/outputlist').size

# Read in data from the CONCEPT snapshots
particles = [load(this_dir + '/output/snapshot_cython', write_msg=False)]
for i in (1, 2, 4):
    particles.append(load(this_dir + '/output/snapshot_python_' + str(i), write_msg=False))

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
        particles[j].posx = particles[j].posx[ID]
        particles[j].posy = particles[j].posy[ID]
        particles[j].posz = particles[j].posz[ID]
        particles[j].momx = particles[j].momx[ID]
        particles[j].momy = particles[j].momy[ID]
        particles[j].momz = particles[j].momz[ID]

# Compute distance between particles in the two snapshots
fig_file = this_dir + '/result.pdf'
x = [particles[j].posx for j in range(4)]
y = [particles[j].posx for j in range(4)]
z = [particles[j].posx for j in range(4)]
dist = [sqrt(array([min([(x[0][i] - x[j][i] + xsgn*boxsize)**2 + (y[0][i] - y[j][i] + ysgn*boxsize)**2 + (z[0][i] - z[j][i] + zsgn*boxsize)**2 for xsgn in (-1, 0, +1) for ysgn in (-1, 0, 1) for zsgn in (-1, 0, 1)]) for i in range(N)])) for j in range(4)]

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

# Compare CONCEPT to GADGET
tol = 1e-6
if any(np.mean(dist[j]/boxsize) > tol for j in range(4)):
    print('\033[1m\033[91m' + 'Some or all pure Python runs with nprocs = {1, 2, 4} yielded results\n'
          + 'different from the compiled run!\n'
          + 'See ' + fig_file + ' for a visualization.' + '\033[0m')
    sys.exit(1)
