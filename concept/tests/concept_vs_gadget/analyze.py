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
from IO import Gadget_snapshot
from graphics import animate

# Use a matplotlib backend that does not require a running X-server
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Determine the number of snapshots from the outputlist file
N_snapshots = np.loadtxt(this_dir + '/outputlist').size

# Instantiate a Gadget_snapshot instance which will be reused for all GADGET snapshots
snapshot = Gadget_snapshot()
snapshot.load(this_dir + '/IC')

# Read in data from the CONCEPT snapshots
a = []
particles = []
for i in range(N_snapshots):
    snapshot.load(this_dir + '/output/snapshot_' + str(i))
    a.append(snapshot.header['Time'])
    particles.append(snapshot.particles)

# Read in data from the GADGET snapshots
a_gadget = []
particles_gadget = []
for i in range(N_snapshots):
    snapshot.load(this_dir + '/output/snapshot_gadget_' + '0'*(3-len(str(i))) + str(i))
    a_gadget.append(snapshot.header['Time'])
    particles_gadget.append(snapshot.particles)

# Using the particle order of CONCEPT as the standard, find the corresponding
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
            if dx > half_boxsize:
                dx -= boxsize
            elif dx < -half_boxsize:
                dx += boxsize
            dy = y[j] - y_gadget[k]
            if dy > half_boxsize:
                dy -= boxsize
            elif dy < -half_boxsize:
                dy += boxsize
            dz = z[j] - z_gadget[k]
            if dz > half_boxsize:
                dz -= boxsize
            elif dz < -half_boxsize:
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
fig_file = this_dir + '/result.pdf'
x = particles[-1].posx
y = particles[-1].posy
z = particles[-1].posz
x_gadget = particles_gadget[-1].posx
y_gadget = particles_gadget[-1].posy
z_gadget = particles_gadget[-1].posz
dist = sqrt(array([min([(x[i] - x_gadget[i] + xsgn*boxsize)**2 + (y[i] - y_gadget[i] + ysgn*boxsize)**2 + (z[i] - z_gadget[i] + zsgn*boxsize)**2 for xsgn in (-1, 0, +1) for ysgn in (-1, 0, 1) for zsgn in (-1, 0, 1)]) for i in range(N)]))

# Plot
plt.plot(dist/boxsize, 'sr')
plt.xlabel('Particle number')
plt.ylabel('$|\mathbf{x}_{\mathrm{CONCEPT}} - \mathbf{x}_{\mathrm{GADGET}}|/\mathrm{boxsize}$')
plt.xlim(0, N -1)
plt.ylim(0, 1)
plt.savefig(fig_file)

# Compare CONCEPT to GADGET
tol = 1e-2
if np.mean(dist/boxsize) > tol:
    print('\033[1m\033[91m' + 'The results from CONCEPT disagree with those from GADGET.\n'
          + 'See ' + fig_file + ' for a visualization.' + '\033[0m')
    sys.exit(1)

