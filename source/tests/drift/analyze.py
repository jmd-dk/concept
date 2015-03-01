# This file has to be run in pure Python mode!

# Include the actual code directory in the searched paths
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))[:-1]))

# Imports from the N-body code
from commons import *
from IO import Gadget_snapshot

# Use a matplotlib backend that does not require a running X-server
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

N_snapshots = 25
snapshot = Gadget_snapshot()

# Read in data from the N-body snapshots
a = zeros(N_snapshots)
x = zeros(N_snapshots)
x_std = zeros(N_snapshots)
for i in range(N_snapshots):
    snapshot.load('tests/drift/output/snapshot_' + str(i))
    a[i] = snapshot.header['Time']
    x[i] = np.mean(snapshot.particles.posx)
    x_std[i] = np.std(snapshot.particles.posx)

# Read in data from the Gadget snapshots
a_gadget = zeros(N_snapshots)
x_gadget = zeros(N_snapshots)
x_std_gadget = zeros(N_snapshots)
for i in range(N_snapshots):
    snapshot.load('tests/drift/output/snapshot_gadget_' + '0'*(3-len(str(i))) + str(i))
    a_gadget[i] = snapshot.header['Time']
    x_gadget[i] = np.mean(snapshot.particles.posx)
    x_std_gadget[i] = np.std(snapshot.particles.posx)

# Plot
fig_file = 'tests/drift/result.pdf'
plt.text(0.5*max(a), 0.93*boxsize, r'$\uparrow$ End of simulation box $\uparrow$', ha='center')
plt.errorbar(a, x/units.kpc, yerr=x_std/units.kpc, fmt='-or', label='$N$-body')
plt.errorbar(a_gadget, x_gadget/units.kpc, yerr=x_std_gadget/units.kpc, fmt='--xb', label='Gadget')
plt.legend(loc='best')
plt.xlabel('$a$')
plt.ylabel(r'$x\,\mathrm{[kpc]}$')
plt.ylim(0, boxsize)
plt.savefig(fig_file)

# Analyze
# There should be no variance on the x positions
tol = N_snapshots*100*np.finfo('float32').eps
if np.sum(x_std_gadget) > tol:
    print('\033[1m\033[91m' + 'Unequal x-positions for the 4 particles in the Gadget snapshots. '
          + 'It is no good to compare the N-body results to these.' + '\033[0m')
    sys.exit(1)
if np.sum(x_std) > tol:
    print('\033[1m\033[91m' + 'Unequal x-positions for the 4 particles in the snapshots. '
          + 'The symmetric initial conditions has produced nonsymmetric results!' + '\033[0m')
    sys.exit(1)
# Compare N-body to Gadget
tol = 1e-3
if max(np.abs(x/x_gadget - 1)) > tol:
    print('\033[1m\033[91m' + 'The results from the N-body code disagree with those from Gadget. '
          + 'See ' + fig_file + ' for a visualization.' + '\033[0m')
    sys.exit(1)
