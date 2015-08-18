# Copyright (C) 2015 Jeppe Mosgard Dakin
#
# This file is part of CONCEPT, the cosmological N-body code in Python
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.



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

# Use a matplotlib backend that does not require a running X-server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Determine the number of snapshots from the outputlist file
N_snapshots = len(np.loadtxt(this_dir + '/outputlist'))

# Instantiate a Gadget_snapshot instance which will be reused for all GADGET snapshots
snapshot = Gadget_snapshot()

# Read in data from the CONCEPT snapshots
a = zeros(N_snapshots)
x0 = zeros(N_snapshots)
x0_std = zeros(N_snapshots)
x1 = zeros(N_snapshots)
x1_std = zeros(N_snapshots)
for i in range(N_snapshots):
    fname = 'snapshot_a={:.3f}'.format(np.loadtxt(this_dir + '/outputlist')[i])
    snapshot.load(this_dir + '/output/' + fname, write_msg=False)
    a[i] = snapshot.header['Time']
    x0[i] = np.mean(snapshot.particles.posx[:4])
    x0_std[i] = np.std(snapshot.particles.posx[:4])
    x1[i] = np.mean(snapshot.particles.posx[4:])
    x1_std[i] = np.std(snapshot.particles.posx[4:])

# Read in data from the GADGET snapshots
a_gadget = zeros(N_snapshots)
x0_gadget = zeros(N_snapshots)
x0_std_gadget = zeros(N_snapshots)
x1_gadget = zeros(N_snapshots)
x1_std_gadget = zeros(N_snapshots)
for i in range(N_snapshots):
    snapshot.load(this_dir + '/output/snapshot_gadget_' + '0'*(3-len(str(i))) + str(i), write_msg=False)
    a_gadget[i] = snapshot.header['Time']
    x_gadget = snapshot.particles.posx[np.argsort(snapshot.ID)]
    x0_gadget[i] = np.mean(x_gadget[:4])
    x0_std_gadget[i] = np.std(x_gadget[:4])
    x1_gadget[i] = np.mean(x_gadget[4:])
    x1_std_gadget[i] = np.std(x_gadget[4:])

# Plot
fig_file = this_dir + '/result.pdf'
plt.errorbar(a, x0/units.kpc, yerr=x0_std/units.kpc, fmt='-sr', label='CO$N$CEPT (left)')
plt.errorbar(a, x1/units.kpc, yerr=x1_std/units.kpc, fmt='-Dr', label='CO$N$CEPT (right)')
plt.errorbar(a_gadget, x0_gadget/units.kpc, yerr=x0_std_gadget/units.kpc, fmt='--<b', label='GADGET (left)')
plt.errorbar(a_gadget, x1_gadget/units.kpc, yerr=x1_std_gadget/units.kpc, fmt='-->b', label='GADGET (right)')
plt.legend(loc='best')
plt.xlabel('$a$')
plt.ylabel(r'$x\,\mathrm{[kpc]}$')
plt.ylim(0, boxsize)
plt.savefig(fig_file)

# Analyze
# There should be no variance on the x positions
tol = N_snapshots*100*np.finfo('float32').eps
if np.sum(x0_std_gadget) > tol or np.sum(x1_std_gadget) > tol:
    print('\033[1m\033[91m' + 'Unequal x-positions for the 2*4 particles in the GADGET snapshots.\n'
          + 'It is no good to compare the CONCEPT results to these.' + '\033[0m')
    sys.exit(1)
if np.sum(x0_std) > tol or np.sum(x1_std) > tol:
    print('\033[1m\033[91m' + 'Unequal x-positions for the 2*4 particles in the snapshots.\n'
          + 'The symmetric initial conditions has produced nonsymmetric results!' + '\033[0m')
    sys.exit(1)
# Compare CONCEPT to GADGET
tol = 1e-2
if max(np.abs(x0/x0_gadget - 1)) > tol or max(np.abs(x1/x1_gadget - 1)) > tol:
    masterwarn('The results from CONCEPT disagree with those from GADGET.\n'
          + 'See ' + fig_file + ' for a visualization.')
    sys.exit(1)

