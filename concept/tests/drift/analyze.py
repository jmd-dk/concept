# This file is part of CONCEPT, the cosmological N-body code in Python.
# Copyright (C) 2015 Jeppe Mosgard Dakin.
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CONCEPT. If not, see http://www.gnu.org/licenses/
#
# The auther of CONCEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CONCEPT is available at
# https://github.com/jmd-dk/concept/



# This file has to be run in pure Python mode!

# Include the concept_dir in the searched paths and get directory of this file
import sys, os
sys.path.append(os.environ['concept_dir'])
this_dir = os.path.dirname(os.path.realpath(__file__))

# Imports from the CONCEPT code
from commons import *
from IO import GadgetSnapshot

# Use a matplotlib backend that does not require a running X-server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Determine the number of snapshots from the outputlist file
N_snapshots = np.loadtxt(this_dir + '/outputlist').size

# Instantiate a GadgetSnapshot instance which will be reused for all GADGET snapshots
snapshot = GadgetSnapshot()

# Read in data from the CONCEPT snapshots
a = zeros(N_snapshots)
x = zeros(N_snapshots)
x_std = zeros(N_snapshots)
for i in range(N_snapshots):
    fname = 'snapshot_a={:.2f}'.format(np.loadtxt(this_dir + '/outputlist')[i])
    snapshot.load(this_dir + '/output/' + fname, write_msg=False)
    a[i] = snapshot.header['Time']
    x[i] = np.mean(snapshot.particles.posx)
    x_std[i] = np.std(snapshot.particles.posx)

# Read in data from the GADGET snapshots
a_gadget = zeros(N_snapshots)
x_gadget = zeros(N_snapshots)
x_std_gadget = zeros(N_snapshots)
for i in range(N_snapshots):
    snapshot.load(this_dir + '/output/snapshot_gadget_' + '0'*(3-len(str(i))) + str(i), write_msg=False)
    a_gadget[i] = snapshot.header['Time']
    x_gadget[i] = np.mean(snapshot.particles.posx)
    x_std_gadget[i] = np.std(snapshot.particles.posx)

# Plot
fig_file = this_dir + '/result.pdf'
plt.text(0.5*max(a), 0.93*boxsize, r'$\uparrow$ End of simulation box $\uparrow$', ha='center')
plt.errorbar(a, x/units.kpc, yerr=x_std/units.kpc, fmt='-sr', label='CO$N$CEPT')
plt.errorbar(a_gadget, x_gadget/units.kpc, yerr=x_std_gadget/units.kpc, fmt='--*b', label='GADGET')
plt.legend(loc='best')
plt.xlabel('$a$')
plt.ylabel(r'$x\,\mathrm{[kpc]}$')
plt.ylim(0, boxsize)
plt.savefig(fig_file)

# Analyze.
# There should be no variance on the x positions.
tol = N_snapshots*100*np.finfo('float32').eps
if np.sum(x_std_gadget) > tol:
    masterwarn('Unequal x-positions for the 4 particles in the GADGET snapshots.\n'
               + 'It is no good to compare the CONCEPT results to these.')
    sys.exit(1)
if np.sum(x_std) > tol:
    masterwarn('Unequal x-positions for the 4 particles in the snapshots.\n'
               + 'The symmetric initial conditions has produced nonsymmetric results!')
    sys.exit(1)

# Printout error message for unsuccessful test
tol = 1e-3
if max(np.abs(x/x_gadget - 1)) > tol:
    masterwarn('The results from CONCEPT disagree with those from GADGET.\n'
               + 'See "{}" for a visualization.'.format(fig_file))
    sys.exit(1)

