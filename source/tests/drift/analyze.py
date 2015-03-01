# This file has to be run in pure Python mode!

# Include the actual code directory in the searched paths
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))[:-1]))

# Imports
from commons import *
from IO import Gadget_snapshot

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
    snapshot.load('tests/drift/output/snapshot_gadget_' + str(i))
    a_gadget[i] = snapshot.header['Time']
    x_gadget[i] = np.mean(snapshot.particles.posx)
    x_std_gadget[i] = np.std(snapshot.particles.posx)
