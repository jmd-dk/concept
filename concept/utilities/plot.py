# This file has to be run in pure Python mode!
# This file will read in the snapshots produced by the chosen parameterfile,
# plot the particles and save the plot in the framefolder.

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

# Other imports
from os.path import isfile

# Plot any generated snapshots
any_snapshots = False
for i, t in enumerate(outputtimes):
    snapshot_filename = output_dir + '/' + snapshot_base + '_' + str(i)
    if isfile(snapshot_filename):
        any_snapshots = True
        particles = load(snapshot_filename)
        animate(particles, i, t, t)
if not any_snapshots:
    warn('No snapshot files were found and no plots were made.')
