# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load

# Absolute path and name of this test
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(os.path.dirname(this_dir))

# Read in data from the CO𝘕CEPT snapshots
for fname in sorted(
    glob(f'{this_dir}/output/snapshot_t=*'),
    key=(lambda s: s[(s.index('=') + 1):]),
):
    snapshot = load(fname, compare_params=False)
    x = snapshot.components[0].posx[0]
    y = snapshot.components[0].posy[0]
    z = snapshot.components[0].posz[0]

# Begin analysis
masterprint(f'Analysing {this_test} data ...')

# The particle should have a position of (0.25, 0.25, 0.25)*boxsize
if (   not isclose(x, 0.25*boxsize)
    or not isclose(y, 0.25*boxsize)
    or not isclose(z, 0.25*boxsize)
):
    abort(
        f'The particle should have drifted to '
        f'(0.25, 0.25, 0.25)*boxsize,\n'
        f'but the actual position is '
        f'({x/boxsize}, {y/boxsize}, {z/boxsize})*boxsize.'
    )

# Done analysing
masterprint('done')
