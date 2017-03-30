# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2017 Jeppe Mosgaard Dakin.
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
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Read in data from the CO𝘕CEPT snapshots
for fname in sorted(glob(this_dir + '/output/snapshot_t=*'),
                    key=lambda s: s[(s.index('=') + 1):]):
    snapshot = load(fname, compare_params=False)
    x = snapshot.components[0].posx[0]
    y = snapshot.components[0].posy[0]
    z = snapshot.components[0].posz[0]

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# The particle should have a position of (0.25, 0.25, 0.25)*boxsize
if (   not isclose(x, 0.25*boxsize)
    or not isclose(y, 0.25*boxsize)
    or not isclose(z, 0.25*boxsize)):
    abort('The particle should have drifted to '
          '(0.25, 0.25, 0.25)*boxsize,\n'
          'but the actual position is '
          '({}, {}, {})*boxsize.'.format(x/boxsize,
                                         y/boxsize,
                                         z/boxsize))

# Done analyzing
masterprint('done')

