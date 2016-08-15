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

# Include the concept_dir in the searched paths
import sys, os
sys.path.append(os.environ['concept_dir'])

# Imports from the CO𝘕CEPT code
from commons import *
from species import Component
from snapshot import save

# Create a global sine wave along the x-direction,
# traversing the box in the x-direction over 10 Gyr.
# The y-velocity is 0 and the z-velocity is random.
gridsize = 64
Vcell = (boxsize/gridsize)**3
speed = boxsize/(10*units.Gyr)
N = gridsize**3
mass = ϱmbar*boxsize**3/N
component = Component('test fluid', 'dark matter fluid', gridsize, mass)
ϱ_noghosts = empty([gridsize]*3)
for i in range(gridsize):
    ϱ_noghosts[i, :, :] = 2 + np.sin(2*π*i/gridsize)  # Unitless
ϱ_noghosts /= sum(ϱ_noghosts)                         # Normalize
ϱ_noghosts *= ϱmbar*gridsize**3                       # Apply units
component.populate(ϱ_noghosts,                                           'ϱ'    )
component.populate(ϱ_noghosts*ones([gridsize]*3)*speed,                  'ϱu', 0)
component.populate(ϱ_noghosts*zeros([gridsize]*3),                       'ϱu', 1)
component.populate(ϱ_noghosts*ones([gridsize]*3)*speed*(random()*2 - 1), 'ϱu', 2)

# Save snapshot
save([component], IC_file)

