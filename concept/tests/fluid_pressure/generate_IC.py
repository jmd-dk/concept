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
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from species import Component
from snapshot import save

# Create stationary, homogeneous matter distribution.
# Perturb this homogeneous distribution with a 
# global, stationary sine wave along the x-direction.
# Make the sound speed of the fluid be such that
# a pressure wave traverses the box in 10 Gyr.
gridsize = 64
cs = boxsize/(10*units.Gyr)
if cs >= light_speed:
    abort('Too large sound speed assigned: cs = {} c'.format(cs/light_speed))
w = cs**2/(light_speed**2 - cs**2)
Vcell = (boxsize/gridsize)**3
speed = boxsize/(10*units.Gyr)
N = gridsize**3
mass = ρmbar*boxsize**3/N
component = Component('test fluid', 'dark matter fluid', gridsize, mass, w=w)
ρ = empty([gridsize]*3)
for i in range(gridsize):
    ρ[i, :, :] = 200 + np.sin(2*π*i/gridsize)  # Unitless
ρ /= sum(ρ)                                    # Normalize
ρ *= ρmbar*gridsize**3                         # Apply units
component.populate(ρ,                   'ρ'    )
component.populate(zeros([gridsize]*3), 'ρu', 0)
component.populate(zeros([gridsize]*3), 'ρu', 1)
component.populate(zeros([gridsize]*3), 'ρu', 2)

# Save snapshot
save([component], IC_file)

