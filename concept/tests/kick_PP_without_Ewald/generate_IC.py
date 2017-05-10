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
from species import Component
from snapshot import save

# Create the particles.
# It is important that no interparticle separation exceeds boxsize/2 in
# any direction, as the nearest particle image in all cases must be the
# actual particle itself.
N = 8
mass = ϱ_mbar*boxsize**3/N
component = Component('GADGET halos', 'matter particles', N, mass=mass)
component.populate(np.array([0.26]*4 + [0.74]*4)*boxsize, 'posx')
component.populate(np.array([0.25, 0.25, 0.75, 0.75]*2)*boxsize, 'posy')
component.populate(np.array([0.25, 0.75, 0.75, 0.25]*2)*boxsize, 'posz')
component.populate(zeros(N), 'momx')
component.populate(zeros(N), 'momy')
component.populate(zeros(N), 'momz')

# Save snapshot
save(component, initial_conditions)

