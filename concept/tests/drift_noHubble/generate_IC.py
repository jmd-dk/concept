# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2018 Jeppe Mosgaard Dakin.
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

# Create the particle
N = 1
mass = ρ_mbar*boxsize**3/N
particles = Component('test particles', 'matter particles', N, mass=mass)
particles.populate(np.array([0.75])*boxsize, 'posx')
particles.populate(np.array([0.75])*boxsize, 'posy')
particles.populate(np.array([0.75])*boxsize, 'posz')
particles.populate(ones(N)*boxsize/(10*units.Gyr)*mass, 'momx')
particles.populate(ones(N)*boxsize/(10*units.Gyr)*mass, 'momy')
particles.populate(ones(N)*boxsize/(10*units.Gyr)*mass, 'momz')

# Save snapshot
save(particles, initial_conditions)
