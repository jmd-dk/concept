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
from species import construct_particles
from snapshot import save

# Create the particles
N = 4
mass = Ωm*ϱ*boxsize**3/N
particles = construct_particles('GADGET halos', 'dark matter', mass, N)
particles.populate(np.array([0.1]*N)*boxsize, 'posx')
particles.populate(np.array([0.25, 0.25, 0.75, 0.75])*boxsize, 'posy')
particles.populate(np.array([0.25, 0.75, 0.75, 0.25])*boxsize, 'posz')
particles.populate(ones(N)*100*units.km/units.s*mass, 'momx')
particles.populate(zeros(N), 'momy')
particles.populate(zeros(N), 'momz')

# Save snapshot
save([particles], a_begin, IC_file)

