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



# Include the concept_dir in the searched paths
import sys, os
sys.path.append(os.environ['concept_dir'])

# Imports from the CO𝘕CEPT code
from commons import *
from species import construct_particles
from snapshot import save

# Function for generating random numbers from a normal distribution
from numpy.random import normal

# Create close to homogeneous particles
N_lin = 128
N = N_lin**3
mass = Ωm*ϱ*boxsize**3/N
component = construct_particles('dark matter particles', 'dark matter', N, mass)
posx = empty(N)
posy = empty(N)
posz = empty(N)
momx = zeros(N)
momy = zeros(N)
momz = zeros(N)
count = 0
boxsize_over_N_lin = boxsize/N_lin
for i in range(N_lin):
    X = i*boxsize_over_N_lin
    for j in range(N_lin):
        Y = j*boxsize_over_N_lin
        for k in range(N_lin):
            Z = k*boxsize_over_N_lin
            posx[count] = normal(loc=X, scale=R_tophat) % boxsize
            posy[count] = normal(loc=Y, scale=R_tophat) % boxsize
            posz[count] = normal(loc=Z, scale=R_tophat) % boxsize
            count += 1
component.populate(posx, 'posx')
component.populate(posy, 'posy')
component.populate(posz, 'posz')
component.populate(momx, 'momx')
component.populate(momy, 'momy')
component.populate(momz, 'momz')

# Save snapshot
save([component], a_begin, IC_file)

