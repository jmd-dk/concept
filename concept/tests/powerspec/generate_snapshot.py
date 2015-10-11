# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: you can redistribute it and/or modify
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
# The auther of CO𝘕CEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Include the concept_dir in the searched paths
import sys, os
sys.path.append(os.environ['concept_dir'])
this_dir = os.path.dirname(os.path.realpath(__file__))

# Imports from the CO𝘕CEPT code
from commons import *
from species import construct
from IO import save

# Function for generating random numbers from a normal distribution
from numpy.random import normal

# Create close to homogeneous particles
N_lin = 128
N = N_lin**3
mass = Ωm*ϱ*boxsize**3/N
particles = construct('dark matter particles', 'dark matter', mass, N)
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
particles.populate(posx, 'posx')
particles.populate(posy, 'posy')
particles.populate(posz, 'posz')
particles.populate(momx, 'momx')
particles.populate(momy, 'momy')
particles.populate(momz, 'momz')

# Save snapshot
save(particles, a_begin, this_dir + '/snapshot')
