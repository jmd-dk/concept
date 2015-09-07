# This file is part of CONCEPT, the cosmological N-body code in Python.
# Copyright (C) 2015 Jeppe Mosgard Dakin.
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CONCEPT. If not, see http://www.gnu.org/licenses/
#
# The auther of CONCEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CONCEPT is available at
# https://github.com/jmd-dk/concept/



# This file has to be run in pure Python mode!

# Include the concept_dir in the searched paths
import sys, os
sys.path.append(os.environ['concept_dir'])

# Imports from the CONCEPT code
from commons import *
from species import construct
from IO import save

# Create the particles
# Create close to homogeneous particles
N = 8**3
mass = Ωm*ϱ*boxsize**3/N
mean_sep = boxsize/N**one_third
max_mom = 2*boxsize/(13*units.Gyr)*mass
particles = construct('dark matter', 'dark matter', mass, N)
posx = zeros(N)
posy = zeros(N)
posz = zeros(N)
momx = zeros(N)
momy = zeros(N)
momz = zeros(N)
count = 0
for i in range(round(N**one_third)):
    for j in range(round(N**one_third)):
        for k in range(round(N**one_third)):
            x = (i/N**one_third*boxsize + (random()*2 - 1)*mean_sep*0.1) % boxsize
            y = (j/N**one_third*boxsize + (random()*2 - 1)*mean_sep*0.1) % boxsize
            z = (k/N**one_third*boxsize + (random()*2 - 1)*mean_sep*0.1) % boxsize
            posx[count] = x
            posy[count] = y
            posz[count] = z
            momx[count] = (random()*2 - 1)*max_mom
            momy[count] = (random()*2 - 1)*max_mom
            momz[count] = (random()*2 - 1)*max_mom
            count += 1
particles.populate(posx, 'posx')
particles.populate(posy, 'posy')
particles.populate(posz, 'posz')
particles.populate(momx, 'momx')
particles.populate(momy, 'momy')
particles.populate(momz, 'momz')

# Save snapshot
save(particles, a_begin, IC_file)

