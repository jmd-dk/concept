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

# Create close to homogeneous particles
N = 8**3
mass = Ωm*ϱ*boxsize**3/N
mean_sep = boxsize/N**(1/3)
max_mom = 0.5e+10*units.kpc/units.Gyr*units.m_sun
component = Component('GADGET halos', 'dark matter particles', N, mass)
posx = zeros(N)
posy = zeros(N)
posz = zeros(N)
momx = zeros(N)
momy = zeros(N)
momz = zeros(N)
count = 0
for i in range(int(round(N**(1/3)))):
    for j in range(int(round(N**(1/3)))):
        for k in range(int(round(N**(1/3)))):
            x = (i/N**(1/3)*boxsize + (random()*2 - 1)*mean_sep*0.1) % boxsize
            y = (j/N**(1/3)*boxsize + (random()*2 - 1)*mean_sep*0.1) % boxsize
            z = (k/N**(1/3)*boxsize + (random()*2 - 1)*mean_sep*0.1) % boxsize
            posx[count] = x
            posy[count] = y
            posz[count] = z
            momx[count] = (random()*2 - 1)*max_mom
            momy[count] = (random()*2 - 1)*max_mom
            momz[count] = (random()*2 - 1)*max_mom
            count += 1
component.populate(posx, 'posx')
component.populate(posy, 'posy')
component.populate(posz, 'posz')
component.populate(momx, 'momx')
component.populate(momy, 'momy')
component.populate(momz, 'momz')

# Save snapshot
save([component], IC_file)

