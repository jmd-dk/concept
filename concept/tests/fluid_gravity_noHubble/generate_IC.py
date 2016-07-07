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

# Create a global, stationary sine wave along the x-direction
for sine_resolution in range(1, 2*φ_gridsize):
    N_sine = [round(sine_resolution*(2 + np.sin(2*π*i/φ_gridsize)))
              for i in range(φ_gridsize)]
    if N_sine.count(max(N_sine)) == 1:
        N_sine = asarray(N_sine, dtype=C2np['int'])
        break
Nx = np.sum(N_sine)
Ny = Nz = φ_gridsize
N = Nx*Ny*Nz
mass = Ωm*ϱ*boxsize**3/N
particles = Component('control particles', 'dark matter particles', N, mass)
posx = zeros(N)
posy = zeros(N)
posz = zeros(N)
momx = zeros(N)
momy = zeros(N)
momz = zeros(N)
count = 0
for i in range(φ_gridsize):
    x = i/φ_gridsize*boxsize
    for n in range(N_sine[i]):
        for j in range(Ny):
            y = j/Ny*boxsize
            for k in range(Nz):
                z = k/Nz*boxsize
                posx[count] = x
                posy[count] = y
                posz[count] = z
                count += 1
        # Increase x slightly, filling the entire x-length of the fluid element with particles.
        # Each fluid element is then represented by a line of particles.
        x += boxsize/φ_gridsize/N_sine[i]
# Shift all x-positions so that the particle lines corresponding to fluid elements are centeres
# around the fluid elements.
posx = np.mod(posx + 0.5*boxsize/φ_gridsize, boxsize)
particles.populate(posx, 'posx')
particles.populate(posy, 'posy')
particles.populate(posz, 'posz')
particles.populate(momx, 'momx')
particles.populate(momy, 'momy')
particles.populate(momz, 'momz')

# Save snapshot
save([particles], IC_file)

