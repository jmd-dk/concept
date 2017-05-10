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

# Create the fluid.
# This is a global sine wave along the x-direction,
# moving with the speed "speed" along the x-direction,
# the speed 0 along the y-direction and a random speed
# in the z-direction.
components = []
gridsize = 64
speed = a_begin**2*boxsize/(0.5*units.Gyr)
N = gridsize                   # Number of particles
N_fluidelements = gridsize**3  # Number of fluid elements
Vcell = (boxsize/gridsize)**3
mass_tot = ϱ_mbar*boxsize**3
mass_fluid_tot = mass_particles_tot = 0.5*mass_tot
mass_fluid = mass_fluid_tot/N_fluidelements  # Mass of each fluid element
mass_particles = mass_particles_tot/N        # Mass of each particle
component = Component('test fluid', 'matter fluid', gridsize)
ϱ = empty([gridsize]*3)
for i in range(gridsize):
    ϱ[i, :, :] = 2 + np.sin(2*π*i/gridsize)  # Unitless
ϱ /= sum(ϱ)                                  # Normalize
ϱ *= mass_fluid_tot/Vcell                    # Apply units
component.populate(ϱ,                        'ϱ'   )
component.populate(ϱ*speed,                  'J', 0)
component.populate(zeros([gridsize]*3),      'J', 1)
component.populate(ϱ*speed*(random()*2 - 1), 'J', 2)
components.append(component)

# Create the particles.
# These are N particles strewn uniformly over the x-axis, with a velocity
# only in the x-direction. These should then remain equally spaced
# in the x-direction throughout time, each following a specific fluid element.
# The y-positions are given by a sine (in order to match
# these against the fluid density profile) and the z-positions are random.
component = Component('control particles', 'matter particles', N, mass=mass_particles)
offset = 0.5*boxsize
A = 0.4*boxsize
component.populate(linspace(0, boxsize, N, endpoint=False),        'posx')
component.populate(offset + A*np.sin([2*π*i/N for i in range(N)]), 'posy')
component.populate(random(N)*boxsize,                              'posz')
component.populate(ones(N)*speed*mass_particles,                   'momx')
component.populate(zeros(N),                                       'momy')
component.populate(zeros(N),                                       'momz')
components.append(component)

# Save snapshot
save(components, initial_conditions)

