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

# Create a global sine wave along the x-direction:
# ϱ(x, y, z) = ϱ(x) ∝ (1.05 + sin(2*π*x/boxsize)).
# The function ∫_{x1}^{x2}dxϱ(x)
ᔑdxϱ = lambda x1, x2: 1.05*(x2 - x1) + boxsize/π*(cos(π*x1/boxsize)**2 - cos(π*x2/boxsize)**2)
# Function which finds x2 in ∫_{x1}^{x2}dxϱ(x) == mass_unitless
def binary_search(x, mass_unitless, x_lower=None, x_upper=None):
    # Find ᔑdxϱ(x, x_next) == mass_unitless
    if x_lower is None:
        x_lower = x
    if x_upper is None:
        x_upper = boxsize
    x_next = 0.5*(x_lower + x_upper)
    mass_unitless_test = ᔑdxϱ(x, x_next)
    if isclose(mass_unitless_test, mass_unitless, rel_tol=1e-12):
        return x_next
    elif mass_unitless_test < mass_unitless:
        x_lower = x_next
    elif mass_unitless_test > mass_unitless:
        x_upper = x_next
    return binary_search(x, mass_unitless, x_lower=x_lower, x_upper=x_upper)
# Compute positions of particles
Nx = φ_gridsize*20
Ny = Nz = φ_gridsize
N = Nx*Ny*Nz
mass = ρ_mbar*boxsize**3/N
mass_unitless_tot = ᔑdxϱ(0, boxsize)
mass_unitless = mass_unitless_tot/Nx
posx = zeros(N)
posy = zeros(N)
posz = zeros(N)
momx = -ones(N)*mass*boxsize/(60*units.Gyr)
momy =  ones(N)*mass*boxsize/(60*units.Gyr)*(2*random() - 1)
momz =  ones(N)*mass*boxsize/(60*units.Gyr)*(2*random() - 1)
x = 0
count = 0
for i in range(Nx):          
    for j in range(Ny):
        y = j/Ny*boxsize
        for k in range(Nz):
            z = k/Nz*boxsize
            posx[count] = x
            posy[count] = y
            posz[count] = z
            # Make the momenta diverge from the point of lowest density
            if isclose(x, 0.75*boxsize, abs_tol=boxsize/Nx):
                momx[count] -= 0.01*mass*boxsize/(100*units.Gyr)
            elif abs(x - 0.75*boxsize) < 0.3*boxsize:
                if x <= 0.75*boxsize:
                    fac = -0.5*exp(-0.07*(x - 0.75*boxsize + 0.1*boxsize)**2)
                else:
                    fac = +0.5*exp(-0.07*(x - 0.75*boxsize - 0.1*boxsize)**2)
                momx[count] += fac*mass*boxsize/(100*units.Gyr)
            count += 1
    if i < Nx - 1:
        x = binary_search(x, mass_unitless)

# Instantiate particles
particles = Component('control particles', 'matter particles', N, mass=mass)
particles.populate(posx, 'posx')
particles.populate(posy, 'posy')
particles.populate(posz, 'posz')
particles.populate(momx, 'momx')
particles.populate(momy, 'momy')
particles.populate(momz, 'momz')

# Save snapshot
save(particles, initial_conditions)

