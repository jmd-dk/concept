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

# Create stationary, homogeneous matter distribution,
# perturbed with global, stationary sine wave along
# the x-direction.
ρ0 = user_params['_ρ0']
A  = user_params['_A']
σ  = user_params['_σ']
gridsize = 4*16  # Should be a multiple of 4
N = gridsize**3
component = Component('test fluid', 'matter fluid', gridsize, N_fluidvars=3)
ρ = empty([gridsize]*3)
for i in range(gridsize):
    x = boxsize*i/gridsize
    ρ[i, :, :] = ρ0 + A*sin(x/boxsize*2*π)
component.populate(ρ, 'ϱ')
for multi_index in component.J.multi_indices:
    component.populate(zeros([gridsize]*3), 'J', multi_index)
for multi_index in component.σ.multi_indices:
    component.populate(ones([gridsize]*3)*σ, 'σ', multi_index)

# Save snapshot
save(component, initial_conditions)

