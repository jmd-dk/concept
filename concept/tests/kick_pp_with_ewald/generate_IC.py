# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from species import Component
from snapshot import save

# Create the particles
N = 8
mass = ρ_mbar*boxsize**3/N
components = Component('GADGET halos', 'matter particles', N, mass=mass)
d = 0.005
components.populate(np.array([0.25 - d]*4 + [0.75 + d]*4)*boxsize, 'posx')
components.populate(np.array([0.25, 0.25, 0.75, 0.75]*2) *boxsize, 'posy')
components.populate(np.array([0.25, 0.75, 0.75, 0.25]*2) *boxsize, 'posz')
components.populate(zeros(N), 'momx')
components.populate(zeros(N), 'momy')
components.populate(zeros(N), 'momz')

# Save snapshot
save(components, initial_conditions)
