# This file has to be run in pure Python mode!

# Include the code directory in the searched paths
import sys, os
Nbody_dir = os.path.realpath(__file__)
while True:
    if Nbody_dir == '/':
        raise Exception('Cannot find the .paths file!')
    if '.paths' in os.listdir(os.path.dirname(Nbody_dir)):
        break
    Nbody_dir = os.path.dirname(Nbody_dir)
sys.path.append(Nbody_dir)

# Imports from the N-body code
from commons import *
from species import construct
from IO import save

# Create the particles
N = 4
mass = Ωm*ϱ*boxsize**3/N
particles = construct('drift test', 'dark matter', mass, N)
particles.populate(array([0.1]*N)*boxsize, 'posx')
particles.populate(array([0.25, 0.25, 0.75, 0.75])*boxsize, 'posy')
particles.populate(array([0.25, 0.75, 0.75, 0.25])*boxsize, 'posz')
particles.populate(ones(4)*100*units.km/units.s*mass, 'momx')
particles.populate(zeros(N), 'momy')
particles.populate(zeros(N), 'momz')

# Save snapshot
save(particles, a_begin, IC_file)
