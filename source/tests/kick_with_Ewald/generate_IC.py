# This file has to be run in pure Python mode!

# Include the code directory in the searched paths
import sys, os
Nbody_dir = os.path.realpath(__file__)
this_dir = os.path.dirname(Nbody_dir)
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
N = 8
mass = Ωm*ϱ*boxsize**3/N
particles = construct(this_dir + ' test', 'dark matter', mass, N)
d = -0.005
particles.populate(array([0.25 - d]*4 + [0.75 + d]*4)*boxsize, 'posx')
particles.populate(array([0.25, 0.25, 0.75, 0.75]*2)*boxsize, 'posy')
particles.populate(array([0.25, 0.75, 0.75, 0.25]*2)*boxsize, 'posz')
particles.populate(zeros(N), 'momx')
particles.populate(zeros(N), 'momy')
particles.populate(zeros(N), 'momz')

# Save snapshot
save(particles, a_begin, IC_file)
