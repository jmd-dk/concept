# This file has to be run in pure Python mode!

# Include the code directory in the searched paths
import sys, os
concept_dir = os.path.realpath(__file__)
this_dir = os.path.dirname(concept_dir)
while True:
    if concept_dir == '/':
        raise Exception('Cannot find the .paths file!')
    if '.paths' in os.listdir(os.path.dirname(concept_dir)):
        break
    concept_dir = os.path.dirname(concept_dir)
sys.path.append(concept_dir)

# Imports from the CONCEPT code
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
particles.populate(ones(N)*100*units.km/units.s*mass, 'momx')
particles.populate(zeros(N), 'momy')
particles.populate(zeros(N), 'momz')

# Save snapshot
save(particles, a_begin, IC_file)
