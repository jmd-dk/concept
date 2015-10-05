# Include the concept_dir in the searched paths
import sys, os
sys.path.append(os.environ['concept_dir'])
this_dir = os.path.dirname(os.path.realpath(__file__))

# Imports from the CONCEPT code
from commons import *
from species import construct
from IO import save

# Function for generating random numbers from a normal distribution
from numpy.random import normal

# Create close to homogeneous particles
N_lin = 128
N = N_lin**3
mass = Ωm*ϱ*boxsize**3/N
particles = construct('dark matter', 'dark matter', mass, N)
posx = empty(N)
posy = empty(N)
posz = empty(N)
momx = zeros(N)
momy = zeros(N)
momz = zeros(N)
count = 0
boxsize_over_N_lin = boxsize/N_lin
for i in range(N_lin):
    X = i*boxsize_over_N_lin
    for j in range(N_lin):
        Y = j*boxsize_over_N_lin
        for k in range(N_lin):
            Z = k*boxsize_over_N_lin
            posx[count] = normal(loc=X, scale=R_tophat) % boxsize
            posy[count] = normal(loc=Y, scale=R_tophat) % boxsize
            posz[count] = normal(loc=Z, scale=R_tophat) % boxsize
            count += 1
particles.populate(posx, 'posx')
particles.populate(posy, 'posy')
particles.populate(posz, 'posz')
particles.populate(momx, 'momx')
particles.populate(momy, 'momy')
particles.populate(momz, 'momz')

# Save snapshot
save(particles, a_begin, this_dir + '/snapshot')
