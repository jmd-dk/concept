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

# Create close to homogeneous particles
N = 8**3
mass = Ωm*ϱ*boxsize**3/N
mean_sep = boxsize/N**one_third
max_mom = 0.5e+10*units.kpc/units.Gyr*units.m_sun
particles = construct(this_dir + ' test', 'dark matter', mass, N)
posx = zeros(N)
posy = zeros(N)
posz = zeros(N)
momx = zeros(N)
momy = zeros(N)
momz = zeros(N)
count = 0
for i in range(round(N**one_third)):
    for j in range(round(N**one_third)):
        for k in range(round(N**one_third)):
            x = (i/N**one_third*boxsize + (random()*2 - 1)*mean_sep*0.1) % boxsize
            y = (j/N**one_third*boxsize + (random()*2 - 1)*mean_sep*0.1) % boxsize
            z = (k/N**one_third*boxsize + (random()*2 - 1)*mean_sep*0.1) % boxsize
            posx[count] = x
            posy[count] = y
            posz[count] = z
            momx[count] = (random()*2 - 1)*max_mom
            momy[count] = (random()*2 - 1)*max_mom
            momz[count] = (random()*2 - 1)*max_mom
            count += 1
particles.populate(posx, 'posx')
particles.populate(posy, 'posy')
particles.populate(posz, 'posz')
particles.populate(momx, 'momx')
particles.populate(momy, 'momy')
particles.populate(momz, 'momz')

# Save snapshot
save(particles, a_begin, IC_file)
