# This file is part of CONCEPT, the cosmological N-body code in Python.
# Copyright (C) 2015 Jeppe Mosgard Dakin.
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CONCEPT. If not, see http://www.gnu.org/licenses/
#
# The auther of CONCEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CONCEPT is available at
# https://github.com/jmd-dk/concept/



# This file has to be run in pure Python mode!

# Include the concept_dir in the searched paths and get directory of this file
import sys, os
sys.path.append(os.environ['concept_dir'])
this_dir = os.path.dirname(os.path.realpath(__file__))

# Imports from the CONCEPT code
from commons import *
from IO import load

# Read in the snapshot
particles = load(this_dir + '/snapshot')
N = particles.N
posx = particles.posx
posy = particles.posy
posz = particles.posz

# Volume and linear size of cube the volume of a sphere with radius R_tophat
V = 4*π/3*R_tophat**3
L = V**(1/3)
# The number of complete L*L*L cubes within the box
N_cubes_lin = int(boxsize//L)
N_cubes = N_cubes_lin**3
# The number of particles in each of these cubes, if the snapshot is completely homogeneous
N_in_cubes_homo = N*V/boxsize**3

# Count how many particles lie within each of the L*L*L cubes
counts = np.zeros([N_cubes_lin + 1]*3, dtype=C2np['int'])
posx_grid = posx//L
posy_grid = posy//L
posz_grid = posz//L
for i in range(N):
    counts[posx_grid[i], posy_grid[i], posz_grid[i]] += 1
# The upper "cubes" are not cubic and should be discarded
counts = counts[:-1, :-1, :-1]

# Normalize counts to get the contrast from homogeneity
counts_contrast = counts.flatten()/N_in_cubes_homo - 1

# The RMS (std) of the count contrast is also the RMS of the density contrast
σ = np.std(counts_contrast)

# Is the distribution Gaussian? If not, the snapshot has not been generated correctly.
plt.figure()
plt.hist(counts_contrast, 100)
s = r'{:g}% within $1\sigma$'.format(np.round(100*sum(abs(counts_contrast) < σ)/counts_contrast.size))
plt.text(np.sum(plt.xlim())/2, np.sum(plt.ylim())/2, s, ha='center', bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'})
plt.xlabel('Count contrast')
plt.ylabel('# of occurrences')
plt.savefig(this_dir + '/histogram.png')
if abs(erf(1/sqrt(2)) - sum(abs(counts_contrast) < σ)/counts_contrast.size) > 0.1:
    masterwarn('The particle distribution do not seem to be Gaussian.\n'
               + 'See "{}".'.format(this_dir + '/histogram.png'))
    sys.exit(1)

# Load in σ
powerspec_filename = this_dir + '/powerspec_snapshot'
with open(powerspec_filename, encoding='utf-8') as powespec_file:
    header = powespec_file.readline()
σ_concept = float(re.search('=(.*?) ±', header).group(1))

# Do the σ from CONCEPT agree with the one computed via the cubic boxes?
tol = 1e-2
if abs(1 - σ_concept/σ) > tol:
    masterwarn(('The rms density variation σ = {} from "{}" do not agree with direct computation ({}).\n'
                + 'The power spectrum from which σ is calulated is plotted in "{}"').format(σ_concept, powerspec_filename,
                                                                                            σ,         powerspec_plot_filename))
    sys.exit(1)

















