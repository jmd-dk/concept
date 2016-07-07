# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2016 Jeppe Mosgaard Dakin.
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
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# This file has to be run in pure Python mode!

# Standard test imports
import glob, sys, os

# Absolute paths to the directory of this file
this_dir = os.path.dirname(os.path.realpath(__file__))

# Pull in environment variables
for env_var in ('concept_dir', 'this_test'):
    exec('{env_var} = os.environ["{env_var}"]'.format(env_var=env_var))

# Include the concept_dir in the searched paths
sys.path.append(concept_dir)

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load

# Read in the particles
component = load(IC_file, only_components=True)[0]
N = component.N
posx = component.posx
posy = component.posy
posz = component.posz

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Volume and linear size of cube with the volume of a sphere with radius R_tophat
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

# Is the distribution Gaussian? If not, the snapshot has not been generated correctly
plt.figure()
plt.hist(counts_contrast, 100)
s = (r'{:g}% within $1\sigma$'
     .format(np.round(100*sum(abs(counts_contrast) < σ)/counts_contrast.size)))
plt.text(np.sum(plt.xlim())/2, np.sum(plt.ylim())/2, s, ha='center',
         bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'})
plt.xlabel('Count contrast')
plt.ylabel('\# of occurrences')
plt.savefig(this_dir + '/histogram.png')
if abs(erf(1/sqrt(2)) - sum(abs(counts_contrast) < σ)/counts_contrast.size) > 0.1:
    masterprint('done')
    masterwarn('The particle distribution do not seem to be Gaussian.\n'
               'See "{}".'.format(this_dir + '/histogram.png'))
    sys.exit(1)

# Load in σ
powerspec_filename = this_dir + '/powerspec_snapshot'
with open(powerspec_filename, encoding='utf-8') as powespec_file:
    search = None
    while not search:
        header = powespec_file.readline()
        search = re.search('=(.*?) ±', header)
σ_concept = float(search.group(1))

# Done analyzing
masterprint('done')

# Do the σ from CO𝘕CEPT agree with the one computed via the cubic boxes?
tol = 1e-2
if abs(1 - σ_concept/σ) > tol:
    masterwarn(('The rms density variation σ = {:.6g} from "{}" do not agree with direct '
                'computation ({:.6g}). The power spectrum from which σ is calulated is plotted '
                'in "{}"').format(σ_concept, powerspec_filename, σ, powerspec_filename + '.png'))
    sys.exit(1)

