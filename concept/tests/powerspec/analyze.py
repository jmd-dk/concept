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
from snapshot import load

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Read in the particles
component = load(initial_conditions, only_components=True)[0]
N = component.N
posx = component.posx
posy = component.posy
posz = component.posz

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Volume and linear size of cube with the volume of a sphere with radius R_tophat
V = 4*π/3*R_tophat**3
L = cbrt(V)
# The number of complete L*L*L cubes within the box
N_cubes_lin = int(boxsize//L)
N_cubes = N_cubes_lin**3
# The number of particles in each of these cubes, if the snapshot is completely homogeneous
N_in_cubes_homo = N*V/boxsize**3

# Count how many particles lie within each of the L*L*L cubes
counts = np.zeros([N_cubes_lin + 1]*3, dtype=C2np['int'])
posx_grid = asarray(posx//L, dtype=C2np['int'])
posy_grid = asarray(posy//L, dtype=C2np['int'])
posz_grid = asarray(posz//L, dtype=C2np['int'])
for i in range(N):
    counts[posx_grid[i], posy_grid[i], posz_grid[i]] += 1
# The upper "cubes" are not cubic and should be discarded
counts = counts[:-1, :-1, :-1]

# Normalize counts to get the contrast from homogeneity
counts_contrast = counts.flatten()/N_in_cubes_homo - 1

# The rms (std) of the count contrast is also the rms of the density contrast
σ = np.std(counts_contrast)

# Is the distribution Gaussian? If not, the snapshot has not been generated correctly
plt.figure()
plt.hist(counts_contrast, 100)
s = (r'{:g}% within $1\sigma$'
     .format(np.round(100*sum(abs(counts_contrast) < σ)/counts_contrast.size)))
plt.text(np.sum(plt.xlim())/2, np.sum(plt.ylim())/2, s, ha='center',
         bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'})
plt.xlabel('Count contrast')
plt.ylabel('# of occurrences')
plt.savefig(this_dir + '/histogram.png')
if abs(erf(1/sqrt(2)) - sum(abs(counts_contrast) < σ)/counts_contrast.size) > 0.1:
    abort('The particle distribution does not seem to be Gaussian.\n'
          'See "{}".'.format(this_dir + '/histogram.png'))

# Load in σ
powerspec_filename = '{}/{}_{}'.format(this_dir,
                                       output_bases['powerspec'],
                                       os.path.basename(os.path.splitext(initial_conditions)[0]))
with open(powerspec_filename, encoding='utf-8') as powespec_file:
    search = None
    while not search:
        header = powespec_file.readline()
        search = re.search(''.join(['σ',
                                    unicode_subscript(f'{R_tophat/units.Mpc:.2g}'),
                                    ' = ([0-9.]*)',
                                    ]
                                   ), header)
σ_concept = float(search.group(1))

# Do the σ from CO𝘕CEPT agree with the one computed via the cubic boxes?
rel_tol = 1e-2
if not isclose(σ, σ_concept, rel_tol=rel_tol):
    abort('The rms density variation σ = {:.6g} from "{}" do not agree with direct computation '
           '({:.6g}). The power spectrum from which σ is calulated is plotted in "{}"'
           .format(σ_concept, powerspec_filename, σ, powerspec_filename + '.png'))

# Check the scaling of the power spectrum against the boxsize.
# Doubling the boxsize (and expanding the particle configuration with it)
# should result in k being halved and the power being multiplied by 2³.
powerspec_filename_single_boxsize = powerspec_filename
powerspec_filename_double_boxsize = '{}_double_boxsize'.format(powerspec_filename)
(k_single_boxsize,
 modes,
 power_single_boxsize,
 ) = np.loadtxt(powerspec_filename_single_boxsize, unpack=True)
(k_double_boxsize,
 modes,
 power_double_boxsize,
 ) = np.loadtxt(powerspec_filename_double_boxsize, unpack=True)
tol = 1e-4
if not all(np.abs((k_single_boxsize/2 - k_double_boxsize)/k_double_boxsize) < tol):
    abort('Bad scaling of k against the boxsize. '
          'The compared power spectra are plotted in "{}.png" and "{}.png"'
          .format(powerspec_filename_single_boxsize, powerspec_filename_double_boxsize)
          )
if not all(np.abs((power_single_boxsize*2**3 - power_double_boxsize)/power_double_boxsize) < tol):
    abort('Bad scaling of power against the boxsize. '
          'The compared power spectra are plotted in "{}.png" and "{}.png"'
          .format(powerspec_filename_single_boxsize, powerspec_filename_double_boxsize)
          )

# Check the scaling of the power spectrum against the gridsize.
# Halving the gridsize should result in the same min(k), but max(k) should be halved.
# Also, halving the gridsize should not affect the power at a given k.
powerspec_filename_whole_gridsize = powerspec_filename
powerspec_filename_half_gridsize = '{}_half_gridsize'.format(powerspec_filename)
k_whole_gridsize, power_whole_gridsize = k_single_boxsize, power_single_boxsize
(k_half_gridsize,
 modes,
 power_half_gridsize,
 ) = np.loadtxt(powerspec_filename_half_gridsize, unpack=True)
if not k_whole_gridsize[0] == k_half_gridsize[0]:
    abort('The smallest k value should not depend on the gridsize. '
          'The compared power spectra are plotted in "{}.png" and "{}.png"'
          .format(powerspec_filename_whole_gridsize, powerspec_filename_half_gridsize)
          )
# New, trimmed (k, power) of whole_gridsize with only the same k as in half_gridsize
k_whole_gridsize_trimmed = k_half_gridsize
power_whole_gridsize_trimmed = []
for k in k_half_gridsize:
    where = np.argwhere(k == k_whole_gridsize)
    if where.size == 0:
        abort('The k value {} is present in "{}" but not in "{}"'
              .format(k, powerspec_filename_half_gridsize, powerspec_filename_whole_gridsize)
              )
    index = where[0][0]
    power_whole_gridsize_trimmed.append(power_whole_gridsize[index])
power_whole_gridsize_trimmed = asarray(power_whole_gridsize_trimmed)
# Compare the powers(k) below k_max/2 = (k2_max/sqrt(3))/2,
# where the CIC noise should not be significant.
k_max = k_half_gridsize[-1]/sqrt(3)
power_half_gridsize_firstpart = power_half_gridsize[k_half_gridsize < 0.5*k_max]
power_whole_gridsize_trimmed_firstpart = power_whole_gridsize_trimmed[k_whole_gridsize_trimmed
                                                                      < 0.5*k_max]
if not np.all(abs((  power_whole_gridsize_trimmed_firstpart
                   - power_half_gridsize_firstpart
                   )/power_half_gridsize_firstpart
                  ) < rel_tol
              ):
    abort('Bad scaling of power against the gridsize. '
          'The compared power spectra are plotted in "{}.png" and "{}.png"'
          .format(powerspec_filename_whole_gridsize, powerspec_filename_half_gridsize)
          )

# Done analyzing
masterprint('done')

