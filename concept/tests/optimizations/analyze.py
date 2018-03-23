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
import species

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Absolute and relative tolerence to be used in the
# comparisons between power spectra and 2D renders.
abs_tol = 0
rel_tol = 1e-9

# Compare power spectra
powerspec_with_optimizations_filename = (
    glob(this_dir + '/output_with_optimizations/powerspec*')[0]
)
powerspec_without_optimizations_filename = (
    glob(this_dir + '/output_without_optimizations/powerspec*')[0]
)
powerspec_with_optimizations = np.loadtxt(powerspec_with_optimizations_filename)
powerspec_without_optimizations = np.loadtxt(powerspec_without_optimizations_filename)
if not np.allclose(
    powerspec_with_optimizations,
    powerspec_without_optimizations,
    rel_tol,
    abs_tol,
    equal_nan=True,
    ):
    abort(
        f'The power spectra from the optimized and unoptimized run differ.\n'
        f'See "{powerspec_with_optimizations_filename}" and '
        f'"{powerspec_without_optimizations_filename}".'
    )

# Compare 2D renders
render2D_with_optimizations_filename = (
    glob(this_dir + '/output_with_optimizations/render2D*')[0]
)
render2D_without_optimizations_filename = (
    glob(this_dir + '/output_without_optimizations/render2D*')[0]
)
with open_hdf5(render2D_with_optimizations_filename, mode='r') as hdf5_file:
    render2D_with_optimizations = hdf5_file['data'][...]
with open_hdf5(render2D_without_optimizations_filename, mode='r') as hdf5_file:
    render2D_without_optimizations = hdf5_file['data'][...]
if not np.allclose(
    render2D_with_optimizations,
    render2D_without_optimizations,
    rel_tol,
    abs_tol,
    equal_nan=True,
    ):
    abort(
        f'The 2D renders from the optimized and unoptimized run differ.\n'
        f'See "{render2D_with_optimizations_filename}" and '
        f'"{render2D_without_optimizations_filename}".'
    )

