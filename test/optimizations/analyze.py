# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load
import species

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Absolute and relative tolerance to be used in the
# comparisons between power spectra and 2D renders.
abs_tol = 0
rel_tol = 1e-9

# Compare power spectra
powerspec_optimized_filename = (
    glob(this_dir + '/output_optimized/powerspec*')[0]
)
powerspec_unoptimized_filename = (
    glob(this_dir + '/output_unoptimized/powerspec*')[0]
)
powerspec_optimized = np.loadtxt(powerspec_optimized_filename)
powerspec_unoptimized = np.loadtxt(powerspec_unoptimized_filename)
if not np.allclose(
    powerspec_optimized,
    powerspec_unoptimized,
    rel_tol,
    abs_tol,
    equal_nan=True,
    ):
    abort(
        f'The power spectra from the optimized and unoptimized run differ.\n'
        f'See "{powerspec_optimized_filename}" and '
        f'"{powerspec_unoptimized_filename}".'
    )

# Compare 2D renders
render2D_optimized_filename = (
    glob(this_dir + '/output_optimized/render2D*')[0]
)
render2D_unoptimized_filename = (
    glob(this_dir + '/output_unoptimized/render2D*')[0]
)
with open_hdf5(render2D_optimized_filename, mode='r') as hdf5_file:
    render2D_optimized = hdf5_file['data'][...]
with open_hdf5(render2D_unoptimized_filename, mode='r') as hdf5_file:
    render2D_unoptimized = hdf5_file['data'][...]
if not np.allclose(
    render2D_optimized,
    render2D_unoptimized,
    rel_tol,
    abs_tol,
    equal_nan=True,
    ):
    abort(
        f'The 2D renders from the optimized and unoptimized run differ.\n'
        f'See "{render2D_optimized_filename}" and '
        f'"{render2D_unoptimized_filename}".'
    )
