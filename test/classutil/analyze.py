# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *

# Other imports
import scipy.interpolate

# Absolute path and name of this test
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(os.path.dirname(this_dir))

# Check the "full" CLASS HDF5
masterprint(f'Analysing {this_test} data ...')
a_size = 32
k_size = 4
linear_species = ['g', 'ncdm[0]', 'ncdm[1]', 'ncdm[2]', 'metric']
def construct_δρ(f, species):
    a_bg = f['background/a'][:]
    a_pt = f['perturbations/a'][:]
    δρ = zeros((a_size, k_size), dtype=C2np['double'])
    for α in species:
        ρ = f[f'background/rho_{α}'][:]
        spline = scipy.interpolate.CubicSpline(
            np.log(a_bg), np.log(ρ), bc_type='natural',
        )
        δ = f[f'perturbations/delta_{α}'][...]
        for i, (a_i, δ_i) in enumerate(zip(a_pt, δ)):
            ρ_i = np.exp(spline(np.log(a_i)))
            δρ[i] += ρ_i*δ_i
    return δρ[a_pt >= a_begin, :]
with open_hdf5(glob(f'{this_dir}/output_full/*.hdf5')[0], mode='r') as f:
    # Check the h attribute
    h = f['background'].attrs['h']
    h_expected = H0/(100*units.km/(units.s*units.Mpc))
    if not isclose(h, h_expected):
        abort(f'Expected h = {h_expected} but got {h}')
    # Get length unit for later use
    unit_length_full = f['units'].attrs['unit length']
    # Check for default extra background quantities
    for key in ['tau', 'D']:
        if key not in f['background']:
            abort(f'Expected \'{key}\' to be included in the background')
    # Check the size of the perturbations
    a_bg = f['background/a'][:]
    a_pt = f['perturbations/a'][:]
    if a_pt.size != a_size:
        abort(f'Expected the \'a\' array to be of size {a_size}')
    k = f['perturbations/k'][:]
    if k.size != k_size:
        abort(f'Expected the \'k\' array to be of size {k_size}')
    for key in f['perturbations']:
        if not key.startswith('delta_'):
            continue
        if f[f'perturbations/{key}'].shape != (a_size, k_size):
            abort(f'Expected the \'{key}\' array to be of shape ({a_size}, {k_size})')
    # Construct δρ of the linear species
    δρ_full = construct_δρ(f, linear_species)
    # Get σ of ncdm[2] at the highest k mode
    σ_full = f['perturbations/sigma_ncdm[2]'][:, -1]

# Check the untrusted σ ncdm[2] perturbation
with open_hdf5(glob(f'{this_dir}/output_untrustedsigma/*.hdf5')[0], mode='r') as f:
    σ_untrusted = f['perturbations/sigma_ncdm[2]'][:, -1]
σ_tol = 0.1
σ_err = sqrt(mean((σ_full/σ_untrusted - 1)[3:-3]**2))
if (not (σ_err > 0)) or σ_err > σ_tol:
    abort(
        f'The extrapolation of sigma_ncdm[2] appears bad: '
        f'{list(σ_full)} (computed) vs. {list(σ_untrusted)} (extrapolated).'
    )

# Check the "extra" CLASS HDF5
with open_hdf5(glob(f'{this_dir}/output_extra/*.hdf5')[0], mode='r') as f:
    # Check for extra background quantities
    for key in ['comov. dist.', ]:
        if key not in f['background']:
            abort(f'Expected \'{key}\' to be included in the background')
    # Check for extra perturbations
    for key in ['theta_tot', ]:
        if key not in f['perturbations']:
            abort(f'Expected \'{key}\' to be included among the perturbations')
    # Check for processed plots of θ_tot
    if not glob(
        f'{this_dir}/output_extra/class_perturbations_processed/theta_tot/*.png'
    ):
        abort('Expected figures of processed theta_tot')
    # Check for plots of ψ and ϕ (these are produced since they are
    # used for the requested metric).
    for perturbation in ['psi', 'phi']:
        if not len(glob(
            f'{this_dir}/output_extra/class_perturbations/{perturbation}/*.png'
        )) == k_size:
            abort(f'Expected {k_size} figures of {perturbation}')
    # Compare a's and k's with that of the other HDF5
    if not np.all(f['background/a'][:] == a_bg):
        abort('The two HDF5 files have different background tabulations')
    if not np.all(f['perturbations/a'][:] == a_pt):
        abort('The two HDF5 files have different temporal perturbation tabulations')
    if not np.all(f['perturbations/k'][:] == k):
        abort('The two HDF5 files have different spatial perturbation tabulations')
    # Compare δρ of the linear species with that of the other HDF5
    linear_species_combined = '+'.join(linear_species)
    δρ_extra = construct_δρ(f, [linear_species_combined])
    if not np.all(np.isclose(δρ_full, δρ_extra, rtol=1e-3, atol=0)):
        abort(
            f'Disagreement between δρ of {linear_species_combined} '
            f'between the two HDF5 files'
        )

# Check the "reuse" CLASS HDF5
with open_hdf5(glob(f'{this_dir}/output_reuse/*.hdf5')[0], mode='r') as f:
    # Compare times with other HDF5
    if not np.all(f['perturbations/a'][:] == a_pt):
        abort('Times not properly reused')
    # Compare modes with other HDF5
    k_size_reuse = f['perturbations/k'].size
    if k_size_reuse !=  k_size - 2:
        abort(f'Expected {k_size - 2} modes but found {k_size_reuse}')
    unit_length_reuse = f['units'].attrs['unit length']
    if not np.all(np.isclose(
        f['perturbations/k'][:]*1/eval_unit(unit_length_reuse),
        k[1:-1]*1/eval_unit(unit_length_full),
        atol=0,
    )):
        abort('Modes not properly reused')

# Done analysing
masterprint('done')

