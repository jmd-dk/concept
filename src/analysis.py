# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2024 Jeppe Mosgaard Dakin.
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
# along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport(
    'from communication import '
    '    communicate_ghosts,   '
    '    get_buffer,           '
)
cimport('from graphics import get_output_declarations')
cimport('from ic import realize')
cimport(
    'from linear import        '
    '    compute_cosmo,        '
    '    get_linear_powerspec, '
    '    get_treelevel_bispec, '
)
cimport(
    'from mesh import          '
    '    diff_domaingrid,      '
    '    domain_decompose,     '
    '    domain_loop,          '
    '    fft,                  '
    '    fourier_loop,         '
    '    fourier_shell_loop,   '
    '    get_fftw_slab,        '
    '    interpolate_upstream, '
    '    nullify_modes,        '
)

# Pure Python imports
from graphics import plot_bispec, plot_powerspec
from linear import get_linear_component



# Top-level function for computing, plotting and saving power spectra
@cython.pheader(
    # Arguments
    components=list,
    filename=str,
    # Locals
    declaration=object,  # PowerspecDeclaration
    declarations=list,
    returns='void',
)
def powerspec(components, filename):
    # Get power spectrum declarations
    declarations = get_powerspec_declarations(components)
    # Compute power spectrum for each declaration
    for declaration in declarations:
        # Compute the power spectrum of the non-linearly evolved
        # components in this power spectrum declaration.
        # The result is stored in declaration.power.
        # Only the master process holds the full power spectrum.
        compute_powerspec(declaration)
        # If specified, also compute the linear power spectrum.
        # The result is stored in declaration.power_linear.
        # Only the master process holds the linear power spectrum.
        # Note that we should compute the linear power spectrum before
        # the corrected power spectrum.
        compute_powerspec_linear(declaration)
        # If specified, also compute the corrected power spectrum.
        # The result is stored in declaration.power_corrected. Only the
        # master process holds the full corrected power spectrum.
        compute_powerspec_corrected(declaration)
    # Dump power spectra to collective data file
    save_powerspec(declarations, filename)
    # Dump power spectra to individual image files
    plot_powerspec(declarations, filename)

# Function for getting declarations for all needed power spectra,
# given a list of components.
@cython.header(
    # Arguments
    components=list,
    # Locals
    cache_key=tuple,
    components_str=str,
    declaration=object,  # PowerspecDeclaration
    declarations=list,
    do_attr=str,
    do_data='bint',
    i='Py_ssize_t',
    k2_max='Py_ssize_t',
    k_bin_centers='double[::1]',
    k_bin_indices='Py_ssize_t[::1]',
    n_modes='Py_ssize_t[::1]',
    power='double[::1]',
    power_corrected='double[::1]',
    power_linear='double[::1]',
    size='Py_ssize_t',
    returns=list,
)
def get_powerspec_declarations(components):
    # Look up declarations in cache
    cache_key = tuple(components)
    declarations = powerspec_declarations_cache.get(cache_key)
    if declarations:
        return declarations
    # Get declarations with basic fields populated
    declarations = get_output_declarations(
        'powerspec',
        components,
        powerspec_select,
        powerspec_options,
        PowerspecDeclaration,
    )
    # Add missing declaration fields
    for i, declaration in enumerate(declarations):
        # Enable do_data if any of the other "do attributes" are enabled
        do_data = declaration.do_data
        if not do_data:
            for do_attr in ['corrected', 'linear', 'plot']:
                if not getattr(declaration, f'do_{do_attr}'):
                    continue
                components_str = ', '.join([
                    component.name for component in declaration.components
                ])
                if len(declaration.components) > 1:
                    components_str = f'{{{components_str}}}'
                masterprint(
                    f'Enabling \'data\' for power spectra of {components_str} '
                    f'because \'{do_attr}\' is enabled'
                )
                do_data = True
                break
        # Get bin information
        k2_max, k_bin_indices, k_bin_centers, n_modes = get_powerspec_bins(
            declaration.gridsize,
            declaration.k_max,
            declaration.bins_per_decade,
        )
        # Allocate arrays for storing the power
        size = bcast(k_bin_centers.shape[0] if master else None)
        power = empty(size, dtype=C2np['double'])
        power_corrected = (
            empty(size, dtype=C2np['double'])
            if declaration.do_corrected
            else None
        )
        power_linear = (
            empty(size, dtype=C2np['double'])
            if master and declaration.do_linear
            else None
        )
        # Replace old declaration with a new, fully populated one
        declaration = declaration._replace(
            do_data=do_data,
            k2_max=k2_max,
            k_bin_indices=k_bin_indices,
            k_bin_centers=k_bin_centers,
            n_modes=n_modes,
            power=power,
            power_corrected=power_corrected,
            power_linear=power_linear,
        )
        declarations[i] = declaration
    # Store declarations in cache and return
    powerspec_declarations_cache[cache_key] = declarations
    return declarations
# Cache used by the get_powerspec_declarations() function
cython.declare(powerspec_declarations_cache=dict)
powerspec_declarations_cache = {}
# Create the PowerspecDeclaration type
fields = (
    'components', 'do_data', 'do_corrected', 'do_linear', 'do_plot', 'gridsize',
    'interpolation', 'deconvolve', 'interlace', 'realization_correction',
    'k2_max', 'k_max', 'bins_per_decade', 'tophat', 'significant_figures',
    'k_bin_indices', 'k_bin_centers', 'n_modes', 'power', 'power_corrected', 'power_linear',
)
PowerspecDeclaration = collections.namedtuple(
    'PowerspecDeclaration', fields, defaults=[None]*len(fields),
)

# Function for constructing arrays k_bin_indices, k_bin_centers and
# n_modes, describing the binning of power spectra.
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    k_max=object,  # double or str
    bins_per_decade=dict,
    # Locals
    cache_key=tuple,
    dist_left='double',
    dist_right='double',
    factor='double',
    index='Py_ssize_t',
    k2='Py_ssize_t',
    k2_max='Py_ssize_t',
    k_bin_center='double',
    k_bin_centers='double[::1]',
    k_bin_index='Py_ssize_t',
    k_bin_index_prev='Py_ssize_t',
    k_bin_indices='Py_ssize_t[::1]',
    k_fundamental='double',
    k_magnitude='double',
    k_min='double',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    logk_bin_centers='double[::1]',
    logk_magnitude='double',
    mask=object,  # bool np.ndarray
    n_modes='Py_ssize_t[::1]',
    n_modes_fine='Py_ssize_t[::1]',
    nyquist='Py_ssize_t',
    powerspec_bins=tuple,
    θ='double',
    returns=tuple,
)
def get_powerspec_bins(gridsize, k_max, bins_per_decade):
    """The returned objects are:
    - k2_max: Maximum value of k² (grid units).
    - k_bin_indices: Array mapping k² (grid units) to bin index, i.e.
        k_bin_index = k_bin_indices[k2]
      All processes will have a copy of this array.
    - k_bin_centers: Array mapping bin index to |k⃗|, i.e.
        k_bin_center = k_bin_centers[k_bin_index]
      This array lives on the master process only.
    - n_modes: Array mapping bin index to number of modes, i.e.
        n = n_modes[bin_index]
      This array lives on the master process only.
    """
    # Look up in the cache
    cache_key = (gridsize, k_max, tuple(bins_per_decade.items()))
    powerspec_bins = powerspec_bins_cache.get(cache_key)
    if powerspec_bins:
        return powerspec_bins
    # Minimum value of k (physical)
    k_fundamental = ℝ[2*π/boxsize]
    k_min = k_fundamental
    # Maximum value of k (physical) and k² (grid units)
    nyquist = gridsize//2
    if isinstance(k_max, str):
        k_max = eval_bin_str(
            k_max,
            {
                'nyquist'      : k_fundamental*nyquist,
                'gridsize'     : gridsize,
                'k_min'        : k_min,
                'k_fundamental': k_min,
                'k_f'          : k_min,
            },
        )
    if k_max < k_min:
        masterwarn(
            f'Power spectrum k_max was set to {k_max} {unit_length}⁻¹ '
            f'< k_min = 2π/boxsize = {k_min} {unit_length}⁻¹. '
            f'Setting k_max = k_min.'
        )
        k_max = k_min
    k2_max = int(round((k_max/k_fundamental)**2))
    # No need for k2_max to be larger than the largest possible mode
    k2_max = pairmin(k2_max, 3*nyquist**2)
    # Correct k_max
    k_max = k_fundamental*sqrt(k2_max)
    # Get k of bin centres using a running bin size
    # as specified by bins_per_decade.
    k_bin_centers = construct_powerspec_k_bin_centers(
        k_min, k_max, bins_per_decade, gridsize, nyquist,
    )
    # Construct array mapping k2 (grid units) to bin index
    logk_bin_centers = np.log(k_bin_centers)
    k_bin_indices = empty(1 + k2_max, dtype=C2np['Py_ssize_t'])
    k_bin_indices[0] = 0
    for k2 in range(1, k_bin_indices.shape[0]):
        k_magnitude = k_fundamental*sqrt(k2)
        logk_magnitude = log(k_magnitude)
        # Find index of closest (in log distance) bin centre
        index = np.searchsorted(logk_bin_centers, logk_magnitude)
        if index == ℤ[k_bin_centers.shape[0]]:
            index -= 1
        elif index != 0:
            dist_left  = logk_magnitude - logk_bin_centers[index - 1]
            dist_right = logk_bin_centers[index] - logk_magnitude
            index -= (dist_left <= dist_right)
        k_bin_indices[k2] = index
    # Loop as if over 3D Fourier slabs, tallying up the multiplicity
    # (number of modes) for each k².
    n_modes_fine = zeros(k_bin_indices.shape[0], dtype=C2np['Py_ssize_t'])
    for index, ki, kj, kk, factor, θ in fourier_loop(
        gridsize,
        sparse=True,
        skip_origin=True,
        k2_max=k2_max,
    ):
        k2 = ℤ[ℤ[ℤ[kj**2] + ki**2] + kk**2]
        n_modes_fine[k2] += 1
    # Sum n_modes_fine into the master process
    Reduce(
        sendbuf=(MPI.IN_PLACE if master else n_modes_fine),
        recvbuf=(n_modes_fine if master else None),
        op=MPI.SUM,
    )
    # The master process now holds all the information needed
    if not master:
        # The slave processes return now.
        # Updated values of k_bin_indices are received from the master.
        # This is the only data known to the slaves.
        Bcast(k_bin_indices)
        k_bin_centers = n_modes = None
        powerspec_bins_cache[cache_key] = (
            k2_max,
            k_bin_indices,
            k_bin_centers,
            n_modes,
        )
        return powerspec_bins_cache[cache_key]
    # Redefine k_bin_centers so that each element is the (geometric)
    # mean of all the k values that falls within the bin, using the
    # multiplicity (n_modes_fine) as weight. Simultaneously construct
    # n_modes from n_modes_fine, where n_modes is just like
    # n_modes_fine, but counting the multiplicity of the bins,
    # rather than the individual k² elements.
    k_bin_centers[:] = 0
    n_modes = zeros(k_bin_centers.shape[0], dtype=C2np['Py_ssize_t'])
    for k2 in range(1, n_modes_fine.shape[0]):
        if ℤ[n_modes_fine[k2]] == 0:
            continue
        k_magnitude = k_fundamental*sqrt(k2)
        k_bin_index = k_bin_indices[k2]
        n_modes[k_bin_index] += ℤ[n_modes_fine[k2]]
        k_bin_centers[k_bin_index] += ℤ[n_modes_fine[k2]]*log(k_magnitude)
    for k_bin_index in range(k_bin_centers.shape[0]):
        if ℤ[n_modes[k_bin_index]] > 0:
            k_bin_centers[k_bin_index] = exp(k_bin_centers[k_bin_index]/ℤ[n_modes[k_bin_index]])
    # We wish to remove bins with a mode count of 0.
    # Modify k_bin_indices so that consecutive bin indices
    # correspond to non-empty bins.
    k_bin_index_prev = k_bin_indices[0]
    for k2 in range(1, k_bin_indices.shape[0]):
        k_bin_index = k_bin_indices[k2]
        if k_bin_index == k_bin_index_prev or n_modes[k_bin_index] == 0:
            k_bin_indices[k2] = k_bin_indices[k2 - 1]
        elif k_bin_index > k_bin_index_prev:
            k_bin_indices[k2] = k_bin_indices[k2 - 1] + 1
            k_bin_index_prev = k_bin_index
    # The final values of k_bin_indices should be known to all processes
    Bcast(k_bin_indices)
    # Remove bins with mode count 0
    mask = (asarray(n_modes) > 0)
    n_modes = asarray(n_modes)[mask]
    k_bin_centers = asarray(k_bin_centers)[mask]
    # Cache and return result
    powerspec_bins = (
        k2_max,
        k_bin_indices,
        k_bin_centers,
        n_modes,
    )
    powerspec_bins_cache[cache_key] = powerspec_bins
    return powerspec_bins
# Cache used by the get_powerspec_bins() function
cython.declare(powerspec_bins_cache=dict)
powerspec_bins_cache = {}

# Helper function for get_powerspec_bins()
def construct_powerspec_k_bin_centers(k_min, k_max, bins_per_decade, gridsize, nyquist):
    # A bin size below binsize_min is guaranteed to never bin
    # separate k² together in the same bin, and so binsize_min is the
    # smallest bin size allowed.
    k_fundamental = ℝ[2*π/boxsize]
    binsize_min = (
        0.5*(1 - 1e-2)*k_fundamental
        *(sqrt(3*nyquist**2 + 1) - sqrt(3*nyquist**2))
    )
    # Evaluate str keys in the bins_per_decade dict
    bins_per_decade_float = {}
    mapping_str2float = {
        'nyquist'      : k_fundamental*nyquist,
        'gridsize'     : gridsize,
        'k_min'        : k_min,
        'k_max'        : k_max,
        'k_fundamental': k_min,
        'k_f'          : k_min,
    }
    for k, val in bins_per_decade.items():
        if isinstance(k, str):
            k = eval_bin_str(k, mapping_str2float)
        if isinstance(val, str):
            val = eval_bin_str(val, mapping_str2float)
        bins_per_decade_float[k] = val
    bins_per_decade = bins_per_decade_float
    if len(bins_per_decade) == 1:
        bins_per_decade.update({k + 1: val for k, val in bins_per_decade.items()})
    # Construct k_bin_centers using a running number of bins per decade
    logk_min = log10(k_min)
    logk_max = log10(k_max)
    k_bin_centers = []
    logk_bins_per_decade_interp = get_controlpoint_spline(bins_per_decade, np.log10)
    logk_bin_right = logk_min - 0.5/logk_bins_per_decade_interp(logk_min)
    while logk_bin_right <= logk_max:
        logk_bin_left = logk_bin_right
        logk_bin_right = logk_bin_left + 1/logk_bins_per_decade_interp(logk_bin_left)
        logk_bin_right = np.max((logk_bin_right, log10(10**logk_bin_left + binsize_min)))
        k_bin_center = 10**(0.5*(logk_bin_left + logk_bin_right))
        k_bin_centers.append(k_bin_center)
    if not k_bin_centers:
        k_bin_centers.append(sqrt(k_min*k_max))
    k_bin_centers = asarray(k_bin_centers, dtype=C2np['double'])
    # Stretch and shift the array so that the end points match their
    # appropriate values exactly. Note that while the rightmost bin only
    # just includes k_max, half of the leftmost bin is less than k_min,
    # disfavouring binning of different modes into the very first bin.
    if len(k_bin_centers) > 1:
        k_bin_center_leftmost = k_min
        k_bin_center_rightmost = 10**(logk_max - 0.5/logk_bins_per_decade_interp(logk_max))
        k_bin_centers = 10**(
            log10(k_bin_center_leftmost) + (np.log10(k_bin_centers) - log10(k_bin_centers[0]))*(
                (log10(k_bin_center_rightmost) - log10(k_bin_center_leftmost))
                /(log10(k_bin_centers[-1]) - log10(k_bin_centers[0]))
            )
        )
    return k_bin_centers

# Function for evaluating a string expression with variable
# substitutions given as the second argument.
def eval_bin_str(s, d=None, fail_on_error=True):
    s = re.sub(r'min *\(', 'MIN(', s)
    s = re.sub(r'max *\(', 'MAX(', s)
    if d is not None:
        d_complete = {}
        for key, val in d.items():
            key = key.removeprefix('k_')
            for key in (key, key.lower(), key.capitalize()):
                for key in (f'k_{key}', f'k{key}', key, key.replace('_', '')):
                    for key in (unicode(key), asciify(key)):
                        d_complete[key] = val
        for key, val in d_complete.items():
            s = s.replace(key, f'({val})')
    s = re.sub(r'MIN\(', 'min(', s)
    s = re.sub(r'MAX\(', 'max(', s)
    s_evaluated = eval_unit(s, units_dict, fail_on_error=fail_on_error)
    if s_evaluated is None:
        return s
    return s_evaluated

# Function which given a power spectrum declaration correctly populated
# with all fields will compute its power spectrum.
@cython.header(
    # Arguments
    declaration=object,  # PowerspecDeclaration
    # Locals
    a='double',
    component='Component',
    components=list,
    components_str=str,
    deconvolve='bint',
    factor='double',
    gridsize='Py_ssize_t',
    gridsizes_upstream=list,
    im='double',
    index='Py_ssize_t',
    interlace=str,
    interpolation='int',
    k2='Py_ssize_t',
    k2_max='Py_ssize_t',
    k_bin_indices='Py_ssize_t[::1]',
    k_bin_indices_ptr='Py_ssize_t*',
    k_bin_index='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    n_modes='Py_ssize_t[::1]',
    n_modes_ptr='Py_ssize_t*',
    normalization='double',
    power='double[::1]',
    power_ijk='double',
    power_ptr='double*',
    re='double',
    slab='double[:, :, ::1]',
    slab_ptr='double*',
    θ='double',
    returns='void',
)
def compute_powerspec(declaration):
    # Extract some variables from the power spectrum declaration
    components    = declaration.components
    gridsize      = declaration.gridsize
    interpolation = declaration.interpolation
    deconvolve    = declaration.deconvolve
    interlace     = declaration.interlace
    k2_max        = declaration.k2_max
    k_bin_indices = declaration.k_bin_indices
    n_modes       = declaration.n_modes
    power         = declaration.power
    # Begin progress message
    components_str = ', '.join([component.name for component in components])
    if len(components) > 1:
        components_str = f'{{{components_str}}}'
    masterprint(f'Computing power spectrum of {components_str} ...')
    # Interpolate the physical density of all components onto a global
    # grid by first interpolating onto individual upstream grids,
    # transforming to Fourier space and then adding them together.
    gridsizes_upstream = [
        component.powerspec_upstream_gridsize
        for component in components
    ]
    slab = interpolate_upstream(
        components, gridsizes_upstream, gridsize, 'ρ', interpolation,
        deconvolve=deconvolve, interlace=interlace, output_space='Fourier',
    )
    # Nullify the reused power array
    power[:] = 0
    # Loop over the slabs,
    # tallying up the power in the different k² bins.
    k_bin_indices_ptr = cython.address(k_bin_indices[:])
    power_ptr         = cython.address(power[:])
    slab_ptr          = cython.address(slab[:, :, :])
    for index, ki, kj, kk, factor, θ in fourier_loop(
        gridsize,
        sparse=True,
        skip_origin=True,
        k2_max=k2_max,
    ):
        k2 = ℤ[ℤ[ℤ[kj**2] + ki**2] + kk**2]
        # Compute the power at this k²
        re = slab_ptr[index    ]
        im = slab_ptr[index + 1]
        power_ijk = re**2 + im**2
        # Add power at this k² to the corresponding bin
        k_bin_index = k_bin_indices_ptr[k2]
        power_ptr[k_bin_index] += power_ijk
    # Sum power into the master process
    Reduce(
        sendbuf=(MPI.IN_PLACE if master else power),
        recvbuf=(power        if master else None),
        op=MPI.SUM,
    )
    # The master process now holds all the information needed
    if not master:
        return
    # We need to transform power from being the sum to being the
    # mean, by dividing by n_modes.
    # To completely remove the current normalization of the power, we
    # need to divide by the squared mean of values on the grids/slabs.
    # As we interpolated physical densities ρ to the grids, the mean of
    # all values will be
    #   mean(ρᵢⱼₖ) = ρ_bar = a**(-3(1 + w_eff))*ϱ_bar,
    # summed over all components.
    # As said, we need to divide the power by the square of mean(ρᵢⱼₖ).
    # To now add in a proper normalization, we need to multiply by
    # boxsize**3, resulting in a properly normalized power spectrum in
    # units of unit_length**3.
    a = universals.a
    normalization = 0
    for component in components:
        normalization += a**(-3*(1 + component.w_eff(a=a)))*component.ϱ_bar
    normalization **= -2
    normalization *= ℝ[boxsize**3]
    n_modes_ptr = cython.address(n_modes[:])
    for k_bin_index in range(power.shape[0]):
        power_ptr[k_bin_index] *= normalization/n_modes_ptr[k_bin_index]
    # Power spectrum computation complete
    masterprint('done')

# Function which given a power spectrum declaration correctly populated
# with all fields will compute its corrected power spectrum.
@cython.header(
    # Arguments
    declaration=object,  # PowerspecDeclaration
    # Locals
    a_backup='double',
    a_correction='double',
    cache_key=tuple,
    components=list,
    components_str=str,
    cosmoresults=object,  # CosmoResults
    correction='double[::1]',
    declaration_corrected=object,  # PowerspecDeclaration
    declaration_linear=object,  # PowerspecDeclaration
    filename=str,
    gridsize='Py_ssize_t',
    k_bin_centers='double[::1]',
    k_bin_index='Py_ssize_t',
    linear_component='Component',
    name=str,
    power='double[::1]',
    power_corrected='double[::1]',
    power_linear='double[::1]',
    size='Py_ssize_t',
    species_str=str,
    unit='double',
    returns='void',
)
def compute_powerspec_corrected(declaration):
    if not declaration.do_corrected:
        return
    # Time at which to evaluate the correction factor
    a_correction = 1
    # Extract some variables from the power spectrum declaration
    components      = declaration.components
    gridsize        = declaration.gridsize
    power           = declaration.power
    power_corrected = declaration.power_corrected
    # Begin progress message
    components_str = ', '.join([component.name for component in components])
    if len(components) > 1:
        components_str = f'{{{components_str}}}'
    masterprint(f'Computing corrected power spectrum of {components_str} ...')
    # Look up corrections in cache
    cache_key = tuple(components)
    if master:
        correction = power_corrected_cache.get(cache_key)
        if correction is not None:
            apply_powerspec_correction(declaration, correction)
    if bcast(correction is not None if master else None):
        masterprint('done')
        return
    # Get linear component
    linear_component = get_linear_component(components, gridsize)
    if not declaration.realization_correction:
        # Do not correct for the realization noise (cosmic variance),
        # i.e. only correct for the binning. That is, we ignore the
        # realization by using fixed amplitudes.
        # Note that the phases are unimportant.
        linear_component.realization_options['fixedamplitude'] = True
    # Never include non-Gaussianity in the linear component,
    # as this is similarly left out of the linear-theory prediction.
    linear_component.realization_options['nongaussianity'] = False
    # Check if corrections are available from the disk cache
    cosmoresults = compute_cosmo()
    if master:
        k_bin_centers = declaration.k_bin_centers
        unit = units.Mpc**(-1)  # some definite inverse length unit
        filename = get_reusable_filename(
            'powerspec',
            # Time of evaluation of correction factor
            a_correction,
            # Cosmology
            cosmoresults.id,
            {
                key: val
                for key, val in primordial_spectrum.items()
                if key != 'A_s'  # absolute scale does not matter
            },
            # Species
            set(linear_component.class_species.split('+')),
            enable_warm_dark_matter,
            # Modes
            gridsize,
            [f'{k_bin_center/unit:.11e}' for k_bin_center in k_bin_centers],
            class_dedicated_spectra,
            class_modes_per_decade,
            # Realization
            random_generator,
            random_seeds['primordial amplitudes'],
            primordial_noise_imprinting,
            linear_component.realization_options['fixedamplitude'],
            linear_component.realization_options['gauge'],
            linear_component.realization_options['backscale'],
            linear_component.realization_options['nongaussianity'],
        )
        if os.path.exists(filename):
            correction = np.loadtxt(filename)
            apply_powerspec_correction(declaration, correction)
    if bcast(correction is not None if master else None):
        masterprint('done')
        return
    # Temporarily substitute universals.a for a_correction
    a_backup = universals.a
    universals.a = a_correction
    # Realize linear component
    realize(linear_component, variables=0)
    # Compute power spectrum of linear imprinted component
    declaration_corrected = declaration._replace(
        do_data=True,
        components=[linear_component],
        power=power_corrected,
    )
    name = linear_component.name
    linear_component.name = f'{components_str} (linear imprint)'
    compute_powerspec(declaration_corrected)
    linear_component.name = name
    # Compute linear power spectrum
    if declaration.do_linear and a_backup == a_correction:
        # Linear power spectrum already computed
        declaration_linear = declaration
    else:
        declaration_linear = declaration._replace(
            do_linear=True,
            power_linear=(
                empty(power.shape[0], dtype=C2np['double'])
                if master else None
            ),
        )
        compute_powerspec_linear(declaration_linear)
    power_linear = declaration_linear.power_linear
    # Substitute back the original value for universals.a
    universals.a = a_backup
    # All data is now held by master
    if not master:
        return
    # Create array of corrections
    size = power.shape[0]
    correction = empty(size, dtype=C2np['double'])
    for k_bin_index in range(size):
        correction[k_bin_index] = power_linear[k_bin_index]/power_corrected[k_bin_index]
    # Apply corrections
    apply_powerspec_correction(declaration, correction)
    # Cache corrections to memory
    power_corrected_cache[cache_key] = correction
    # Cache corrections to disk
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    species_str = ', '.join(linear_component.species.split('+'))
    if len(linear_component.species.split('+')) > 1:
        species_str = f'{{{species_str}}}'
    np.savetxt(
        filename,
        correction,
        header=unicode(
            f'Power spectrum corrections for {species_str} at '
            f'k ∈ [{k_bin_centers[0]} {unit_length}⁻¹, {k_bin_centers[size - 1]} {unit_length}⁻¹]'
        ),
        encoding='utf-8',
    )
    masterprint('done')
# Cache used for power spectrum correction factors
cython.declare(power_corrected_cache=dict)
power_corrected_cache = {}

# Helper function for applying already computed corrections
# to power spectra.
@cython.header(
    # Arguments
    declaration=object,  # PowerspecDeclaration
    correction='double[::1]',
    # Locals
    k_bin_index='Py_ssize_t',
    power='double[::1]',
    power_corrected='double[::1]',
    returns='void',
)
def apply_powerspec_correction(declaration, correction):
    power           = declaration.power
    power_corrected = declaration.power_corrected
    for k_bin_index in range(power.shape[0]):
        power_corrected[k_bin_index] = power[k_bin_index]*correction[k_bin_index]

# Function which given a power spectrum declaration correctly populated
# with all fields will compute its linear power spectrum.
@cython.header(
    # Arguments
    declaration=object,  # PowerspecDeclaration
    # Locals
    component='Component',
    components=list,
    components_str=str,
    k_bin_centers='double[::1]',
    power_linear='double[::1]',
    returns='void',
)
def compute_powerspec_linear(declaration):
    if not declaration.do_linear:
        return
    # Extract some variables from the power spectrum declaration
    components    = declaration.components
    k_bin_centers = declaration.k_bin_centers
    power_linear  = declaration.power_linear
    # Begin progress message
    components_str = ', '.join([component.name for component in components])
    if len(components) > 1:
        components_str = f'{{{components_str}}}'
    masterprint(f'Computing linear power spectrum of {components_str} ...')
    # Fill power_linear with values of the linear power spectrum.
    # Only the master will hold the values.
    get_linear_powerspec(components, k_bin_centers, power_linear)
    # Done with the linear power spectrum computation
    masterprint('done')

# Function for saving power spectra
def save_powerspec(declarations, filename):
    column_headings_left = [
        ('do_data', f'k [{unit_length}⁻¹]', 'k_bin_centers'),
        ('do_data', 'modes',                'n_modes'),
    ]
    column_headings_components = [
        ('do_data'  ,    'component',   f'P [{unit_length}³]', 'power'),
        ('do_corrected', '(corrected)', f'P [{unit_length}³]', 'power_corrected'),
        ('do_linear',    '(linear)',    f'P [{unit_length}³]', 'power_linear'),
    ]
    grouping_func = lambda declaration: (
        len(declaration.k_bin_centers),
        declaration.k2_max,
        hashlib.sha1(asarray(declaration.k_bin_centers)).hexdigest(),
    )
    σ_unit = (  # σ₈ or similar. Units are Mpc/h by convention
        units.Mpc/(H0/(100*units.km/(units.s*units.Mpc))) if enable_Hubble else units.Mpc
    )
    extra_heading_func = (
        lambda declaration: (
            'σ{} = {{}}'
            .format(unicode_subscript(f'{declaration.tophat/σ_unit:.3g}'))
        )
    )
    extra_heading_fmt = (
        lambda declaration, do_attr: (
            r'= \{.+?\}',
            lambda m, σ=compute_powerspec_σ(
                declaration,
                kind=do_attr.removeprefix('do_'),
            ): m.group().format(σ),
        )
    )
    save_polyspec(
        'powerspec', declarations, filename,
        column_headings_left, column_headings_components, grouping_func,
        extra_heading_func, extra_heading_fmt,
    )

# Function which given a power spectrum declaration with the
# k_bin_centers and power fields correctly populated will compute the
# rms density variation of the power spectrum.
@cython.pheader(
    # Arguments
    declaration=object,  # PowerspecDeclaration
    kind=str,
    # Locals
    W='double',
    i='Py_ssize_t',
    k_bin_centers='double[::1]',
    k_bin_index='Py_ssize_t',
    k_magnitude='double',
    kR='double',
    mask=object,  # np.ndarray
    power='double[::1]',
    tophat='double',
    σ2='double',
    σ2_integrand='double[::1]',
    returns='double',
)
def compute_powerspec_σ(declaration, kind='data'):
    tophat        = declaration.tophat
    k_bin_centers = declaration.k_bin_centers
    power         = declaration.power
    # Change power array according to the requested kind
    kind = kind.lower().replace(' ', '').replace('-', '').replace('_', '')
    if kind == 'corrected':
        power = declaration.power_corrected
    elif kind == 'linear':
        power = declaration.power_linear
    elif kind != 'data':
        abort(
            f'compute_powerspec_σ() called with kind = "{kind}" '
            f'∉ {{"data", "corrected", "linear"}}'
        )
    # We need to truncate power and k_bin_centers
    # so that they do not contain NaNs.
    mask = np.isnan(power)
    if mask.any():
        power = asarray(power)[~mask]
        k_bin_centers = asarray(k_bin_centers)[~mask]
    # We cannot compute the integral if we have less than two points
    size = k_bin_centers.shape[0]
    if size < 2:
        return NaN
    # Ensure that the global σ2_integrand array is large enough
    if σ2_integrand_arr.shape[0] < size:
        σ2_integrand_arr.resize(size, refcheck=False)
    σ2_integrand = σ2_integrand_arr
    # The rms density variation σ_R (usually σ₈) is given by
    # σ² = ᔑd³k/(2π)³ W² power
    #    = 1/(2π)³ᔑ_0^∞ dk 4πk² W² power
    #    = 1/(2π²)ᔑ_0^∞ dk k² W² power,
    # where W = 3(sin(kR) - kR*cos(kR))/(kR)³.
    # Note that below, the factor 3 has been left out,
    # meaing that the variable W is really W/3.
    for k_bin_index in range(size):
        k_magnitude = k_bin_centers[k_bin_index]
        kR = k_magnitude*tophat
        if kR < 1e-3:
            # Use a Taylor expansion of W/3 around kR = 0
            W = 1./3. - 1./30.*kR**2
        else:
            W = (sin(kR) - kR*cos(kR))/kR**3
        σ2_integrand[k_bin_index] = (k_magnitude*W)**2*power[k_bin_index]
    # Do the integral, disregarding constant factors
    σ2 = np.trapz(σ2_integrand[:size], k_bin_centers)
    # The integrand above starts from k = k_min > 0, which means that
    # the interval from 0 to k_min has been left out. At k = 0, the
    # integrand vanishes. According to the trapezoidal rule, this means
    # that the full integral is missing the area of the triangle with
    # vertices (0, 0), (k_min, 0), (k_min, σ2_integrand[0]),
    # with k_min = k_bin_centers[0].
    σ2 += 0.5*k_bin_centers[0]*σ2_integrand[0]
    # Finally, remember the constant factor 1/(2π²) from the integral,
    # as well as the 3² missing from W².
    σ2 *= 3**2/(2*π**2)
    # Return the rms density variation σ
    return sqrt(σ2)
# Array used by the compute_powerspec_σ() function
cython.declare(σ2_integrand_arr=object)
σ2_integrand_arr = empty(1, dtype=C2np['double'])

# Top-level function for computing, plotting and saving bispectra
@cython.pheader(
    # Arguments
    components=list,
    filename=str,
    # Locals
    declaration=object,  # BispecDeclaration
    declarations=list,
    returns='void',
)
def bispec(components, filename):
    # Ensure cleared bispectrum grid cache
    bispec_grid_cache.clear()
    # Get bispectrum declarations.
    # This will also compute the number of modes within each bin.
    declarations = get_bispec_declarations(components)
    bispec_grid_cache.clear()
    # Compute bispectrum for each declaration
    for declaration in declarations:
        # Compute the bispectrum of the components in this
        # bispectrum declaration.
        # The results are stored in declaration.bpower
        # and declaration.bpower_reduced.
        # Only the master process holds the full bispectrum.
        compute_bispec(declaration)
        bispec_grid_cache.clear()
        # If specified, also compute the tree-level bispectrum.
        # The results are stored in declaration.bpower_treelevel
        # and declaration.bpower_reduced_treelevel.
        # Only the master process holds the tree-level bispectrum.
        compute_bispec_treelevel(declaration)
    # Dump bispectra to collective data file
    save_bispec(declarations, filename)
    # Dump bispectra to individual image files
    plot_bispec(declarations, filename)

# Function for getting declarations for all needed bispectra,
# given a list of components.
@cython.header(
    # Arguments
    components=list,
    # Locals
    bins=list,
    bpower='double[::1]',
    bpower_reduced='double[::1]',
    bpower_reduced_treelevel='double[::1]',
    bpower_treelevel='double[::1]',
    cache_key=tuple,
    component='Component',
    component_class_species_set=set,
    components_str=str,
    computation_order='Py_ssize_t[::1]',
    declaration=object,  # BispecDeclaration
    declarations=list,
    do_treelevel='bint',
    do_data='bint',
    i='Py_ssize_t',
    matter_like='bint',
    matter_class_species_set=set,
    n_modes='double[::1]',
    n_modes_expected='double[::1]',
    n_modes_power=dict,
    power=tuple,
    size='Py_ssize_t',
    returns=list,
)
def get_bispec_declarations(components):
    # Look up declarations in cache
    cache_key = tuple(components)
    declarations = bispec_declarations_cache.get(cache_key)
    if declarations:
        return declarations
    # Get declarations with basic fields populated
    declarations = get_output_declarations(
        'bispec',
        components,
        bispec_select,
        bispec_options,
        BispecDeclaration,
    )
    # Add missing declaration fields
    for i, declaration in enumerate(declarations):
        components_str = ', '.join([component.name for component in declaration.components])
        if len(declaration.components) > 1:
            components_str = f'{{{components_str}}}'
        # Enable do_data if any of the other "do attributes" are enabled
        do_data = declaration.do_data
        if not do_data:
            for do_attr in ['reduced', 'treelevel', 'plot']:
                if not getattr(declaration, f'do_{do_attr}'):
                    continue
                masterprint(
                    f'Enabling \'data\' for bispectra of {components_str} '
                    f'because \'{do_attr}\' is enabled'
                )
                do_data = True
                break
        # Get bin information
        (
            bins,
            n_modes, n_modes_expected,
            n_modes_power,
            computation_order,
        ) = get_bispec_bins(declaration)
        if len(bins) == 0:
            # Discard declaration as no valid bins were found
            masterwarn(
                f'Could not produce bispectrum for {components_str} due to '
                f'no valid bins existing. Try changing bispec_options["configuration"] '
                f'and/or bispec_options["shellthickness"].'
            )
            declarations[i] = None
            continue
        # Only do tree-level bispectrum computation if at least some of
        # the components are matter-like.
        matter_like = any([component.is_matter_like() for component in declaration.components])
        do_treelevel = declaration.do_treelevel
        if do_treelevel and not matter_like:
            plural = 'this is' if len(declaration.components) == 1 else 'these are'
            masterprint(
                f'Disabling \'tree-level\' for bispectra of {components_str} '
                f'because {plural} not matter-like'
            )
            do_treelevel = False
        # Allocate arrays for storing the data
        size = bcast(n_modes.shape[0] if master else None)
        bpower = empty(size, dtype=C2np['double'])
        bpower_reduced = (
            empty(size, dtype=C2np['double'])
            if master and declaration.do_reduced
            else None
        )
        power = tuple([
            (
                empty(size, dtype=C2np['double'])
                if master and declaration.do_reduced
                else None
            )
            for n in range(3)
        ])
        bpower_treelevel = (
            empty(size, dtype=C2np['double'])
            if master and do_treelevel
            else None
        )
        bpower_reduced_treelevel = (
            empty(size, dtype=C2np['double'])
            if master and declaration.do_reduced and do_treelevel
            else None
        )
        # Replace old declaration with a new, fully populated one
        declaration = declaration._replace(
            do_data=do_data,
            do_treelevel=do_treelevel,
            bins=bins,
            n_modes=n_modes,
            n_modes_expected=n_modes_expected,
            bpower=bpower,
            bpower_reduced=bpower_reduced,
            n_modes_power=n_modes_power,
            power=power,
            bpower_treelevel=bpower_treelevel,
            bpower_reduced_treelevel=bpower_reduced_treelevel,
            computation_order=computation_order,
        )
        declarations[i] = declaration
    # Only keep valid declarations
    declarations = [
        declaration
        for declaration in declarations
        if declaration is not None
    ]
    # Store declarations in cache and return
    bispec_declarations_cache[cache_key] = declarations
    return declarations
# Cache used by the get_bispec_declarations() function
cython.declare(bispec_declarations_cache=dict)
bispec_declarations_cache = {}
# Create the BispecDeclaration type
fields = (
    'components', 'do_data', 'do_reduced', 'do_treelevel', 'do_plot',
    'configuration', 'shellthickness', 'gridsize',
    'interpolation', 'deconvolve', 'interlace', 'significant_figures',
    'bins',
    'n_modes', 'n_modes_expected', 'n_modes_power',
    'bpower', 'bpower_reduced', 'power', 'bpower_treelevel', 'bpower_reduced_treelevel',
    'computation_order',
)
BispecDeclaration = collections.namedtuple(
    'BispecDeclaration', fields, defaults=[None]*len(fields),
)

# Function for constructing the bispectrum bins
@cython.header(
    # Arguments
    declaration=object,  # BispecDeclaration
    # Locals
    all_bins_known='bint',
    bin=object,  # BispecBin
    bin_index='Py_ssize_t',
    bin_index_all='Py_ssize_t',
    bins=list,
    bispec_bins=tuple,
    cache_key=tuple,
    component='Component',
    components=list,
    components_str=str,
    computation_order='Py_ssize_t[::1]',
    configuration=list,
    dset_name=str,
    filename=str,
    gridsize='Py_ssize_t',
    index_power='Py_ssize_t',
    k_fundamental='double',
    k_max='double',
    k_min='double',
    n='int',
    n_mode_power='double',
    n_modes='double[::1]',
    n_modes_bin='double',
    n_modes_expected='double[::1]',
    n_modes_power_arr='double[::1]',
    n_modes_power=dict,
    normalization='double',
    nyquist='Py_ssize_t',
    shell=tuple,
    shells_new=list,
    shellthickness=list,
    shellthickness_input=tuple,
    shellthickness_interp=tuple,
    shellthickness_value='double',
    returns=tuple,
)
def get_bispec_bins(declaration):
    """The returned objects are:
    - bins: List of BispecBin objects.
    - n_modes: Array storing the number of bispectrum modes (triangles)
      in each bin. Note that these are generally not integers.
      Only the master process holds true values.
    - n_modes_expected: Array storing the approximative expected number
      of bispectrum modes in each bin.
    - n_modes_power: Dictionary of arrays similar to n_modes,
      to hold number of power spectrum modes for P₁, P₂, P₃.
      The keys are shells (k_inner, k_outer) using grid units.
    - computation_order: Array storing the order in which the bispectrum
      modes should be computed.
    """
    components           = declaration.components
    gridsize             = declaration.gridsize
    configuration        = declaration.configuration
    shellthickness_input = declaration.shellthickness
    # Look up in the cache. What components are passed do not matter.
    cache_key = (
        gridsize,
        tuple([
            str(sorted(stringify_dict(d).items()) if isinstance(d, dict) else d)
            for d in configuration
        ]),
        tuple([
            str(sorted(stringify_dict(d).items()) if isinstance(d, dict) else d)
            for d in shellthickness_input
        ]),
    )
    bispec_bins = bispec_bins_cache.get(cache_key)
    if bispec_bins:
        return bispec_bins
    # Begin progress message
    components_str = ', '.join([component.name for component in components])
    if len(components) > 1:
        components_str = f'{{{components_str}}}'
    masterprint(f'Preprocessing bispectrum modes for {components_str} ...')
    # Parse shell thicknesses
    k_fundamental = ℝ[2*π/boxsize]
    nyquist = gridsize//2
    k_min, k_max = k_fundamental, k_fundamental*sqrt(3)*nyquist
    shellthickness = [
        {
            key/k_fundamental: val/k_fundamental
            for key, val in parse_bispec_param(
                shellthickness_input[n],
                gridsize,
                nyquist,
                k_min,
                k_max,
            ).items()
        }
        for n in range(3)
    ]
    # Canonicalize the shell thickness dictionaries
    # if they only contain one unique value.
    for n in range(3):
        if len(set(shellthickness[n].values())) == 1:
            shellthickness_value = next(iter(shellthickness[n].values()))
            shellthickness[n] = {
                1.0: shellthickness_value,
                2.0: shellthickness_value,
            }
    # Throw away the last few significant digits within the
    # shellthickness dicts, improving caching.
    shellthickness = [
        {
            float(f'{key:.11e}'): float(f'{val:.11e}')
            for key, val in shellthickness[n].items()
        }
        for n in range(3)
    ]
    # Create splines mapping linear k to shell thicknesses
    shellthickness_interp = tuple([
        get_controlpoint_spline(shellthickness[n], np.log10, transform_x_onlookup=True)
        for n in range(3)
    ])
    # Construct list of BispecBin objects
    bins, n_modes_expected = parse_bispec_configuration(
        configuration, declaration, shellthickness_interp, gridsize, nyquist, k_min, k_max,
    )
    # Get the computation order of the bins. This is constructed so that
    # bins containing shells of equal k² and shell thicknesses are
    # processed right after each other.
    computation_order = get_bispec_computation_order(bins)
    # Create power spectrum dictionary
    n_modes_power = {}
    # The master reads in the n_modes from disk
    filename = get_reusable_filename(
        'bispec',
        gridsize,
        *shellthickness,
        bispec_antialiasing,
        extension='hdf5',
    )
    all_bins_known = True
    n_modes = None
    if master:
        n_modes = zeros(len(bins), dtype=C2np['double'])
        if os.path.exists(filename):
            with open_hdf5(filename, mode='r') as hdf5_file:
                bispec_h5    = hdf5_file['B']
                powerspec_h5 = hdf5_file['P']
                for bin_index in range(n_modes.shape[0]):
                    # Bispectrum
                    bin = bins[bin_index]
                    dset_name = get_bispec_hdf5_dset_name(bin.shells)
                    dset = bispec_h5.get(dset_name)
                    if dset is None:
                        n_modes[bin_index] = -1  # flag as not found
                        all_bins_known = False
                        continue
                    else:
                        n_modes[bin_index] = dset[()]
                    # Power spectrum
                    for shell in bin.shells:
                        if shell in n_modes_power:
                            continue
                        dset_name = get_bispec_hdf5_dset_name(shell)
                        dset = powerspec_h5[dset_name]
                        n_modes_power[shell] = dset[()]
        else:
            all_bins_known = False
            n_modes[:] = -1  # flag all as not found
    # If any bins were not found on disk,
    # compute n_modes for these bins.
    all_bins_known = bcast(all_bins_known)
    if not all_bins_known:
        masterprint('Computing bispectrum modes ...')
        # Broadcast n_modes and n_modes_power.
        # Elements equal to -1 signals a bin with missing information.
        if not master:
            n_modes = empty(len(bins), dtype=C2np['double'])
        Bcast(n_modes)
        n_modes_power = bcast(n_modes_power)
        # The normalization of n_modes is the same for power spectrum
        # and bispectrum modes. A factor of gridsize⁻³ is required due
        # to the FFT. In addition, a factor ½ is used because each
        # (unique, real) mode is counted twice due to the complex
        # conjugation ↔ inversion symmetry of the real field(s).
        normalization = 0.5/gridsize**3
        for bin_index in range(n_modes.shape[0]):
            bin_index = computation_order[bin_index]
            bin = bins[bin_index]
            if n_modes[bin_index] != -1:
                # The number of modes for this bin is already known.
                # Impotantly, this knowledge is held by all processes.
                # To make the below Reduce() not add this number up
                # nprocs times, we replace the number by 0 on all
                # processes but the master.
                if not master:
                    n_modes[bin_index] = 0
                    for shell in bin.shells:
                        n_modes_power[shell] = 0
                continue
            n_modes_bin, shells_new = compute_bispec_value(
                gridsize,
                bin.shells,
                n_modes_power,
            )
            n_modes_bin *= normalization
            n_modes[bin_index] = n_modes_bin
            for shell in shells_new:
                n_modes_power[shell] *= normalization
        # Sum n_modes into the master process
        Reduce(
            sendbuf=(MPI.IN_PLACE if master else n_modes),
            recvbuf=(n_modes      if master else None),
            op=MPI.SUM,
        )
        n_modes_power_arr = np.fromiter(
            n_modes_power.values(),
            dtype=C2np['double'],
        )
        Reduce(
            sendbuf=(MPI.IN_PLACE      if master else n_modes_power_arr),
            recvbuf=(n_modes_power_arr if master else None),
            op=MPI.SUM,
        )
        for index_power, shell in enumerate(n_modes_power.keys()):
            n_modes_power[shell] = n_modes_power_arr[index_power]
        # The master now saves the missing n_modes to disk
        if master:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open_hdf5(filename, mode='a') as hdf5_file:
                # Bispectrum modes
                bispec_h5 = hdf5_file.require_group('B')
                for bin_index in range(n_modes.shape[0]):
                    bin = bins[bin_index]
                    dset_name = get_bispec_hdf5_dset_name(bin.shells)
                    if dset_name in bispec_h5:
                        continue
                    dset = bispec_h5.create_dataset(
                        dset_name,
                        (),
                        dtype=C2np['double'],
                    )
                    # When not using antialiasing the number of modes
                    # comes out as integers up to floating-point
                    # inaccuracies. Rounding the values ensures that
                    # they are saved correctly in the data (text) file.
                    # We store them in the HDF5 file in rounded
                    # form as well.
                    with unswitch:
                        if not bispec_antialiasing:
                            n_modes[bin_index] = round(n_modes[bin_index])
                    dset[()] = n_modes[bin_index]
                # Power spectrum modes
                powerspec_h5 = hdf5_file.require_group('P')
                for shell, n_mode_power in n_modes_power.items():
                    dset_name = get_bispec_hdf5_dset_name(shell)
                    if dset_name in powerspec_h5:
                        continue
                    dset = powerspec_h5.create_dataset(
                        dset_name,
                        (),
                        dtype=C2np['double'],
                    )
                    with unswitch:
                        if not bispec_antialiasing:
                            n_mode_power = round(n_mode_power)
                            n_modes_power[shell] = n_mode_power
                    dset[()] = n_mode_power
        # Done computing n_modes
        masterprint('done')
    # Cache result
    bispec_bins = (
        bins,
        n_modes, n_modes_expected,
        n_modes_power,
        computation_order,
    )
    bispec_bins_cache[cache_key] = bispec_bins
    # Done with pre-processing the modes
    masterprint('done')
    return bispec_bins
# Cache used by the get_bispec_bins() function
cython.declare(bispec_bins_cache=dict)
bispec_bins_cache = {}

# Helper function used by get_bispec_bins()
def get_bispec_hdf5_dset_name(shell_or_shells):
    return ', '.join(map(str, shell_or_shells))

# Function for parsing the user-defined bispectrum bins
@cython.header(
    # Arguments
    configuration=object,
    declaration=object,  # BispecDeclaration
    shellthickness_interp=tuple,
    gridsize='Py_ssize_t',
    nyquist='Py_ssize_t',
    k_min='double',
    k_max='double',
    # Locals
    abs_tol='double',
    bin=object,  # BispecBin
    bin_index='Py_ssize_t',
    bins=list,
    bins_all=list,
    conf=object,
    fac_logistic='double',
    index_k='Py_ssize_t',
    index_t='Py_ssize_t',
    index_μ='Py_ssize_t',
    invalid_shellthickness_k=set,
    linearised='bint',
    k='double',
    k_arr='double[::1]',
    k_arr_list=list,
    k_bgn='double',
    k_end='double',
    k_fundamental='double',
    k_grid='double',
    k_inner='double[::1]',
    k_inner_min='double',
    k_outer='double[::1]',
    k_outer_max='double',
    k_vec='double[::1]',
    k3_min='double',
    n='int',
    n_per_decade='int',
    n_modes_expected='double[::1]',
    n_modes_expected_all='double[::1]',
    n_modes_expected_bin='double',
    n_vec='int[::1]',
    ordering=tuple,
    rel_tol='double',
    shells=tuple,
    shellthickness='double',
    shellthickness_vec='double[::1]',
    size='Py_ssize_t',
    t='double',
    t_arr='double[::1]',
    t_arr_i='double[::1]',
    t_arr_list=list,
    t_max='double',
    t_min='double',
    μ='double',
    μ_arr='double[::1]',
    μ_arr_i='double[::1]',
    μ_arr_list=list,
    μ_max='double',
    μ_min='double',
    returns=tuple,
)
def parse_bispec_configuration(
    configuration, declaration, shellthickness_interp, gridsize, nyquist, k_min, k_max,
):
    """This function takes in the bispectrum configurations in a general
    format and returns a list of BispecBin objects.
    We work with two sets of bispec bin (triangle) parameters:
    - (k, t, μ): This is the parameterization as given by the user.
      We allow the largest possible parameter space:
         0 ≤ k
         0 ≤ t
        -1 ≤ μ ≤ +1
    - k_vec = (k₁, k₂, k₃): Magnitudes of the three Fourier vectors in
      floating-point grid units. These are obtained from (k, t, μ) as
        k₁ = k                           / k_fundamental
        k₂ = t*k                         / k_fundamental
        k₃ = sqrt(k₁² + k₂² - 2*μ*k₁*k₂) / k_fundamental
      but reindexed to obtain the canonical ordering k₁ >= k₂ >= k₃.
      Note that while this canonical (k₁, k₂, k₃) is obtainable from
      the user-defined (k, t, μ), the reverse is not generally true.
    If one stick to the canonical ordering k₁ >= k₂ >= k₃, the entire
    parameter space is spanned without duplicates using
      ½ ≤ t ≤ 1
      ½ ≤ μ ≤ 1
      ½ ≤ t*μ
    As we do not enforce the canonical ordering on the user input,
    these criteria are not checked. They are however respected by all
    named configurations implemented below.
    Even when fisregarding the canonical ordering k₁ >= k₂ >= k₃, it is
    always the case that
       0 ≤ k < ∞
       0 ≤ t < ∞
      -1 ≤ μ ≤ 1
    These criteria are checked.
    """
    k_fundamental = ℝ[2*π/boxsize]
    # Handle each configuration specification in turn
    if isinstance(configuration, list):
        bins_all = []
        n_modes_expected_all = empty(0, dtype=C2np['double'])
        for conf in configuration:
            bins, n_modes_expected = parse_bispec_configuration(
                conf, declaration, shellthickness_interp, gridsize, nyquist, k_min, k_max,
            )
            bins_all += bins
            if n_modes_expected is not None:
                n_modes_expected_all = np.concatenate(
                    (n_modes_expected_all, n_modes_expected),
                )
        return bins_all, n_modes_expected_all
    # Create arrays of k, t, μ, spanning the configuration space
    linearised = False
    if isinstance(configuration, dict):
        # Format: {'k': ..., 't': ..., 'μ': ...}
        k_arr = parse_bispec_param(configuration['k'], gridsize, nyquist, k_min, k_max)
        t_arr = parse_bispec_param(configuration['t'], gridsize, nyquist, k_min, k_max)
        μ_arr = parse_bispec_param(configuration['μ'], gridsize, nyquist, k_min, k_max)
    elif isinstance(configuration, tuple) and len(configuration) == 3:
        # Format: (k, t, μ)
        k_arr = parse_bispec_param(configuration[0], gridsize, nyquist, k_min, k_max)
        t_arr = parse_bispec_param(configuration[1], gridsize, nyquist, k_min, k_max)
        μ_arr = parse_bispec_param(configuration[2], gridsize, nyquist, k_min, k_max)
    elif isinstance(configuration, tuple) and len(configuration) == 2:
        # Format: (str, n_per_decade)
        if not isinstance(configuration[0], str):
            abort(f'Do not know how to parse bispectrum configuration {configuration}')
        configuration, n_per_decade = configuration
        # Set parameter space boundaries
        k_bgn = 5*k_fundamental
        k_end = 2./3.*nyquist*k_fundamental
        k3_min = 1.5*k_fundamental
        if k_end <= k_bgn:
            abort(f'Grid size {gridsize} too small for auto-setup of bispectrum configuration')
        # Number of bins n along the k₁ dimension. This will also be
        # the number of modes along the entire t and μ dimensions
        # (both of these dimension are considered to have a size of ½).
        n_per_decade = int(round(
            parse_bispec_param(n_per_decade, gridsize, nyquist, k_min, k_max, allow_scalar=True)
        ))
        n = np.max((2, int(round(n_per_decade*log10(k_end/k_bgn)))))
        # Tolerances for use with fuzzy comparisons
        rel_tol = 1e-2/n
        abs_tol = 5e-3/n
        # Create k₁ array. This will not be altered by any of the below.
        k_arr = logspace(log10(k_bgn), log10(k_end), n, dtype=C2np['double'])
        # Handle the various named configurations.
        # Some will set the linearised flag to True, meaning that
        # (k_arr[index], t_arr[index], μ_arr[index])
        # constitute a bin, as opposed to
        # (k_arr[index_k], t_arr[index_t], μ_arr[index_μ]).
        configuration = configuration.replace(' ', '').replace('-', '').replace('_', '').lower()
        if configuration.startswith('equilat'):
            # Equilateral configurations (1D);
            # k₁ = k₂ = k₃,
            # t = 1, μ = ½.
            configuration = 'equilateral'
            t = 1
            μ = 0.5
            t_arr = asarray([t], dtype=C2np['double'])
            μ_arr = asarray([μ], dtype=C2np['double'])
        elif configuration.startswith('stretch'):
            # Stretched configurations (1D);
            # k₁ = 2k₂ = 2k₃,
            # t = ½, μ = 1.
            # Though strict stretched triangles are completely collapsed
            # (i.e. have no enclosed area), measuring their bispectrum
            # still works out.
            configuration = 'stretched'
            t = 0.5
            μ = 1
            t_arr = asarray([t], dtype=C2np['double'])
            μ_arr = asarray([μ], dtype=C2np['double'])
        elif configuration.startswith('squeez'):
            # Squeezed configurations (1D);
            # k₁ = k₂, k₃ = 0,
            # t = 1, μ = 1.
            # With k₃ exactly equal to 0, no bispectrum data will be
            # obtainable. We instead use a value of μ slightly less than
            # unity, and only use k₁ for which k₃ is at least k3_min.
            configuration = 'squeezed'
            t = 1
            μ = 0.99
            k_bgn = pairmax(k_bgn, k3_min/sqrt(1 + t**2 - 2*t*μ))
            if k_end <= k_bgn:
                abort(
                    f'Grid size {gridsize} too small for auto-setup '
                    f'of {configuration} bispectrum configuration'
                )
            n = np.max((2, int(round(n*log10(k_end/k_bgn)))))
            k_arr = logspace(log10(k_bgn), log10(k_end), n, dtype=C2np['double'])
            t_arr = asarray([t], dtype=C2np['double'])
            μ_arr = asarray([μ], dtype=C2np['double'])
        elif 'iso' in configuration and 'right' in configuration:
            # Isosceles right configurations (1D);
            # k₁ = sqrt(2)k₂ = sqrt(2)k₃,
            # t = 1/sqrt(2), μ = 1/sqrt(2).
            configuration = 'isosceles right'
            t = 1/sqrt(2)
            μ = 1/sqrt(2)
            t_arr = asarray([t], dtype=C2np['double'])
            μ_arr = asarray([μ], dtype=C2np['double'])
        elif (
            ('iso' in configuration and 'large' in configuration)
            or 'liso' in configuration
        ):
            # L-isosceles configurations (2D);
            # k₁ = k₂ ≥ k₃,
            # t = 1, ½ ≤ μ ≤ 1.
            # With k₃ exactly equal to 0, no bispectrum data will be
            # obtainable. We thus use a maximum value of μ slightly less
            # than unity. The maximum value of μ used is chosen
            # such that the corresponding k₃ equals k3_min,
            # for each value of k₁.
            configuration = 'L-isosceles'
            t = 1
            μ_min = 0.5
            μ_max = 1
            linearised = True
            μ_arr = linspace(
                μ_min,
                μ_max,
                int(round(2*n*(μ_max - μ_min))),
                dtype=C2np['double'],
            )
            k_arr_list = []
            μ_arr_list = []
            for index_k in range(k_arr.shape[0]):
                k = k_arr[index_k]
                μ_max = pairmax(μ_min, (1 + t**2 - k3_min**2/k**2)/(2*t))
                for index_μ in range(μ_arr.shape[0] - 1, -1, -1):
                    if μ_arr[index_μ] < μ_max:
                        break
                μ_arr_i = unique(
                    np.append(μ_arr[:index_μ+1], μ_max),
                    rel_tol,
                    abs_tol,
                )
                μ_arr_list.append(μ_arr_i)
                k_arr_list.append(
                    asarray(np.repeat(k, μ_arr_i.shape[0]), dtype=C2np['double'])
                )
            k_arr = np.concatenate(k_arr_list)
            t_arr = asarray(np.repeat(t, k_arr.shape[0]), dtype=C2np['double'])
            μ_arr = np.concatenate(μ_arr_list)
        elif (
            ('iso' in configuration and 'small' in configuration)
            or 'siso' in configuration
        ):
            # S-isosceles configurations (2D);
            # k₁ ≥ k₂ = k₃,
            # ½ ≤ t ≤ 1, μ = 1/(2t).
            # As the limit configurations (t = ½ (stretched),
            # t = 1 (equilateral)) both behave nicely,
            # no special care is needed.
            configuration = 'S-isosceles'
            t_min = 0.5
            t_max = 1
            linearised = True
            t_arr_i = linspace(
                t_min,
                t_max,
                int(round(2*n*(t_max - t_min)/0.5*0.720599)),  # arc length = 0.720599
                dtype=C2np['double'],
            )
            μ_arr_i = 1/(2*asarray(t_arr_i))
            k_arr_list = []
            t_arr_list = []
            μ_arr_list = []
            for index_k in range(k_arr.shape[0]):
                k = k_arr[index_k]
                k_arr_list.append(
                    asarray(np.repeat(k, t_arr_i.shape[0]), dtype=C2np['double'])
                )
                t_arr_list.append(t_arr_i)
                μ_arr_list.append(μ_arr_i)
            k_arr = np.concatenate(k_arr_list)
            t_arr = np.concatenate(t_arr_list)
            μ_arr = np.concatenate(μ_arr_list)
        elif (
               configuration.startswith('elongat')
            or configuration.startswith('flat')
            or configuration.startswith('fold')
            or configuration.startswith('linear')
        ):
            # Elongated/flattened/folded/linear configurations (2D);
            # k₁ = k₂ + k₃,
            # ½ ≤ t ≤ 1, μ = 1.
            # With k₃ exactly equal to 0, no bispectrum data will be
            # obtainable. We thus use a maximum value of t slightly less
            # than unity. The maximum value of t used is chosen
            # such that the corresponding k₃ equals k3_min,
            # for each value of k₁.
            configuration = 'elongated'
            t_min = 0.5
            t_max = 1
            μ = 1
            linearised = True
            t_arr = linspace(
                t_min,
                t_max,
                int(round(2*n*(t_max - t_min))),
                dtype=C2np['double'],
            )
            k_arr_list = []
            t_arr_list = []
            for index_k in range(k_arr.shape[0]):
                k = k_arr[index_k]
                t_max = pairmax(
                    t_min,
                    μ - sqrt(k3_min**2/k**2 + μ**2 - 1),
                )
                for index_t in range(t_arr.shape[0] - 1, -1, -1):
                    if t_arr[index_t] < t_max:
                        break
                t_arr_i = unique(
                    np.append(t_arr[:index_t+1], t_max),
                    rel_tol,
                    abs_tol,
                )
                t_arr_list.append(t_arr_i)
                k_arr_list.append(
                    asarray(np.repeat(k, t_arr_i.shape[0]), dtype=C2np['double'])
                )
            k_arr = np.concatenate(k_arr_list)
            t_arr = np.concatenate(t_arr_list)
            μ_arr = asarray(np.repeat(μ, k_arr.shape[0]), dtype=C2np['double'])
        elif configuration.startswith('right'):
            # Right configurations (2D);
            # k₁² = k₂² + k₃³,
            # 1/sqrt(2) ≤ t = μ ≤ 1.
            # With k₃ exactly equal to 0, no bispectrum data will be
            # obtainable. We thus use a maximum value of t = μ slightly
            # less than unity. The maximum value of t = μ used is chosen
            # such that the corresponding k₃ equals k3_min,
            # for each value of k₁.
            configuration = 'right'
            t_min = 1/sqrt(2)
            t_max = 1
            linearised = True
            t_arr = linspace(
                t_min,
                t_max,
                int(round(2*n*sqrt(2)*(t_max - t_min))),
                dtype=C2np['double'],
            )
            k_arr_list = []
            t_arr_list = []
            for index_k in range(k_arr.shape[0]):
                k = k_arr[index_k]
                t_max = pairmax(t_min, sqrt(1 - k3_min**2/k**2))
                for index_t in range(t_arr.shape[0] - 1, -1, -1):
                    if t_arr[index_t] < t_max:
                        break
                t_arr_i = unique(
                    np.append(t_arr[:index_t+1], t_max),
                    rel_tol,
                    abs_tol,
                )
                t_arr_list.append(t_arr_i)
                k_arr_list.append(
                    asarray(np.repeat(k, t_arr_i.shape[0]), dtype=C2np['double'])
                )
            k_arr = np.concatenate(k_arr_list)
            t_arr = np.concatenate(t_arr_list)
            μ_arr = t_arr
        elif configuration.startswith('acute'):
            # Acute configurations (3D);
            # k₁² ≤ k₂² + k₃³,
            # 1/sqrt(2) ≤ t ≤ 1, 1/(2t) ≤ μ ≤ t.
            # With k₃ exactly equal to 0, no bispectrum data will be
            # obtainable. We thus use a maximum value of μ slightly less
            # than unity. The maximum value of μ used is chosen such
            # that the corresponding k₃ equals k3_min,
            # for each value of k₁ and t.
            configuration = 'acute'
            t_min = 1/sqrt(2)
            t_max = 1
            μ_min = 1/(2*t_max)
            μ_max = 1
            linearised = True
            t_arr = linspace(
                t_min,
                t_max,
                int(round(2*n*(t_max - t_min))),
                dtype=C2np['double'],
            )
            μ_arr = linspace(
                μ_min,
                μ_max,
                int(round(2*n*(μ_max - μ_min))),
                dtype=C2np['double'],
            )
            k_arr_list = []
            t_arr_list = []
            μ_arr_list = []
            for index_k in range(k_arr.shape[0]):
                k = k_arr[index_k]
                for index_t in range(t_arr.shape[0]):
                    t = t_arr[index_t]
                    μ_min = 1/(2*t)
                    μ_max = pairmin(t, (1 + t**2 - k3_min**2/k**2)/(2*t))
                    μ_min = pairmin(μ_min, μ_max)
                    for index_μ in range(μ_arr.shape[0] - 1, -1, -1):
                        if μ_arr[index_μ] < μ_max:
                            break
                    μ_arr_i = μ_arr[:index_μ+1]
                    for index_μ in range(μ_arr.shape[0]):
                        if μ_arr[index_μ] > μ_min:
                            break
                    μ_arr_i = unique(
                        np.concatenate(([μ_min], μ_arr_i[index_μ:], [μ_max])),
                        rel_tol,
                        abs_tol,
                    )
                    k_arr_list.append(
                        asarray(np.repeat(k, μ_arr_i.shape[0]), dtype=C2np['double'])
                    )
                    t_arr_list.append(
                        asarray(np.repeat(t, μ_arr_i.shape[0]), dtype=C2np['double'])
                    )
                    μ_arr_list.append(μ_arr_i)
            k_arr = np.concatenate(k_arr_list)
            t_arr = np.concatenate(t_arr_list)
            μ_arr = np.concatenate(μ_arr_list)
        elif configuration.startswith('obtuse'):
            # Acute configurations (3D);
            # k₁² ≥ k₂² + k₃³,
            # ½ ≤ t ≤ 1, (1/(2t) if t < 1/sqrt(2) else t) ≤ μ ≤ 1,
            # or (perhaps more naturally)
            # 1/sqrt(2) ≤ μ ≤ 1, 1/(2μ) ≤ t ≤ μ.
            # We will use the first description, as we would like μ to
            # be dependent on t and not vice versa.
            # With k₃ exactly equal to 0, no bispectrum data will be
            # obtainable. We thus use a maximum value of μ slightly less
            # than unity. The maximum value of μ used is chosen such
            # that the corresponding k₃ equals k3_min,
            # for each value of k₁ and t.
            configuration = 'obtuse'
            t_min = 0.5
            t_max = 1
            μ_min = 1/sqrt(2)
            μ_max = 1/(2*t_min)
            linearised = True
            t_arr = linspace(
                t_min,
                t_max,
                int(round(2*n*(t_max - t_min))),
                dtype=C2np['double'],
            )
            μ_arr = linspace(
                μ_min,
                μ_max,
                int(round(2*n*(μ_max - μ_min))),
                dtype=C2np['double'],
            )
            k_arr_list = []
            t_arr_list = []
            μ_arr_list = []
            for index_k in range(k_arr.shape[0]):
                k = k_arr[index_k]
                for index_t in range(t_arr.shape[0]):
                    t = t_arr[index_t]
                    μ_min = (1/(2*t) if t < 1/sqrt(2) else t)
                    μ_max = pairmin(1, (1 + t**2 - k3_min**2/k**2)/(2*t))
                    μ_min = pairmin(μ_min, μ_max)
                    for index_μ in range(μ_arr.shape[0] - 1, -1, -1):
                        if μ_arr[index_μ] < μ_max:
                            break
                    μ_arr_i = μ_arr[:index_μ+1]
                    for index_μ in range(μ_arr.shape[0]):
                        if μ_arr[index_μ] > μ_min:
                            break
                    μ_arr_i = unique(
                        np.concatenate(([μ_min], μ_arr_i[index_μ:], [μ_max])),
                        rel_tol,
                        abs_tol,
                    )
                    k_arr_list.append(
                        asarray(np.repeat(k, μ_arr_i.shape[0]), dtype=C2np['double'])
                    )
                    t_arr_list.append(
                        asarray(np.repeat(t, μ_arr_i.shape[0]), dtype=C2np['double'])
                    )
                    μ_arr_list.append(μ_arr_i)
            k_arr = np.concatenate(k_arr_list)
            t_arr = np.concatenate(t_arr_list)
            μ_arr = np.concatenate(μ_arr_list)
        elif configuration.startswith('all'):
            # All configurations (3D);
            # k₁ ≥ k₂ ≥ k₃,
            # ½ ≤ t ≤ 1, 1/(2t) ≤ μ ≤ 1,
            # With k₃ exactly equal to 0, no bispectrum data will be
            # obtainable. We thus use a maximum value of μ slightly less
            # than unity. The maximum value of μ used is chosen such
            # that the corresponding k₃ equals k3_min,
            # for each value of k₁ and t.
            configuration = 'all'
            t_min = 0.5
            t_max = 1
            μ_min = 1/(2*t_max)
            μ_max = 1
            linearised = True
            t_arr = linspace(
                t_min,
                t_max,
                int(round(2*n*(t_max - t_min))),
                dtype=C2np['double'],
            )
            μ_arr = linspace(
                μ_min,
                μ_max,
                int(round(2*n*(μ_max - μ_min))),
                dtype=C2np['double'],
            )
            k_arr_list = []
            t_arr_list = []
            μ_arr_list = []
            for index_k in range(k_arr.shape[0]):
                k = k_arr[index_k]
                for index_t in range(t_arr.shape[0]):
                    t = t_arr[index_t]
                    μ_min = 1/(2*t)
                    μ_max = pairmin(1, (1 + t**2 - k3_min**2/k**2)/(2*t))
                    μ_min = pairmin(μ_min, μ_max)
                    for index_μ in range(μ_arr.shape[0] - 1, -1, -1):
                        if μ_arr[index_μ] < μ_max:
                            break
                    μ_arr_i = μ_arr[:index_μ+1]
                    for index_μ in range(μ_arr.shape[0]):
                        if μ_arr[index_μ] > μ_min:
                            break
                    μ_arr_i = unique(
                        np.concatenate(([μ_min], μ_arr_i[index_μ:], [μ_max])),
                        rel_tol,
                        abs_tol,
                    )
                    k_arr_list.append(
                        asarray(np.repeat(k, μ_arr_i.shape[0]), dtype=C2np['double'])
                    )
                    t_arr_list.append(
                        asarray(np.repeat(t, μ_arr_i.shape[0]), dtype=C2np['double'])
                    )
                    μ_arr_list.append(μ_arr_i)
            k_arr = np.concatenate(k_arr_list)
            t_arr = np.concatenate(t_arr_list)
            μ_arr = np.concatenate(μ_arr_list)
        else:
            msg = f'Unknown triangle configuration "{configuration}"'
            if 'iso' in configuration:
                msg += '. Note that you can use either "L-isosceles" or "S-isosceles".'
            abort(msg)
        if k_arr.shape[0] == 0 or t_arr.shape[0] == 0 or μ_arr.shape[0] == 0:
            linearised = False
            masterwarn(
                f'No triangles were constructed for bispectrum configuration "{configuration}" '
                f'with grid size {gridsize}. Try increasing the grid size.'
            )
    else:
        abort(f'Do not know how to parse bispectrum configuration {configuration}')
    # Throw away k, t, μ values that are completely out-of-bounds.
    # Values only very slightly out-of-bounds will be corrected.
    if linearised:
        size = k_arr.shape[0]
    else:
        size = k_arr.shape[0]*t_arr.shape[0]*μ_arr.shape[0]
        k_arr = sift_bispec_param(k_arr,  0, ထ, 'k', f'{unit_length}⁻¹')
        t_arr = sift_bispec_param(t_arr,  0, ထ, 't',                   )
        μ_arr = sift_bispec_param(μ_arr, -1, 1, 'μ',                   )
    # Allocate arrays for expected number of modes
    n_modes_expected = (
        empty(size, dtype=C2np['double'])
        if master else None
    )
    # Construct bispectrum bins by iterating over the user-defined
    # configuration space (k, t, μ), which we transform to equivalent
    # (k₁, k₂, k₃), canonicalized so that k₁ >= k₂ >= k₃.
    k_inner_min = 0                        # grid units
    k_outer_max = (0.5 + nyquist)*sqrt(3)  # grid units
    n_vec              = empty(3, dtype=C2np['int'])
    k_vec              = empty(3, dtype=C2np['double'])
    shellthickness_vec = empty(3, dtype=C2np['double'])
    k_inner            = empty(3, dtype=C2np['double'])
    k_outer            = empty(3, dtype=C2np['double'])
    invalid_shellthickness_k = set()
    bins = []
    bin_index = -1
    for index_k in range(ℤ[k_arr.shape[0]]):
        k = k_arr[index_k]
        k_grid = k*ℝ[1/k_fundamental]
        for index_t in range(
            -linearised & index_k,
            (-linearised & (index_k + 1)) | -(not linearised) & ℤ[t_arr.shape[0]],
        ):
            t = t_arr[index_t]
            for index_μ in range(
                -linearised & index_k,
                (-linearised & (index_k + 1)) | -(not linearised) & ℤ[μ_arr.shape[0]],
            ):
                μ = μ_arr[index_μ]
                bin_index += 1
                # Canonicalize so that k₁ >= k₂ >= k₃
                k_vec[0] = k_grid
                k_vec[1] = t*k_vec[0]
                k_vec[2] = sqrt(k_vec[0]**2 + k_vec[1]**2 - 2*μ*k_vec[0]*k_vec[1])
                for n in range(3):
                    n_vec[n] = n
                if k_vec[1] > k_vec[0]:
                    k_vec[0], k_vec[1] = k_vec[1], k_vec[0]
                    n_vec[0], n_vec[1] = n_vec[1], n_vec[0]
                if k_vec[2] > k_vec[1]:
                    k_vec[1], k_vec[2] = k_vec[2], k_vec[1]
                    n_vec[1], n_vec[2] = n_vec[2], n_vec[1]
                if k_vec[1] > k_vec[0]:
                    k_vec[0], k_vec[1] = k_vec[1], k_vec[0]
                    n_vec[0], n_vec[1] = n_vec[1], n_vec[0]
                # Construct the three shells
                for n in range(3):
                    # Using n_vec[n] to index into shellthickness_interp
                    # ensures that we use the intended shell thickness
                    # despite the re-labelling (canonicalization)
                    # of the three sides.
                    shellthickness = shellthickness_interp[n_vec[n]](k_vec[n])
                    if shellthickness < 0:
                        if master and k_vec[n] not in invalid_shellthickness_k:
                            masterwarn(
                                f'Got bispectrum shell thickness {shellthickness} < 0 '
                                f'at k = {k_vec[n]*k_fundamental} {unit_length}⁻¹. '
                                f'A small positive shell thickness will be used in its stead.'
                            )
                            invalid_shellthickness_k.add(k_vec[n])
                        shellthickness = 0
                    shellthickness_vec[n] = shellthickness
                    k_outer[n] = get_bispec_shell_boundary(k_vec[n], shellthickness)
                    k_inner[n] = k_outer[n] - shellthickness
                # Compute the approximative numbers of expected modes
                with unswitch:
                    if n_modes_expected is not None:
                        # In https://arxiv.org/abs/1904.11055 they
                        # derive an approximation for the number of
                        # bispectrum modes, which comes out as
                        #   8π²k₁k₂k₃δk₁δk₂δk₃,
                        # where δkᵢ is the i'th shell thicknesses and
                        # all variables are in grid units.
                        # For "linear triangles" (k₁ = k₂ + k₃) they
                        # find the same expression but multiplied by ½.
                        # Below we implement this ½ by multiplying with
                        # a logistic function
                        #   1/(1 + exp(-fac*(-k₁/δk₁ + k₂/δk₂ + k₃/δk₃))),
                        # with fac some factor found empirically.
                        # This logistic function then interpolates
                        # between ½ and 1. For fac = 1.25,
                        # a value > 0.99 is obtained already at
                        # -k₁/δk₁ + k₂/δk₂ + k₃/δk₃ = 4.
                        # In addition, the above formulae count each
                        # (unique, real) mode twice due to the complex
                        # conjugation ↔ inversion symmetry of the real
                        # field(s). We effectively only use half of the
                        # total Fourier space, so we put in an additonal
                        # factor of ½.
                        n_modes_expected_bin = (
                            4*π**2*prod(k_vec)*prod(shellthickness_vec)
                        )
                        fac_logistic = 1.25  # found empirically
                        n_modes_expected_bin /= 1 + exp(
                            -fac_logistic
                            *(
                                - k_vec[0]/(shellthickness_vec[0] + machine_ϵ)
                                + k_vec[1]/(shellthickness_vec[1] + machine_ϵ)
                                + k_vec[2]/(shellthickness_vec[2] + machine_ϵ)
                            )
                        )
                        n_modes_expected[bin_index] = n_modes_expected_bin
                # Adjust the shells due to hard limits
                for n in range(3):
                    if shellthickness_vec[n] < shellthickness_min:
                        k_outer[n] = get_bispec_shell_boundary(k_vec[n], shellthickness_min)
                        k_inner[n] = k_outer[n] - shellthickness_min
                    # Due to the limits imposed on k_inner and k_outer,
                    # the effective shell thickness (k_outer - k_inner)
                    # for a given shell might be smaller than expected.
                    # Negative effective shell thickness is even
                    # possible, reflecting the fact that the whole shell
                    # is outside the legal region (this will not cause
                    # trouble; the number of modes will be computed to
                    # be 0 and the spectrum will be set to NaN).
                    k_inner[n] = pairmax(k_inner[n], k_inner_min)
                    k_outer[n] = pairmin(k_outer[n], k_outer_max)
                # For the shells we ignore the last few significant
                # digits, as this ensures robust (disk) cache lookup in
                # spite of round-off errors, and further enabling a
                # larger degree of reusability of grids within the
                # bispectrum computation.
                shells = (
                    (float(f'{k_inner[0]:.11e}'), float(f'{k_outer[0]:.11e}')),
                    (float(f'{k_inner[1]:.11e}'), float(f'{k_outer[1]:.11e}')),
                    (float(f'{k_inner[2]:.11e}'), float(f'{k_outer[2]:.11e}')),
                )
                # The correspondence (k, t, μ) → (0, 1, 2)
                ordering = tuple(np.argsort(n_vec))
                # Add bispectrum bin
                bin = BispecBin(
                    k, t, μ,
                    shells,
                    ordering,
                )
                bins.append(bin)
    # When not using antialiasing, the number of modes are written to
    # output files as integers. We explicitly round the expected number
    # of modes now (otherwise they will be rounded down
    # when later casted to integers).
    if not bispec_antialiasing and n_modes_expected is not None:
        for bin_index in range(size):
            n_modes_expected[bin_index] = round(n_modes_expected[bin_index])
    return bins, n_modes_expected
# Minimum allowed shell thickness, in grid units
cython.declare(shellthickness_min='double')
shellthickness_min = sqrt(machine_ϵ)
# Type used for the bispectrum bins
BispecBin = collections.namedtuple(
    'BispecBin', (
        # The (k, t, μ) parameterization specified by the user,
        # with k in inverse length units.
        'k',
        't',
        'μ',
        # The inner and outer shell radii
        #   ((k₁_inner, k₁_outer), (k₂_inner, k₂_outer), (k₃_inner, k₃_outer))
        # in floating-point grid units.
        'shells',
        # The (k₁, k₂, k₃) parameterization used for the shells is
        # equivalent to (k, t, μ), but canonicalized so that
        #   k₁ >= k₂ >= k₃.
        # This canonicalization shuffles the shells, so that it is no
        # longer possible to know which kᵢ belongs to k, t, or μ. The
        # order is recorded below, so that shells[ordering[i]]
        # corresponds to (k, t, μ)[i].
        'ordering',
    ),
)

# Helper functions used by the get_bispec_bins()
# and parse_bispec_configuration() functions.
def parse_bispec_param(
    bispec_input, gridsize, nyquist, k_min, k_max, allow_scalar=False,
):
    k_fundamental = ℝ[2*π/boxsize]
    if isinstance(bispec_input, dict):
        # Dict with arbitrary keys and values.
        # We should likewise return a dict.
        bispec_input_transformed = {}
        for key, val in bispec_input.items():
            key = parse_bispec_param(key, gridsize, nyquist, k_min, k_max, allow_scalar=True)
            if isinstance(key, str):
                abort(f'Failed to parse bispectrum input {key}')
            val = parse_bispec_param(val, gridsize, nyquist, k_min, k_max, allow_scalar=True)
            bispec_input_transformed[key] = val
        # Evaluate str values as functions of k
        keys = sorted(bispec_input_transformed.keys())
        if keys == [0.0, ထ]:
            # Replace 0.0 and ထ with k_min and k_max
            vals = tuple(bispec_input_transformed.values())
            if len(set(vals)) != 1:
                abort(f'Illegal bispectrum parameter "{bispec_input}"')
            val = vals[0]
            bispec_input_transformed = {k_min: val, k_max: val}
            keys = sorted(bispec_input_transformed.keys())
        for k_left, k_right in itertools.zip_longest(keys, keys[1:]):
            val = bispec_input_transformed[k_left]
            if not isinstance(val, str):
                continue
            bispec_input_transformed.pop(k_left)
            val_sub = []
            for i, c in enumerate(val):
                if (
                       (c != 'k')
                    or (i > 0 and val[i - 1].isidentifier())
                    or (i < len(val) - 1 and val[-1].isidentifier())
                ):
                    val_sub.append(c)
                    continue
                val_sub += list('((k))')
            val_sub = ''.join(val_sub)
            if k_right is None:
                k_right = k_max
            for k in arange(k_left, k_right + k_fundamental*(1 - 1e-3), k_fundamental):
                bispec_input_transformed[k] = eval_unit(
                    re.sub(r'\(k\)', str(k), val_sub), units_dict,
                )
        return bispec_input_transformed
    elif isinstance(bispec_input, (int, float, np.integer, np.floating)):
        # Single number
        if allow_scalar:
            return float(bispec_input)
        else:
            arr = asarray([bispec_input], dtype=C2np['double'])
    elif isinstance(bispec_input, str):
        # Str expression
        bispec_input_transformed = eval_bin_str(
            bispec_input,
            {
                'nyquist'      : k_fundamental*nyquist,
                'gridsize'     : gridsize,
                'k_min'        : k_min,
                'k_max'        : k_max,
                'k_fundamental': k_fundamental,
                'k_f'          : k_fundamental,
            },
            fail_on_error=False,
        )
        if isinstance(bispec_input_transformed, str):
            # Function of k given as a str expression
            return bispec_input_transformed
        return parse_bispec_param(
            bispec_input_transformed,
            gridsize,
            nyquist,
            k_min,
            k_max,
            allow_scalar=allow_scalar,
        )
    else:
        # Several numbers
        bispec_input = list(bispec_input)
        if not bispec_input:
            abort('Got empty bispectrum input')
        arr = np.concatenate([
            parse_bispec_param(el, gridsize, nyquist, k_min, k_max, allow_scalar=allow_scalar)
            for el in bispec_input
        ])
    return arr
@cython.header(
    # Arguments
    values='double[::1]',
    value_min='double',
    value_max='double',
    name=str,
    unit=str,
    # Locals
    accepted=list,
    accepted_set=set,
    adjusted='bint',
    i='Py_ssize_t',
    rejected=list,
    value='double',
    returns='double[::1]',
)
def sift_bispec_param(values, value_min, value_max, name, unit=''):
    if unit:
        unit = f' {unit}'
    accepted = []
    accepted_set = set()
    rejected = []
    for i in range(values.shape[0]):
        value = values[i]
        if not (value_min <= value <= value_max):
            adjusted = False
            if value != 0 and abs(value) != ထ:
                if value_min != 0 and value_min != -ထ and isclose(value, value_min):
                    value = value_min
                    adjusted = True
                elif value_max != 0 and value_max != ထ and isclose(value, value_max):
                    value = value_max
                    adjusted = True
            if not adjusted:
                rejected.append(value)
                continue
        if value in accepted_set:
            continue
        accepted.append(value)
        accepted_set.add(value)
    if rejected:
        masterwarn(
            f'The following bispectrum {name} values are out-of-bounds '
            f'({value_min}{unit}, {value_max}{unit}): '
            f'{asarray(rejected)}{unit}'
        )
    return asarray(accepted, dtype=C2np['double'])

# Function for computing the outer k of a bispectrum shell
@cython.header(
    # Arguments
    k='double',
    shellthickness='double',
    # Locals
    k_inner='double',
    k_outer='double',
    k2='double',
    tmp='double',
    shellthickness2='double',
    returns='double',
)
def get_bispec_shell_boundary(k, shellthickness):
    """The arguments should be in (floating point) grid units.
    Though only k_outer is returned, k_inner is trivially
    obtained through k_inner = k_outer - shellthickness.
    """
    import scipy.optimize
    if not bispec_antialiasing:
        # Place k_outer and k_inner so that k is in the middle
        k_outer = k + 0.5*shellthickness
        return k_outer
    # The below formula ensures that the average k over the shell
    # is exactly the k specified.
    k2 = k**2
    shellthickness2 = shellthickness**2
    tmp = cbrt(
        + k2*k
        + shellthickness*sqrt(
            + 144./64.*k2**2
            - 108./64.*k2   *shellthickness2
            +  27./64.      *shellthickness2**2
        )
    )
    k_outer = 1./3.*(
        + k
        + (3./2.*shellthickness      )
        - (3./4.*shellthickness2 - k2)/tmp
        + tmp
    )
    # For reasonable (typical) shell parameters we instead opt for
    # a shell where the average log(k) is exactly
    # the log of the k specified.
    k_inner = k_outer - shellthickness
    if (
            k_inner >= 0.5                            # not near the origin
        and shellthickness > shellthickness_min       # not too thin
        and k_outer > (1 + machine_ϵ)*shellthickness  # not too thick
        and (
             bispec_shell_boundary_logrootexpr(k_outer,            k, shellthickness)
            *bispec_shell_boundary_logrootexpr(k + shellthickness, k, shellthickness)
        ) < 0                                         # zero bracketed by boundaries
    ):
        k_outer = scipy.optimize.root_scalar(
            bispec_shell_boundary_logrootexpr,
            (k, shellthickness),
            bracket=(k_outer, k + shellthickness),
        ).root
    return k_outer
# Helper function for get_bispec_shell_boundary()
@cython.pheader(
    # Arguments
    k_outer='double',
    k='double',
    shellthickness='double',
    # Locals
    returns='double',
)
def bispec_shell_boundary_logrootexpr(k_outer, k, shellthickness):
    """For some k and shellthickness, solving for the root k_outer of
    the below expression results in the k_outer that ensures that the
    average log(k) over the shell is the log of the specified k.
    """
    return (
        + (shellthickness/k_outer - 1)**3*(log((k_outer - shellthickness)/k) - 1./3.)
        + log(k_outer/k)
        - 1./3.
    )

# Function for fining an ordering of bispectrum bins which lends itself
# (close) to maximum reusage of grids during bispectrum computation.
@cython.header(
    # Arguments
    bins=list,
    n_max='Py_ssize_t',
    offset='Py_ssize_t',
    # Locals
    data_lexsorted='double[:, ::1]',
    dissimilarity='int',
    dissimilarity_min='int',
    dissimilarity_tot='Py_ssize_t',
    dissimilarity_tot_min='Py_ssize_t',
    i='Py_ssize_t',
    indices_lexsorted='Py_ssize_t[::1]',
    indices_sorted='Py_ssize_t[::1]',
    indices_sorted_best='Py_ssize_t[::1]',
    indices_sorted_list=list,
    info_lexsorted=list,
    j='Py_ssize_t',
    j_min='Py_ssize_t',
    n='Py_ssize_t',
    reverse='bint',
    row='double[::1]',
    row_prev='double[::1]',
    shift='int',
    used='signed char[::1]',
    returns='Py_ssize_t[::1]',
)
def get_bispec_computation_order(bins, n_max=512, offset=0):
    """With bins a list of BispecBin's, this function returns indices
    corresponding to an ordering of the bins where bins containing
    one or more identical shells appear next to each other.
    Finding the optimal ordering is reminiscent of the travelling
    salesman problem and so is probably NP-hard. This function finds a
    decent sorting, not necessarily the optimal one.
    """
    n = len(bins)
    if n == 0:
        return empty(n, dtype=C2np['Py_ssize_t'])
    # Carry out the processing in chunks of size n_max
    if n > n_max:
        indices_sorted_list = []
        for i in range(0, n, n_max):
            indices_sorted_list.append(get_bispec_computation_order(bins[i:i+n_max], n_max, i))
        return np.concatenate(indices_sorted_list)
    # Indices corresponding to the ordering of the data.
    # We always start with the zeroth element.
    indices_sorted_best = empty(n, dtype=C2np['Py_ssize_t'])
    indices_sorted      = empty(n, dtype=C2np['Py_ssize_t'])
    indices_sorted[0] = 0
    # Array for flagging whether a bin has been used or not
    used = empty(n, dtype=C2np['signed char'])
    # We do the whole computation 6 times,
    # corresponding to sorting on all column orderings.
    dissimilarity_tot_min = 7*n  # some number greater than 6*n
    for shift in range(3):
        for reverse in range(2):
            if reverse:
                data_lexsorted = np.flip(data_lexsorted, axis=0).copy()
            else:
                # Create n×6 array with rows of the form
                #   (k₁_inner, k₁_outer, k₂_inner, k₂_outer, k₃_inner, k₃_outer)
                info_lexsorted = sorted(
                    [
                        (i, bin.shells)
                        for i, bin in enumerate(bins)
                    ],
                    key=functools.partial(bispec_roll, shift=shift),
                )
                data_lexsorted = asarray(
                    list(map(
                        operator.itemgetter(1),
                        info_lexsorted,
                    )),
                    dtype=C2np['double'],
                ).reshape((n, 6))
            # Reset bin usage
            used[:] = 0
            # The zeroth bin is already included
            row_prev = data_lexsorted[0, :]
            # Given row_prev, find the next bin by requiring
            # that these have the minimal mutual dissimilarity.
            # Do this until all bins are used.
            j_min = 0
            for i in range(1, n):
                dissimilarity_min = 7  # some number greater than 6
                for j in range(1, n):
                    if used[j]:
                        continue
                    row = data_lexsorted[j, :]
                    dissimilarity = compute_bispec_bin_dissimilarity(row_prev, row)
                    if dissimilarity < dissimilarity_min:
                        dissimilarity_min = dissimilarity
                        j_min = j
                # Record index of bin with lowest dissimilarity
                indices_sorted[i] = j_min
                # Mark this bin as used
                used[j_min] = 1
                row_prev = data_lexsorted[j_min, :]
            # Measure total dissimilarity
            dissimilarity_tot = 0
            row_prev = data_lexsorted[indices_sorted[0], :]
            for i in range(1, n):
                row = data_lexsorted[indices_sorted[i], :]
                dissimilarity_tot += compute_bispec_bin_dissimilarity(row_prev, row)
                row_prev = row
            # Record the sorting with the minimal total dissimilarity
            if dissimilarity_tot < dissimilarity_tot_min:
                dissimilarity_tot_min = dissimilarity_tot
                indices_lexsorted = np.fromiter(
                    map(
                        operator.itemgetter(0),
                        reversed(info_lexsorted) if reverse else info_lexsorted,
                    ),
                    dtype=C2np['Py_ssize_t'],
                )
                for i in range(n):
                    indices_sorted_best[i] = indices_lexsorted[indices_sorted[i]] + offset
    # Return the best sorting in the original format
    return indices_sorted_best
# Helper functions for get_bispec_computation_order()
def bispec_roll(t, shift):
    i, a = t
    return tuple(map(tuple, np.roll(asarray(a, dtype=object), shift=shift, axis=0)))
@cython.header(
    # Arguments
    row0='double[::1]',
    row1='double[::1]',
    # Locals
    i='int',
    j='int',
    k_inner='double',
    k_outer='double',
    n_unique='int',
    r='int',
    row='double[::1]',
    returns='int',
)
def compute_bispec_bin_dissimilarity(row0, row1):
    # As the measure of dissimilarity between bins we count the number
    # of unique shells (k_inner, k_outer). This can be done easily as
    #   len({
    #       (row0[0], row0[1]),
    #       (row0[2], row0[3]),
    #       (row0[4], row0[5]),
    #       (row1[0], row1[1]),
    #       (row1[2], row1[3]),
    #       (row1[4], row1[5]),
    #   })
    # though this is several times slower than the code below.
    # Note that this notion of dissimilarity is really equivalent
    # to the Hamming distance.
    n_unique = 0
    shells_encountered[:] = -1
    for r in range(2):
        row = (row0 if r == 0 else row1)
        for i in range(3):
            k_inner = row[2*i    ]
            k_outer = row[2*i + 1]
            for j in range(n_unique):
                if (
                      (shells_encountered[2*j    ] == k_inner)
                    & (shells_encountered[2*j + 1] == k_outer)
                ):
                    break
            else:
                shells_encountered[2*n_unique    ] = k_inner
                shells_encountered[2*n_unique + 1] = k_outer
                n_unique += 1
    return n_unique
cython.declare(shells_encountered='double[::1]')
shells_encountered = empty(12, dtype=C2np['double'])

# Function for computing a single bispectrum value
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    shells=tuple,
    power_dict=dict,
    slab_data='double[:, :, ::1]',
    # Locals
    buffer_numbers=set,
    compute_power_0='bint',
    compute_power_1='bint',
    compute_power_2='bint',
    grid='double[:, :, ::1]',
    grid_0='double[:, :, ::1]',
    grid_0_ptr='double*',
    grid_1='double[:, :, ::1]',
    grid_1_ptr='double*',
    grid_2='double[:, :, ::1]',
    grid_2_ptr='double*',
    gridinfos=list,
    i='Py_ssize_t',
    index='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    loot=tuple,
    multiplicity='int',
    multiplicity_0='int',
    multiplicity_1='int',
    n='int',
    shell=tuple,
    shell_0=tuple,
    shell_1=tuple,
    shell_2=tuple,
    shellinfo=tuple,
    shellinfo_0=tuple,
    shellinfo_1=tuple,
    shellinfo_2=tuple,
    shellinfos=list,
    shellinfos_counter=object,  # collections.Counter
    shells_new=list,
    value='double',
    value_power_0='double',
    value_power_1='double',
    value_power_2='double',
    returns=object, # tuple or double
)
def compute_bispec_value(gridsize, shells, power_dict, slab_data=None):
    """This function computes the bispectrum using the
    Scoccimarro estimator (https://arxiv.org/abs/astro-ph/0004086).
    When no slab_data is supplied, the indicator grids will be
    constructed and the number of modes (triangles) is returned.
    When slab_data is supplied, the data grids will be constructed and
    the bispectrum value is returned. In both cases the returned value
    is unnormalized and still not summed up over processes.
    We choose to hold all three grids in memory simultaneously, as this
    is generally faster than multiplying grids together (as this
    requires more writes). The downside is that we use one grid more
    than minimally needed. One or more of the grids are reused from the
    previous call if the same shell is to be used. The global
    bispec_grid_cache store the three grids, with keys corresponding to
    the shell. Which global buffer number (0, 1 or 2) the grid is
    allocated as is stored as well.
    We do not create all three grids in a case where two or three of the
    shells are identical, but instead reuse the grids as appropriate.
    """
    # Specification and multiplicity of the shells
    shellinfos_counter = collections.Counter(
        zip(shells, [gridsize]*3, [slab_data is None]*3)
    )
    # Fetch available grids from the cache
    shellinfos = []
    gridinfos = []
    buffer_numbers = {0, 1, 2}
    for shellinfo, multiplicity in shellinfos_counter.items():
        loot = bispec_grid_cache.get(shellinfo)
        if loot is not None:
            shellinfos.append(shellinfo)
            n, grid = loot
            gridinfos.append((multiplicity, n, grid))
            buffer_numbers -= {n}
    # Create grids not found in the cache
    for shellinfo, multiplicity in shellinfos_counter.items():
        if shellinfo not in bispec_grid_cache:
            shellinfos.append(shellinfo)
            shell, *_ = shellinfo
            n = buffer_numbers.pop()
            grid = get_bispec_grid(gridsize, shell, n, slab_data)
            gridinfos.append((multiplicity, n, grid))
    # Compute sum over grid points of product of grids.
    # For performance, we write out specialized code for each of the
    # three cases for the number of unique grids.
    value = 0
    value_power_0   = value_power_1   = value_power_2   = 0
    compute_power_0 = compute_power_1 = compute_power_2 = False
    if len(gridinfos) == 3:
        shellinfo_0, shellinfo_1, shellinfo_2 = shellinfos
        shell_0, *_ = shellinfo_0
        shell_1, *_ = shellinfo_1
        shell_2, *_ = shellinfo_2
        (*_, grid_0), (*_, grid_1), (*_, grid_2) = gridinfos
        compute_power_0 = (𝔹[power_dict is not None] and shell_0 not in power_dict)
        compute_power_1 = (𝔹[power_dict is not None] and shell_1 not in power_dict)
        compute_power_2 = (𝔹[power_dict is not None] and shell_2 not in power_dict)
        grid_0_ptr = cython.address(grid_0[:, :, :])
        grid_1_ptr = cython.address(grid_1[:, :, :])
        grid_2_ptr = cython.address(grid_2[:, :, :])
        for index, i, j, k in domain_loop(gridsize, skip_ghosts=True):
            value += grid_0_ptr[index]*grid_1_ptr[index]*grid_2_ptr[index]
            with unswitch:
                if compute_power_0:
                    value_power_0 += grid_0_ptr[index]**2
            with unswitch:
                if compute_power_1:
                    value_power_1 += grid_1_ptr[index]**2
            with unswitch:
                if compute_power_2:
                    value_power_2 += grid_2_ptr[index]**2
        if compute_power_0:
            power_dict[shell_0] = value_power_0
        if compute_power_1:
            power_dict[shell_1] = value_power_1
        if compute_power_2:
            power_dict[shell_2] = value_power_2
    elif len(gridinfos) == 2:
        shellinfo_0, shellinfo_1 = shellinfos
        (multiplicity_0, *_, grid_0), (multiplicity_1, *_, grid_1) = gridinfos
        if multiplicity_0 > multiplicity_1:
            grid_0, grid_1 = grid_1, grid_0
            shellinfo_0, shellinfo_1 = shellinfo_1, shellinfo_0
        shell_0, *_ = shellinfo_0
        shell_1, *_ = shellinfo_1
        compute_power_0 = (𝔹[power_dict is not None] and shell_0 not in power_dict)
        compute_power_1 = (𝔹[power_dict is not None] and shell_1 not in power_dict)
        grid_0_ptr = cython.address(grid_0[:, :, :])
        grid_1_ptr = cython.address(grid_1[:, :, :])
        for index, i, j, k in domain_loop(gridsize, skip_ghosts=True):
            value += grid_0_ptr[index]*grid_1_ptr[index]**2
            with unswitch:
                if compute_power_0:
                    value_power_0 += grid_0_ptr[index]**2
            with unswitch:
                if compute_power_1:
                    value_power_1 += grid_1_ptr[index]**2
        if compute_power_0:
            power_dict[shell_0] = value_power_0
        if compute_power_1:
            power_dict[shell_1] = value_power_1
    else:  # len(gridinfos) == 1
        shellinfo_0, = shellinfos
        shell_0, *_ = shellinfo_0
        (*_, grid_0), = gridinfos
        compute_power_0 = (𝔹[power_dict is not None] and shell_0 not in power_dict)
        grid_0_ptr = cython.address(grid_0[:, :, :])
        for index, i, j, k in domain_loop(gridsize, skip_ghosts=True):
            value += grid_0_ptr[index]**3
            with unswitch:
                if compute_power_0:
                    value_power_0 += grid_0_ptr[index]**2
        if compute_power_0:
            power_dict[shell_0] = value_power_0
    # Populated shell keys in power_dict
    shells_new = []
    if compute_power_0:
        shells_new.append(shell_0)
    if compute_power_1:
        shells_new.append(shell_1)
    if compute_power_2:
        shells_new.append(shell_2)
    # Store grids in cache
    bispec_grid_cache.clear()
    for shellinfo, (multiplicity, n, grid) in zip(shellinfos, gridinfos):
        bispec_grid_cache[shellinfo] = (n, grid)
    # Return the computed value and perhaps
    # the list of populated power spectrum shells/keys.
    if slab_data is None:
        return value, shells_new
    else:
        return value
# Global cache used by the compute_bispec_value() function
cython.declare(bispec_grid_cache=dict)
bispec_grid_cache = {}

# Function for constructing either an indicator field or a data field
# used for bispectra computation.
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    shell=tuple,
    buffer_name=object,
    slab_data='double[:, :, ::1]',
    # Locals
    frac='double',
    grid='double[:, :, ::1]',
    index_neg='Py_ssize_t',
    index_pos='Py_ssize_t',
    k_inner='double',
    k_outer='double',
    k2_inner='double',
    k2_outer='double',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    nyquist='Py_ssize_t',
    slab='double[:, :, ::1]',
    slab_data_ptr='double*',
    slab_ptr='double*',
    slab_size_k='Py_ssize_t',
    returns='double[:, :, ::1]',
)
def get_bispec_grid(gridsize, shell, buffer_name, slab_data):
    # Extract inner and outer radius from shell
    k_inner, k_outer = shell
    k2_inner = k_inner**2
    k2_outer = k_outer**2
    # Fetch nullified slab
    slab = get_fftw_slab(gridsize, 'slab_bispec', nullify=True)
    slab_ptr = cython.address(slab[:, :, :])
    # Populate the shell defined by (k2_inner, k2_outer) with values.
    # We loop over ki ≥ 0 only, obtaining the ki > 0 half
    # through symmetry.
    slab_data_ptr = NULL
    if slab_data is not None:
        slab_data_ptr = cython.address(slab_data[:, :, :])
    nyquist = gridsize//2
    slab_size_k = gridsize + 2
    for index_pos, ki, kj, kk in fourier_shell_loop(
        gridsize, k_inner, k_outer,
        skip_origin=True, skip_negative_ki=True,
    ):
        # The fractional overlap between the shell and this cell
        with unswitch:
            if bispec_antialiasing:
                frac = get_bispec_overlap_cellshell(ki, kj, kk, k_inner, k_outer)
            else:
                frac = (k2_inner < ℤ[ℤ[kj**2] + ki**2] + kk**2 <= k2_outer)
        # Obtain index for (-ki, kj, kk)
        index_neg = index_pos + (-ki + (-(ki != 0) & nyquist))*ℤ[2*slab_size_k]
        # Fill out either indicator or data element
        with unswitch:
            if slab_data is None:
                # Indicator field
                slab_ptr[index_pos] = frac  # imag part stays at 0
                slab_ptr[index_neg] = frac  # imag part stays at 0
            else:
                # Data field
                slab_ptr[index_pos    ] = frac*slab_data_ptr[index_pos    ]
                slab_ptr[index_pos + 1] = frac*slab_data_ptr[index_pos + 1]
                slab_ptr[index_neg    ] = frac*slab_data_ptr[index_neg    ]
                slab_ptr[index_neg + 1] = frac*slab_data_ptr[index_neg + 1]
    # Convert to real space domain grid.
    # Note that we have to nullify the ghosts of all grids as possible
    # appearances of NaN values in the ghost layers otherwise break
    # the code, since 0*NaN != 0.
    fft(slab, 'backward')
    grid = domain_decompose(
        slab,
        buffer_name,
        do_ghost_nullification=True,
    )
    return grid

# Function computing the overlap between a bispectrum shell
# and a (unit) grid cell.
@cython.header(
    # Arguments
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    k_inner='double',
    k_outer='double',
    # Locals
    returns='double',
)
def get_bispec_overlap_cellshell(ki, kj, kk, k_inner, k_outer):
    return (
        + get_bispec_overlap_cellball(ki - 0.5, kj - 0.5, kk - 0.5, k_outer, 1, 1, 1)
        - get_bispec_overlap_cellball(ki - 0.5, kj - 0.5, kk - 0.5, k_inner, 1, 1, 1)
    )

# Below follows several helper functions
# for get_bispec_overlap_cellshell().
@cython.header(
    # Arguments
    x_bgn='double',
    y_bgn='double',
    z_bgn='double',
    r='double',
    Δx='double',
    Δy='double',
    Δz='double',
    # Locals
    frac='double',
    r2='double',
    x_bgn2='double',
    x_end='double',
    y_bgn2='double',
    y_end='double',
    z_bgn2='double',
    z_end='double',
    returns='double',
)
def get_bispec_overlap_cellball(x_bgn, y_bgn, z_bgn, r, Δx, Δy, Δz):
    x_bgn -= (2*x_bgn + Δx)*(x_bgn < -Δx)
    y_bgn -= (2*y_bgn + Δy)*(y_bgn < -Δy)
    z_bgn -= (2*z_bgn + Δz)*(z_bgn < -Δz)
    x_end = x_bgn + Δx
    y_end = y_bgn + Δy
    z_end = z_bgn + Δz
    if 0 < x_end < Δx:
        return (
            + get_bispec_overlap_cellball(0, y_bgn, z_bgn, r, +x_end, Δy, Δz)
            + get_bispec_overlap_cellball(0, y_bgn, z_bgn, r, -x_bgn, Δy, Δz)
        )
    if 0 < y_end < Δy:
        return (
            + get_bispec_overlap_cellball(x_bgn, 0, z_bgn, r, Δx, +y_end, Δz)
            + get_bispec_overlap_cellball(x_bgn, 0, z_bgn, r, Δx, -y_bgn, Δz)
        )
    if 0 < z_end < Δz:
        return (
            + get_bispec_overlap_cellball(x_bgn, y_bgn, 0, r, Δx, Δy,  z_end)
            + get_bispec_overlap_cellball(x_bgn, y_bgn, 0, r, Δx, Δy, -z_bgn)
        )
    x_bgn2 = x_bgn**2
    y_bgn2 = y_bgn**2
    z_bgn2 = z_bgn**2
    r2     = r    **2
    if ℝ[x_bgn2 + y_bgn2] + z_bgn2 >= r2:
        return 0
    x_end2 = x_end**2
    y_end2 = y_end**2
    z_end2 = z_end**2
    if x_end2 + y_end2 + z_end2 <= r2:
        return Δx*Δy*Δz
    frac          = get_bispec_overlap_blockball(x_bgn, y_bgn, z_bgn, r, Δx, y_bgn2, z_bgn2, r2)
    if ℝ[x_bgn2 + y_end2] + z_bgn2 < r2:
        frac     -= get_bispec_overlap_blockball(x_bgn, y_end, z_bgn, r, Δx, y_end2, z_bgn2, r2)
    if ℝ[x_bgn2 + y_bgn2] + z_end2 < r2:
        frac     -= get_bispec_overlap_blockball(x_bgn, y_bgn, z_end, r, Δx, y_bgn2, z_end2, r2)
        if ℝ[x_bgn2 + y_end2] + z_end2 < r2:
            frac += get_bispec_overlap_blockball(x_bgn, y_end, z_end, r, Δx, y_end2, z_end2, r2)
    return frac
@cython.header(
    # Arguments
    x_bgn='double',
    y_bgn='double',
    z_bgn='double',
    r='double',
    Δx='double',
    y_bgn2='double',
    z_bgn2='double',
    r2='double',
    # Locals
    s='double',
    x_end='double',
    returns='double',
)
def get_bispec_overlap_blockball(x_bgn, y_bgn, z_bgn, r, Δx, y_bgn2, z_bgn2, r2):
    s = sqrt(r2 - y_bgn2*(y_bgn > 0) - z_bgn2*(z_bgn > 0))
    x_bgn, x_end = (
        pairmax(x_bgn     , -s),
        pairmin(x_bgn + Δx, +s),
    )
    return (
        + get_bispec_overlap_blockball_indefinite(x_end, y_bgn, z_bgn, r, y_bgn2, z_bgn2, r2)
        - get_bispec_overlap_blockball_indefinite(x_bgn, y_bgn, z_bgn, r, y_bgn2, z_bgn2, r2)
    )
@cython.header(
    # Arguments
    x='double',
    y_bgn='double',
    z_bgn='double',
    r='double',
    y_bgn2='double',
    z_bgn2='double',
    r2='double',
    # Locals
    arctan_xy='double',
    arctan_xyz='double',
    arctan_xz='double',
    arctan_yz='double',
    r8='double',
    sqrt_y='double',
    sqrt_z='double',
    x2='double',
    x4='double',
    xx='double',
    xx_zero='bint',
    yy='double',
    yy_zero='bint',
    ϵ='double',
    returns='double',
)
def get_bispec_overlap_blockball_indefinite(x, y_bgn, z_bgn, r, y_bgn2, z_bgn2, r2):
    x = x*(x <= r) + r*(x > r)
    x2 = x**2
    x4 = x2**2
    r8 = r2**4
    sqrt_y = ℝ[r2 - x2] - y_bgn2
    sqrt_y = sqrt(sqrt_y*(sqrt_y > 0))
    sqrt_z = ℝ[r2 - x2] - z_bgn2
    sqrt_z = sqrt(sqrt_z*(sqrt_z > 0))
    arctan_xy = arctan2(x, sqrt_y)
    arctan_xz = arctan2(x, sqrt_z)
    arctan_yz = arctan2(
        ℝ[sqrt_y*sqrt_z] - ℝ[y_bgn*z_bgn],
        y_bgn*sqrt_z + z_bgn*sqrt_y,
    )
    xx = (
        + x4*ℝ[y_bgn2*z_bgn2]
        + r2*(
            + x4*ℝ[y_bgn2 + z_bgn2]
            + ℝ[2*x2]*y_bgn*(
                + ℝ[y_bgn*z_bgn2]
                - 2*z_bgn*ℝ[sqrt_y*sqrt_z]
            )
            + r2*(
                + x4
                + ℝ[y_bgn2*z_bgn2]
                + r2*(
                    - ℝ[2*x2]
                    - ℝ[y_bgn2 + z_bgn2]
                )
            )
        )
        + r8
    )
    yy = 2*x*r*(
        - x2*(
            + ℝ[y_bgn*z_bgn2]*sqrt_y
            + ℝ[y_bgn2*z_bgn]*sqrt_z
        )
        + r2*(
            + ℝ[y_bgn*sqrt_y]*(ℝ[r2 - x2] - z_bgn2)
            + ℝ[z_bgn*sqrt_z]*(ℝ[r2 - x2] - y_bgn2)
        )
    )
    ϵ = 1e-14*(r8 + 1)
    xx_zero = (-ϵ < xx) & (xx < ϵ)
    yy_zero = (-ϵ < yy) & (yy < ϵ)
    xx = xx*(not xx_zero) - 1e-100*xx_zero
    yy = yy*(not yy_zero) + 1e-200*yy_zero
    arctan_xyz = arctan2(yy, xx)
    return (
        + x*ℝ[y_bgn*z_bgn]
        - 1./3.*x*(ℝ[y_bgn*sqrt_y] + ℝ[z_bgn*sqrt_z])
        + 1./6.*(
            - y_bgn*(ℝ[3*r2] - y_bgn2)*arctan_xy
            - z_bgn*(ℝ[3*r2] - z_bgn2)*arctan_xz
            + x    *(ℝ[3*r2] - x2    )*arctan_yz
            + r    *r2                *arctan_xyz
        )
    )

# Function which given a bispectrum declaration correctly populated
# with all fields will compute its bispectrum.
@cython.header(
    # Arguments
    declaration=object,  # BispecDeclaration
    # Locals
    a='double',
    bin=object,  # BispecBin
    bin_index='Py_ssize_t',
    bins=list,
    bpower='double[::1]',
    bpower_reduced='double[::1]',
    bpower_reduced_ptr='double*',
    bpower_ptr='double*',
    component='Component',
    components=list,
    computation_order='Py_ssize_t[::1]',
    deconvolve='bint',
    do_reduced='bint',
    gridsize='Py_ssize_t',
    gridsizes_upstream=list,
    index_power='Py_ssize_t',
    interlace=str,
    interpolation='int',
    n_mode_power='double',
    n_modes='double[::1]',
    n_modes_bin='double',
    n_modes_power=dict,
    n_modes_ptr='double*',
    normalization='double',
    ordering=object,  # np.ndarray
    power=tuple,
    power_0='double',
    power_1='double',
    power_2='double',
    power_arr='double[::1]',
    power_arr_0='double[::1]',
    power_arr_1='double[::1]',
    power_arr_2='double[::1]',
    power_dict=dict,
    power_ptr_0='double*',
    power_ptr_1='double*',
    power_ptr_2='double*',
    shell=tuple,
    slab='double[:, :, ::1]',
    returns='void',
)
def compute_bispec(declaration):
    if not declaration.do_data:
        return
    # Extract some variables from the bispectrum declaration
    components        = declaration.components
    gridsize          = declaration.gridsize
    interpolation     = declaration.interpolation
    deconvolve        = declaration.deconvolve
    interlace         = declaration.interlace
    bins              = declaration.bins
    n_modes           = declaration.n_modes
    bpower            = declaration.bpower
    do_reduced        = declaration.do_reduced
    n_modes_power     = declaration.n_modes_power
    power             = declaration.power
    bpower_reduced    = declaration.bpower_reduced
    computation_order = declaration.computation_order
    # Begin progress message
    components_str = ', '.join([component.name for component in components])
    if len(components) > 1:
        components_str = f'{{{components_str}}}'
    masterprint(f'Computing bispectrum of {components_str} ...')
    # Interpolate the physical density of all components onto a global
    # grid by first interpolating onto individual upstream grids,
    # transforming to Fourier space and then adding them together.
    gridsizes_upstream = [
        component.bispec_upstream_gridsize
        for component in components
    ]
    slab = interpolate_upstream(
        components, gridsizes_upstream, gridsize, 'ρ', interpolation,
        deconvolve=deconvolve, interlace=interlace, output_space='Fourier',
    )
    # Compute the bispectrum value for each bin
    bpower_ptr = cython.address(bpower[:])
    power_dict = ({} if do_reduced else None)
    for bin_index in range(bpower.shape[0]):
        bin_index = computation_order[bin_index]
        bin = bins[bin_index]
        bpower_ptr[bin_index] = compute_bispec_value(
            gridsize,
            bin.shells,
            power_dict,
            slab,
        )
    # Sum bpower (and power) into the master process
    Reduce(
        sendbuf=(MPI.IN_PLACE if master else bpower),
        recvbuf=(bpower       if master else None),
        op=MPI.SUM,
    )
    if do_reduced:
        power_arr = np.fromiter(
            power_dict.values(),
            dtype=C2np['double'],
        )
        Reduce(
            sendbuf=(MPI.IN_PLACE if master else power_arr),
            recvbuf=(power_arr    if master else None),
            op=MPI.SUM,
        )
        for index_power, shell in enumerate(power_dict.keys()):
            power_dict[shell] = power_arr[index_power]
    # The master process now holds all the information needed
    if not master:
        return
    # We need to transform bpower from being the sum to being the
    # mean, by dividing by n_modes.
    # To completely remove the current normalization of the bpower, we
    # need to divide by the cubed mean of values on the grids/slabs.
    # As we interpolated physical densities ρ to the grids, the mean of
    # all values will be
    #   mean(ρᵢⱼₖ) = ρ_bar = a**(-3(1 + w_eff))*ϱ_bar,
    # summed over all components.
    # As said, we need to divide the bpower by the cube of mean(ρᵢⱼₖ).
    # To now add in a proper normalization, we need to multiply by
    # boxsize**6, resulting in a properly normalized bispectrum in
    # units of unit_length**6.
    # Lastly, n_modes have been normalized by 0.5/gridsize**3.
    # As these are constructed via indicator grids which themselves are
    # constructed analogous to the data grids, we must similarly apply
    # 0.5/gridsize**3 as a normalization here.
    a = universals.a
    normalization = 0
    for component in components:
        normalization += a**(-3*(1 + component.w_eff(a=a)))*component.ϱ_bar
    normalization **= -3
    normalization *= ℝ[0.5*boxsize**6]/gridsize**3
    n_modes_ptr = cython.address(n_modes[:])
    for bin_index in range(bpower.shape[0]):
        n_modes_bin = n_modes_ptr[bin_index]
        if n_modes_bin != 0:
            bpower_ptr[bin_index] *= normalization/n_modes_bin
        else:
            # Empty bin. No bispectrum value.
            bpower_ptr[bin_index] = NaN
    # Similarly normalize power spectra
    # (for use with the reduced bispectrum).
    if do_reduced:
        normalization = 0
        for component in components:
            normalization += a**(-3*(1 + component.w_eff(a=a)))*component.ϱ_bar
        normalization **= -2
        normalization *= ℝ[0.5*boxsize**3]/gridsize**3
        for shell in power_dict.keys():
            n_mode_power = n_modes_power[shell]
            if n_mode_power != 0:
                power_dict[shell] *= normalization/n_mode_power
            else:
                # Empty shell. No power spectrum value.
                power_dict[shell] = NaN
    # Compute the reduced bispectrum
    if do_reduced:
        bpower_reduced_ptr = cython.address(bpower_reduced[:])
        power_arr_0, power_arr_1, power_arr_2 = power
        power_ptr_0 = cython.address(power_arr_0[:])
        power_ptr_1 = cython.address(power_arr_1[:])
        power_ptr_2 = cython.address(power_arr_2[:])
        for bin_index in range(bpower_reduced.shape[0]):
            bin = bins[bin_index]
            ordering = asarray(bin.ordering)
            power_0, power_1, power_2 = asarray([
                power_dict[shell]
                for shell in bin.shells
            ])[ordering]  # order according to (k, t, μ) input
            bpower_reduced_ptr[bin_index] = bpower_ptr[bin_index]/(
                + power_0*power_1
                + power_1*power_2
                + power_2*power_0
            )
            # Fill in power spectrum arrays
            power_ptr_0[bin_index] = power_0
            power_ptr_1[bin_index] = power_1
            power_ptr_2[bin_index] = power_2
    # Bispectrum computation complete
    masterprint('done')

# Function which given a bispectrum declaration correctly populated
# with all fields will compute its (linear, CLASS) tree-level
# matter bispectrum.
@cython.header(
    # Arguments
    declaration=object,  # PowerspecDeclaration
    # Locals
    bin=object,  # BispecBin
    bin_index='Py_ssize_t',
    bins=list,
    bpower_reduced_treelevel='double[::1]',
    bpower_treelevel='double[::1]',
    component='Component',
    components=list,
    components_str=str,
    size='Py_ssize_t',
    k_magnitude_0='double',
    k_magnitude_1='double',
    k_magnitude_2='double',
    k_magnitudes_0='double[::1]',
    k_magnitudes_0_ptr='double*',
    k_magnitudes_1='double[::1]',
    k_magnitudes_1_ptr='double*',
    k_magnitudes_2='double[::1]',
    k_magnitudes_2_ptr='double*',
    returns='void',
)
def compute_bispec_treelevel(declaration):
    if not declaration.do_treelevel:
        return
    # Extract some variables from the bispectrum declaration
    components               = declaration.components
    bins                     = declaration.bins
    bpower_treelevel         = declaration.bpower_treelevel
    bpower_reduced_treelevel = declaration.bpower_reduced_treelevel
    # Begin progress message
    components_str = ', '.join([component.name for component in components])
    if len(components) > 1:
        components_str = f'{{{components_str}}}'
    masterprint(f'Computing tree-level bispectrum of {components_str} ...')
    # Create three arrays of k values corresponding to the bins.
    # As only the master process will compute and store the tree-level
    # bispectrum, only this process needs to know the k values.
    k_magnitudes_0 = k_magnitudes_1 = k_magnitudes_2 = None
    if master:
        size = bpower_treelevel.shape[0]
        k_magnitudes_0 = empty(size, dtype=C2np['double'])
        k_magnitudes_1 = empty(size, dtype=C2np['double'])
        k_magnitudes_2 = empty(size, dtype=C2np['double'])
        k_magnitudes_0_ptr = cython.address(k_magnitudes_0[:])
        k_magnitudes_1_ptr = cython.address(k_magnitudes_1[:])
        k_magnitudes_2_ptr = cython.address(k_magnitudes_2[:])
        for bin_index in range(size):
            bin = bins[bin_index]
            k_magnitude_0 = bin.k
            k_magnitude_1 = bin.t*k_magnitude_0
            k_magnitude_2 = sqrt(
                + k_magnitude_0**2
                + k_magnitude_1**2
                - 2*bin.μ*k_magnitude_0*k_magnitude_1
            )
            k_magnitudes_0_ptr[bin_index] = k_magnitude_0
            k_magnitudes_1_ptr[bin_index] = k_magnitude_1
            k_magnitudes_2_ptr[bin_index] = k_magnitude_2
    # Fill bpower_treelevel (and bpower_reduced_treelevel if allocated)
    # with values of the tree-level bispectrum.
    # Only the master will hold the values.
    get_treelevel_bispec(
        components,
        k_magnitudes_0, k_magnitudes_1, k_magnitudes_2,
        bpower_treelevel, bpower_reduced_treelevel,
    )
    # Done with the tree-level bispectrum computation
    masterprint('done')

# Function for saving bispectra
def save_bispec(declarations, filename):
    column_headings_left = [
        ('do_data', f'k [{unit_length}⁻¹]', 'bins[:].k'),
        ('do_data', f't',                   'bins[:].t'),
        ('do_data', f'μ',                   'bins[:].μ'),
        ('do_data', f'modes',               'n_modes'),
        ('do_data', f'(expected)',          'n_modes_expected'),
    ]
    column_headings_components = [
        ('do_data',                      'component',    f'B [{unit_length}⁶]',  'bpower'),
        ('do_treelevel',                 '(tree-level)', f'B [{unit_length}⁶]',  'bpower_treelevel'),
        ('do_reduced',                   '',             f'Q',                   'bpower_reduced'),
        (('do_reduced', 'do_treelevel'), '(tree-level)', f'Q',                   'bpower_reduced_treelevel'),
        ('do_reduced',                   '',             f'P₁ [{unit_length}³]', 'power[0]'),
        ('do_reduced',                   '',             f'P₂ [{unit_length}³]', 'power[1]'),
        ('do_reduced',                   '',             f'P₃ [{unit_length}³]', 'power[2]'),
    ]
    grouping_func = lambda declaration: (
        len(declaration.bins),
        *(
            + np.max([(bin.k, bin.t, bin.μ) for bin in declaration.bins], axis=0)
            - np.min([(bin.k, bin.t, bin.μ) for bin in declaration.bins], axis=0)
        ),
        hashlib.sha1(str(declaration.bins).encode('utf-8')).hexdigest(),
    )
    fmt_specifications = None
    if not bispec_antialiasing:
        fmt_specifications = {
            'n_modes'         : int,
            'n_modes_expected': int,
        }
    save_polyspec(
        'bispec', declarations, filename,
        column_headings_left, column_headings_components, grouping_func,
        fmt_specifications=fmt_specifications,
    )

# Pure Python function for saving already computed output declarations
# for power and bispectra to a single text file.
def save_polyspec(
    spec, declarations, filename,
    column_headings_left, column_headings_components, grouping_func,
    extra_heading_func=None, extra_heading_fmt=None,
    fmt_specifications=None, reverse=False,
):
    if not master:
        return
    # Discard declarations that should not be saved
    declarations = [declaration for declaration in declarations if declaration.do_data]
    if not declarations:
        return
    spec_printing = ''.join([
        spec.removesuffix('spec'),
        ' '*(spec == 'powerspec'),
        'spectrum' if len(declarations) == 1 else 'spectra',
    ])
    masterprint(f'Saving {spec_printing} to "{filename}" ...')
    # Get specifications for the txt data file
    txt_info = get_txt_info(
        spec,
        declarations,
        column_headings_left,
        column_headings_components,
        grouping_func,
        extra_heading_func,
        fmt_specifications,
    )
    # Closure for iterating over columns in a group of declarations
    def iterate_columns(declaration_group):
        for declaration in declaration_group:
            for (
                do_attr, component_heading, column_heading, attr,
            ) in column_headings_components:
                if all([
                    getattr_nested(declaration, do_attr_i)
                    for do_attr_i in any2list(do_attr)
                ]):
                    yield declaration, do_attr, attr
    # Closure for filling out a column in the 2D data array
    def write_column(attr, arr):
        nonlocal col
        arr = asarray(arr)
        data_col = asarray(arr, dtype=C2np['double'])
        fill = 0.
        if attr_is_float(attr, arr, fmt_specifications):
            fill = NaN
        data[:data_col.shape[0], col] = data_col[::(1 - 2*reverse)]
        data[data_col.shape[0]:, col] = fill
        col += 1
    # Substitute the extra header line if extra_heading_fmt is passed
    header = txt_info.header
    if extra_heading_fmt is not None:
        header_lines = header.split('\n')
        extra_heading = header_lines[len(header_lines) - 2]
        for declaration_group in txt_info.declaration_groups.values():
            for declaration, do_attr, attr in iterate_columns(declaration_group):
                extra_heading = re.sub(
                    *extra_heading_fmt(declaration, do_attr),
                    extra_heading,
                    1,
                )
        header_lines[len(header_lines) - 2] = extra_heading
        header = '\n'.join(header_lines)
    # The top line of the header, stating general information
    header_significant_figures = np.max([
        declaration.significant_figures
        for declaration in declarations
    ])
    topline = unicode(
        f'{spec_printing.capitalize()} from CO𝘕CEPT job {jobid} at t = '
        + f'{{:.{header_significant_figures}g}} '.format(universals.t)
        + f'{unit_time}'
        + (
            f', a = ' + f'{{:.{header_significant_figures}g}}'.format(universals.a)
            if enable_Hubble else ''
        )
    )
    # Though the number of rows in each column may differ, we make all
    # columns the same length by appending NaN's (for floats) or zeros
    # (for ints). This makes it easier to read the data back in.
    # Get a 2D array with the right size for storing all data.
    data = get_buffer((txt_info.n_rows, txt_info.n_cols))
    # Fill in data columns
    col = 0
    for declaration_group in txt_info.declaration_groups.values():
        # Add left columns for this group
        for do_attr, column_heading, attr in column_headings_left:
            for declaration in declaration_group:
                if getattr(declaration, do_attr):
                    break
            else:
                continue
            write_column(attr, getattr_nested(declaration, attr))
        # Add component columns for this group
        for declaration, do_attr, attr in iterate_columns(declaration_group):
            write_column(attr, getattr_nested(declaration, attr))
    # Write to in-memory "file" and extract content as str
    with io.StringIO() as f:
        np.savetxt(
            f,
            data,
            fmt=txt_info.fmt,
            delimiter=txt_info.delimiter,
            header=f'{topline}\n{header}',
            encoding='utf-8',
        )
        file_content = f.getvalue()
    # Remove possible sign on NaN
    file_content = file_content.replace('+nan', ' nan')
    # Write NaN in correct case
    file_lines = file_content.split('\n')
    for row, line in enumerate(file_lines):
        if not line.lstrip().startswith('#'):
            file_lines[row] = line.replace('nan', 'NaN')
    # Enforce accurate delimiter
    horz = unicode('─')
    replacement_char = unicode('�')
    for row, line in enumerate(file_lines):
        if horz in line:
            row_horz = row
            break
    def get_indices_horz(file_lines):
        return [i for i, c in enumerate(file_lines[row_horz]) if c == horz]
    def strip_superfluous_empty_cols(file_lines, n_empty_cols_max):
        for row, line in enumerate(file_lines):
            file_lines[row] = list(line)
        indices_horz = get_indices_horz(file_lines)
        if indices_horz:
            n_empty_cols = 0
            for index_horz in indices_horz:
                for row, line in enumerate(file_lines[row_horz + 1:], row_horz + 1):
                    if len(line) <= index_horz:
                        continue
                    if line[index_horz] != ' ':
                        n_empty_cols = 0
                        break
                else:
                    n_empty_cols += 1
                if n_empty_cols > n_empty_cols_max:
                    # Found empty column to be removed.
                    # Mark characters for removal within in the column,
                    # including the row containing '─'.
                    for row, line in enumerate(file_lines[row_horz:], row_horz):
                        if len(line) <= index_horz:
                            continue
                        line[index_horz] = replacement_char
        file_content = '\n'.join(''.join(line) for line in file_lines)
        file_content = file_content.replace(replacement_char, '')
        file_lines = file_content.split('\n')
        return file_lines
    file_lines = strip_superfluous_empty_cols(file_lines, len(txt_info.delimiter))
    # The horizontal multi-character braces may end slightly too late
    vert = unicode('│')  # used temporarily
    lcrn = unicode('╭')
    rcrn = unicode('╮')
    file_lines[row_horz] = (
        file_lines[row_horz]
        .replace(horz, vert)
        .replace(rcrn, horz)
    )
    file_lines = strip_superfluous_empty_cols(file_lines, 0)
    file_lines[row_horz] += ' '
    file_lines[row_horz] = (
        (file_lines[row_horz] + ' ')
        .replace(vert, horz)
        .replace(f'{horz} ', f'{rcrn} ')
    )
    # Recenter the group (grid size) headings
    indices_horz = get_indices_horz(file_lines)
    if indices_horz:
        indices_corners_bgn = [
            match.start()
            for match in re.finditer(
                lcrn,
                ''.join(file_lines[row_horz]),
            )
        ]
        indices_corners_end = [
            match.end()
            for match in re.finditer(
                rcrn,
                ''.join(file_lines[row_horz]),
            )
        ]
        line = file_lines[row_horz - 1]
        group_headings = re.findall(r'grid size \d+', line)
        line = [('#' if c == '#' else ' ') for c in line]
        for index_corner_bgn, index_corner_end, group_heading in zip(
            indices_corners_bgn, indices_corners_end, group_headings,
        ):
            index = (index_corner_bgn + index_corner_end - len(group_heading))//2
            for i, c in enumerate(group_heading):
                line[index + i] = c
        file_lines[row_horz - 1] = ''.join(line)
    # Strip trailing whitespace
    file_lines = [line.rstrip() for line in file_lines]
    file_content = '\n'.join(file_lines)
    # Save text file to disk
    with open_file(filename, mode='w', encoding='utf-8') as f:
        f.write(file_content)
    masterprint('done')

# Helper function for save_polyspec()
def attr_is_float(attr, arr, fmt_specifications):
    if fmt_specifications is not None and attr in fmt_specifications:
        return fmt_specifications[attr] == float
    else:
        return arr.dtype in (np.float32, np.float64)

# Pure Python function for generating a header, format specifiers,
# delimiters and grouping of output declarations, for a txt data file.
def get_txt_info(
    declaration_type,
    declarations,
    column_headings_left,
    column_headings_components,
    grouping_func,
    extra_heading_func=None,
    fmt_specifications=None,
):
    """The supplied list of declarations should only contain
    declarations for which do_data is True.
    """
    # Look up in the cache
    cache_key = tuple(
        [declaration_type]
        + [tuple(declaration.components) for declaration in declarations]
    )
    txt_info = txt_info_cache.get(cache_key)
    if txt_info:
        # Cached result found.
        # Which components to include in the output may change over time
        # due to components changing their passive/active/terminated
        # state, which in turn change the passed declarations. As the
        # cache key used above depends on the components only,
        # the resulting cached result may hold outdated declaration
        # instances. Update these before returning.
        for declaration_group in txt_info.declaration_groups.values():
            for i, declaration_cached in enumerate(declaration_group):
                for declaration in declarations:
                    if declaration_cached.components == declaration.components:
                        declaration_group[i] = declaration
                        break
        return txt_info
    # A column mapping each component to a number
    components = []
    for declaration in declarations:
        for component in declaration.components:
            if component not in components:
                components.append(component)
    longest_name_size = np.max([len(component.name) for component in components])
    len_i = len(str(len(components) - 1))
    gridsizes_upstream = [
        getattr(component, f'{declaration_type}_upstream_gridsize')
        for component in components
    ]
    len_upstream_gridsize = len(str(np.max(gridsizes_upstream)))
    column_components = ['Components:']
    for i, component in enumerate(components):
        column_components.append(
            f'  {{:>{len_i}}}: {{:<{longest_name_size}}}'
            f'  (upstream grid size {{:>{len_upstream_gridsize}}})'
            .format(
                i,
                component.name,
                gridsizes_upstream[i],
            )
        )
    # Group declarations according to the gridsize
    # as well as the passed grouping_func.
    declaration_groups_unordered = collections.defaultdict(list)
    for key, declarations_iter in itertools.groupby(
        declarations,
        key=(lambda declaration: (declaration.gridsize, ) + grouping_func(declaration)),
    ):
        declaration_groups_unordered[key] += list(declarations_iter)
    declaration_groups = {
        key: declaration_groups_unordered[key]
        for key in sorted(
            declaration_groups_unordered,
            reverse=True,
        )
    }
    # Helper function for obtaining the float format and width
    # given number of significant figures.
    def get_formatting(significant_figures, arr=None):
        flag = '-'
        if arr is not None and np.any(asarray(arr) < 0):
            flag += '+'
        fmt_float = f'%{flag}{{}}.{significant_figures - 1}e'
        width_float = significant_figures + n_chars_nonsignificant
        return fmt_float, width_float
    n_chars_nonsignificant = len(f'{1e+100:.1e}') - 2
    # Determine the headings for each column and their format specifier
    group_spacing = 1
    group_delimiter = ' '*group_spacing
    fmt_int = '%{}u'
    n_rows = 0
    col = 0
    columns_new_group  = []
    components_heading = []
    extras_heading     = []
    columns_heading    = []
    fmt                = []
    for (gridsize, *_), declaration_group in declaration_groups.items():
        columns_new_group.append(col)
        if col > 0:
            # New group with new grid size begins. Insert additional
            # spacing by modifying the last elements of the
            # *s_heading and fmt.
            components_heading.append(components_heading.pop() + group_delimiter)
            if extra_heading_func is not None:
                extras_heading.append(extras_heading.pop() + group_delimiter)
            columns_heading.append(columns_heading.pop() + group_delimiter)
            fmt.append(fmt.pop() + group_delimiter)
        # Add left columns for this group
        for do_attr, column_heading, attr in column_headings_left:
            for declaration in declaration_group:
                if getattr(declaration, do_attr):
                    break
            else:
                continue
            col += 1
            column_heading = unicode(column_heading)
            arr = asarray(getattr_nested(declaration, attr))
            if attr_is_float(attr, arr, fmt_specifications):
                fmt_float, width_float = get_formatting(
                    np.max([declaration.significant_figures for declaration in declaration_group]),
                    arr,
                )
                width = np.max((width_float, len(column_heading) + 2*(col == 1)))
                fmt.append(fmt_float.format(width))
            else:
                width = np.max((maxlen_arr(arr, is_int=True), len(column_heading) + 2*(col == 1)))
                fmt.append(fmt_int.format(width))
            components_heading.append(' '*(width - 2*(col == 1)))
            if extra_heading_func is not None:
                extras_heading.append(' '*(width - 2*(col == 1)))
            extra_spacing = width - len(column_heading) - 2*(col == 1)
            columns_heading.append(
                ' '*(extra_spacing//2) + column_heading + ' '*(extra_spacing - extra_spacing//2)
            )
            # Keep track of the largest number of rows
            if arr.shape[0] > n_rows:
                n_rows = arr.shape[0]
        # Add component columns for this group
        for declaration in declaration_group:
            extra_heading_str = ''
            if extra_heading_func is not None:
                extra_heading_str = unicode(extra_heading_func(declaration))
            for (
                do_attr, component_heading, column_heading, attr,
            ) in column_headings_components:
                if not all([
                    getattr_nested(declaration, do_attr_i)
                    for do_attr_i in any2list(do_attr)
                ]):
                    continue
                col += 1
                component_heading = unicode(component_heading)
                column_heading    = unicode(column_heading)
                if component_heading == 'component':
                    component_heading = get_integerset_strrep([
                        components.index(component)
                        for component in declaration.components
                    ])
                    if len(declaration.components) == 1:
                        component_heading = f'component {component_heading}'
                    else:
                        component_heading = f'components {{{component_heading}}}'
                arr = asarray(getattr_nested(declaration, attr))
                if attr_is_float(attr, arr, fmt_specifications):
                    fmt_component, width_component = get_formatting(
                        declaration.significant_figures,
                        arr,
                    )
                else:
                    fmt_component = fmt_int
                    width_component = maxlen_arr(getattr_nested(declaration, attr), is_int=True)
                extra_heading = ''
                if extra_heading_func is not None and '{}' in extra_heading_str:
                    extra_heading_significant_figures = (
                          width_component
                        - len(extra_heading_str.replace('{}', ''))
                        - n_chars_nonsignificant
                    )
                    if extra_heading_significant_figures < 2:
                        extra_heading_significant_figures = 2
                    size = width_float - len(extra_heading_str)
                    size = np.max((
                        width_component - len(extra_heading_str),
                        len(
                            f'{{:<{size}.{extra_heading_significant_figures - 1}e}}'
                            .format(1e+100)
                        ),
                    ))
                    extra_heading = (
                        extra_heading_str
                        .format(f'{{:<{size}.{extra_heading_significant_figures - 1}e}}')
                    )
                width = np.max((
                    width_component,
                    len(column_heading),
                    len(component_heading),
                    len(extra_heading.format(1e+100)),
                ))
                extra_spacing = width - len(component_heading)
                components_heading.append(
                    ' '*(extra_spacing//2) + component_heading
                    + ' '*(extra_spacing - extra_spacing//2)
                )
                extra_spacing = width - len(extra_heading.format(1e+100))
                if extra_heading_func is not None:
                    extras_heading.append(
                        ' '*(extra_spacing//2) + extra_heading
                        + ' '*(extra_spacing - extra_spacing//2)
                    )
                extra_spacing = width - len(column_heading)
                columns_heading.append(
                    ' '*(extra_spacing//2) + column_heading
                    + ' '*(extra_spacing - extra_spacing//2)
                )
                extra_spacing = width - width_component
                fmt.append(' '*(extra_spacing//2) + fmt_component.format(width - extra_spacing//2))
    # Record the number of columns
    n_cols = col
    # Construct group header
    group_header_underlines = []
    delimiter = ' '*2
    def append_underline():
        group_header_underlines.append(unicode('╭' + unicode('─')*(width - 2) + '╮'))
    for col, column_heading in enumerate(columns_heading):
        if col in columns_new_group:
            if col > 0:
                width -= group_spacing
                append_underline()
            width = len(column_heading)
        else:
            width += len(delimiter) + len(column_heading)
    append_underline()
    group_headers = []
    for (gridsize, *_), group_header_underline in zip(
        declaration_groups, group_header_underlines,
    ):
        group_heading = f'grid size {gridsize}'
        extra_spacing = len(group_header_underline) - len(group_heading)
        group_headers.append(
            ' '*(extra_spacing//2) + group_heading + ' '*(extra_spacing - extra_spacing//2)
        )
    # Put it all together to a collective header string
    header_lines = [
        '',
        *column_components,
        '',
        (delimiter + group_delimiter).join(group_headers),
        (delimiter + group_delimiter).join(group_header_underlines),
        delimiter.join(components_heading),
    ]
    if extra_heading_func is not None:
        header_lines.append(delimiter.join(extras_heading))
    header_lines.append(delimiter.join(columns_heading))
    header = '\n'.join(header_lines)
    # Store in cache and return
    txt_info = TxtInfo(
        header, fmt, delimiter, declaration_groups, n_rows, n_cols,
    )
    txt_info_cache[cache_key] = txt_info
    return txt_info
# Cache and type used by the get_txt_info() function
cython.declare(txt_info_cache=dict)
txt_info_cache = {}
TxtInfo = collections.namedtuple(
    'TxtInfo',
    ('header', 'fmt', 'delimiter', 'declaration_groups', 'n_rows', 'n_cols'),
)

# Helper function for get_txt_info()
@cython.header(
    # Arguments
    arr=object,  # np.ndarray
    is_int='bint',
    # Locals
    el=object,
    maxlen='Py_ssize_t',
    n='Py_ssize_t',
    returns='Py_ssize_t',
)
def maxlen_arr(arr, is_int=False):
    maxlen = 0
    for el in arr:
        with unswitch:
            if is_int:
                el = int(el)
        n = len(str(el))
        if n > maxlen:
            maxlen = n
    return maxlen

# Function which can measure different quantities of a passed component
@cython.header(
    # Arguments
    component='Component',
    quantity=str,
    communicate='bint',
    # Locals
    J_over_ϱ_plus_𝒫_2_i='double',
    J_over_ϱ_plus_𝒫_2_max='double',
    J_over_ϱ_2_i='double',
    J_over_ϱ_2_max='double',
    Jx_mv='double[:, :, ::1]',
    Jx_ptr='double*',
    Jy_mv='double[:, :, ::1]',
    Jy_ptr='double*',
    Jz_mv='double[:, :, ::1]',
    Jz_ptr='double*',
    N='Py_ssize_t',
    N_elements='Py_ssize_t',
    Vcell='double',
    a='double',
    diff_backward='double[:, :, ::1]',
    diff_forward='double[:, :, ::1]',
    diff_max='double[::1]',
    diff_max_dim='double',
    diff_size='double',
    dim='int',
    fluidscalar='FluidScalar',
    h='double',
    i='Py_ssize_t',
    index='Py_ssize_t',
    indexʳ='Py_ssize_t',
    indexˣ='Py_ssize_t',
    indexˣʸᶻ='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    mom='double*',
    mom2='double',
    mom2_max='double',
    mom2_i='double',
    mom_i=object,  # decimal.Decimal
    momxˣ='double*',
    momyˣ='double*',
    momzˣ='double*',
    names=list,
    t='double',
    v_rms='double',
    v_max='double',
    w='double',
    w_eff='double',
    Δdiff='double',
    Δdiff_max='double[::1]',
    Δdiff_max_dim='double',
    Δdiff_max_list=list,
    Δdiff_max_normalized_list=list,
    ΣJ_over_ϱ_plus_𝒫_2='double',
    Σmass='double',
    Σmom='double[::1]',
    Σmom_dim=object,  # decimal.Decimal
    Σmom2_dim=object,  # decimal.Decimal
    Σϱ='double',
    Σϱ2='double',
    ϱ='FluidScalar',
    ϱ_bar='double',
    ϱ_min='double',
    ϱ_mv='double[:, :, ::1]',
    ϱ_noghosts=object, # np.ndarray
    ϱ_ptr='double*',
    σ2mom_dim=object,  # decimal.Decimal
    σ2ϱ='double',
    σmom='double[::1]',
    σmom_dim=object,  # decimal.Decimal
    σϱ='double',
    𝒫_mv='double[:, :, ::1]',
    𝒫_ptr='double*',
    ᐁgrid_dim='double[:, :, ::1]',
    returns=object,  # double or tuple
)
def measure(component, quantity, communicate=True):
    """Implemented quantities are:
    'v_max'
    'v_rms'
    'momentum'
    'ϱ'              (fluid quantity)
    'mass'           (fluid quantity)
    'discontinuity'  (fluid quantity)
    """
    t = universals.t
    a = universals.a
    # Extract variables
    N = (component.N if communicate else component.N_local)
    N_elements = (component.gridsize**3 if communicate else component.size_noghosts)
    Vcell = boxsize**3/N_elements
    w     = component.w    (a=a)
    w_eff = component.w_eff(a=a)
    mom   = component.mom
    momxˣ = component.momxˣ
    momyˣ = component.momyˣ
    momzˣ = component.momzˣ
    ϱ = component.ϱ
    ϱ_noghosts = asarray(ϱ.grid_noghosts)
    # Quantities exhibited by both particle and fluid components
    if quantity == 'v_max':
        # The propagation speed of information in
        # comoving coordinates is
        # v = c*sqrt(w)/a + ẋ, ẋ = dx/dt = u/a,
        # where u is the peculiar velocity.
        # For fluids we have
        # ϱ = a**(3*(1 + w_eff))ρ, J = a**4*(ρ + c⁻²P)u,
        # and so
        # u = a**(-4)*J/(ρ + c⁻²P)
        #   = a**(3*w_eff - 1)*J/(ϱ + c⁻²𝒫),
        # and then
        # v = c*sqrt(w)/a + a**(3*w_eff - 2)*J/(ϱ + c⁻²𝒫),
        # where c*sqrt(w) is an approximation for the local sound speed.
        # For particles we have w = 0 and ẋ = mom/(a**2*m), and so
        # v = mom/(a**2*mass).
        # In the case of decaying (matter) particles, the mass at time a
        # is really a**(-3*w_eff)*mass, and so we get
        # v = mom/(a**(2 - 3*w_eff)*mass)
        if component.representation == 'particles':
            mom2_max = 0
            for indexˣ in range(0, 3*component.N_local, 3):
                mom2_i = momxˣ[indexˣ]**2 + momyˣ[indexˣ]**2 + momzˣ[indexˣ]**2
                if mom2_i > mom2_max:
                    mom2_max = mom2_i
            if communicate:
                mom2_max = allreduce(mom2_max, op=MPI.MAX)
            v_max = sqrt(mom2_max)/(a**(2 - 3*w_eff)*component.mass)
        elif component.representation == 'fluid':
            if (    component.boltzmann_order == -1
                or (component.boltzmann_order == 0 and component.boltzmann_closure == 'truncate')
            ):
                # Without J as a fluid variable,
                # no explicit velocity exists.
                v_max = 0
            elif component.boltzmann_order == 0 and component.boltzmann_closure == 'class':
                # With J as a linear fluid variable, we only need to
                # consider one of its components. Also, the P = wρ
                # approximation is guaranteed to be enabled.
                ϱ_ptr  = component.ϱ .grid
                Jx_ptr = component.Jx.grid
                J_over_ϱ_2_max = 0
                for index in range(component.size):
                    J_over_ϱ_2_i = (Jx_ptr[index]/ϱ_ptr[index])**2
                    if J_over_ϱ_2_i > J_over_ϱ_2_max:
                        J_over_ϱ_2_max = J_over_ϱ_2_i
                if communicate:
                    J_over_ϱ_2_max = allreduce(J_over_ϱ_2_max, op=MPI.MAX)
                J_over_ϱ_plus_𝒫_2_max = 3*J_over_ϱ_2_max/(1 + w)**2
                v_max = a**(3*w_eff - 2)*sqrt(J_over_ϱ_plus_𝒫_2_max)
                # Since no non-linear evolution happens for J, the Euler
                # equation and hence the gradient of the pressure will
                # never be computed. This means that sound waves
                # cannot form, and so we do not need to take the sound
                # speed into account.
            else:
                # J is non-linear
                ϱ_ptr  = component.ϱ .grid
                𝒫_ptr  = component.𝒫 .grid
                Jx_ptr = component.Jx.grid
                Jy_ptr = component.Jy.grid
                Jz_ptr = component.Jz.grid
                J_over_ϱ_plus_𝒫_2_max = 0
                for index in range(component.size):
                    J_over_ϱ_plus_𝒫_2_i = (
                        (Jx_ptr[index]**2 + Jy_ptr[index]**2 + Jz_ptr[index]**2)
                        /(ϱ_ptr[index] + ℝ[light_speed**(-2)]*𝒫_ptr[index])**2
                    )
                    if J_over_ϱ_plus_𝒫_2_i > J_over_ϱ_plus_𝒫_2_max:
                        J_over_ϱ_plus_𝒫_2_max = J_over_ϱ_plus_𝒫_2_i
                if communicate:
                    J_over_ϱ_plus_𝒫_2_max = allreduce(J_over_ϱ_plus_𝒫_2_max, op=MPI.MAX)
                v_max = a**(3*w_eff - 2)*sqrt(J_over_ϱ_plus_𝒫_2_max)
                # Add the sound speed. When the P=wρ approximation is
                # False, the sound speed is non-global and given by the
                # square root of δ𝒫/δϱ. However, constructing δ𝒫/δϱ
                # locally from the ϱ and 𝒫 grids leads to large
                # numerical errors. Regardless of whether the P=wρ
                # approximation is used or not, we simply use the
                # global sound speed.
                v_max += light_speed*sqrt(w)/a
        return v_max
    elif quantity == 'v_rms':
        if component.representation == 'particles':
            mom2 = 0
            for indexʳ in range(3*component.N_local):
                mom2 += mom[indexʳ]**2
            if communicate:
                mom2 = allreduce(mom2, op=MPI.SUM)
            v_rms = sqrt(mom2/N)/(a**(2 - 3*component.w_eff(a=a))*component.mass)
        elif component.representation == 'fluid':
            if (    component.boltzmann_order == -1
                or (component.boltzmann_order == 0 and component.boltzmann_closure == 'truncate')
            ):
                # Without J as a fluid variable, no velocity exists
                v_rms = 0
            elif component.boltzmann_order == 0 and component.boltzmann_closure == 'class':
                # With J as a linear fluid variable, we only need to
                # consider one of its components. Also, the P = wρ
                # approximation is guaranteed to be enabled.
                ϱ_mv  = component.ϱ .grid_mv
                Jx_mv = component.Jx.grid_mv
                ΣJ_over_ϱ_plus_𝒫_2 = 0
                for         i in range(nghosts, ℤ[component.shape[0] - nghosts]):
                    for     j in range(nghosts, ℤ[component.shape[1] - nghosts]):
                        for k in range(nghosts, ℤ[component.shape[2] - nghosts]):
                            ΣJ_over_ϱ_plus_𝒫_2 += 3*(Jx_mv[i, j, k]/(ϱ_mv[i, j, k]*(1 + w)))**2
                if communicate:
                    ΣJ_over_ϱ_plus_𝒫_2 = allreduce(ΣJ_over_ϱ_plus_𝒫_2, op=MPI.SUM)
                v_rms = a**(3*w_eff - 2)*sqrt(ΣJ_over_ϱ_plus_𝒫_2/N_elements)
                # Since no non-linear evolution happens for J, the Euler
                # equation and hence the gradient of the pressure will
                # never be computed. This means that sound waves
                # cannot form, and so we do not need to take the sound
                # speed into account.
            else:
                # J is non-linear
                ϱ_mv  = component.ϱ .grid_mv
                𝒫_mv  = component.𝒫 .grid_mv
                Jx_mv = component.Jx.grid_mv
                Jy_mv = component.Jy.grid_mv
                Jz_mv = component.Jz.grid_mv
                ΣJ_over_ϱ_plus_𝒫_2 = 0
                for         i in range(nghosts, ℤ[component.shape[0] - nghosts]):
                    for     j in range(nghosts, ℤ[component.shape[1] - nghosts]):
                        for k in range(nghosts, ℤ[component.shape[2] - nghosts]):
                            ΣJ_over_ϱ_plus_𝒫_2 += (
                                (Jx_mv[i, j, k]**2 + Jy_mv[i, j, k]**2 + Jz_mv[i, j, k]**2)
                                /(ϱ_mv[i, j, k] + ℝ[light_speed**(-2)]*𝒫_mv[i, j, k])**2
                            )
                if communicate:
                    ΣJ_over_ϱ_plus_𝒫_2 = allreduce(ΣJ_over_ϱ_plus_𝒫_2, op=MPI.SUM)
                v_rms = a**(3*w_eff - 2)*sqrt(ΣJ_over_ϱ_plus_𝒫_2/N_elements)
                # Add the sound speed. When the P=wρ approximation is
                # False, the sound speed is non-global and given by the
                # square root of δ𝒫/δϱ. However, constructing δ𝒫/δϱ
                # locally from the ϱ and 𝒫 grids leads to large
                # numerical errors. Regardless of whether the P=wρ
                # approximation is used or not, we simply use the
                # global sound speed.
                v_rms += light_speed*sqrt(w)/a
        return v_rms
    elif quantity == 'momentum':
        Σmom = empty(3, dtype=C2np['double'])
        σmom = empty(3, dtype=C2np['double'])
        # As the momenta should sum to ~0, floating-point inaccuracies
        # become highly visible. To counteract this, we carry out the
        # computation with increased precision.
        with decimal.localcontext(prec=2*17):
            if component.representation == 'particles':
                # Total momentum and momentum spread of all particles,
                # for each dimension.
                for dim in range(3):
                    Σmom_dim = Σmom2_dim = decimal.Decimal(0)
                    # Add up local particle momenta
                    for indexˣʸᶻ in range(dim, 3*component.N_local, 3):
                        mom_i = decimal.Decimal(mom[indexˣʸᶻ])
                        Σmom_dim  += mom_i
                        Σmom2_dim += mom_i**2
                    # Add up local particle momenta sums
                    if communicate:
                        Σmom_dim  = allreduce(Σmom_dim,  op=MPI.SUM)
                        Σmom2_dim = allreduce(Σmom2_dim, op=MPI.SUM)
                    # Compute global standard deviation
                    σ2mom_dim = Σmom2_dim/N - (Σmom_dim/N)**2
                    if σ2mom_dim < 0:
                        # Negative (about -machine_ϵ) σ² can happen due
                        # to round-off errors.
                        σ2mom_dim = 0
                    else:
                        σmom_dim = σ2mom_dim.sqrt()
                    # Pack results
                    Σmom[dim] = float(Σmom_dim)
                    σmom[dim] = float(σmom_dim)
            elif component.representation == 'fluid':
                # Total momentum of all fluid elements,
                # for each dimension.
                # Here the definition of momenta is chosen as
                #   J*Vcell = (a**4*(ρ + c⁻²P))*Vcell
                #           = (V_phys*(ρ + c⁻²P))*a*u,
                # which reduces to mass*a*u for pressureless fluids
                # and so it is in correspondence with the momentum
                # definition for particles.
                for dim, fluidscalar in enumerate(component.J):
                    Σmom_dim = Σmom2_dim = decimal.Decimal(0)
                    for el in asarray(fluidscalar.grid_noghosts).flat:
                        el = decimal.Decimal(el)
                        Σmom_dim += el
                        Σmom2_dim += el**2
                    # Total dim'th momentum of all fluid elements
                    Σmom_dim *= decimal.Decimal(Vcell)
                    # Total dim'th momentum squared of all fluid elements
                    Σmom2_dim *= decimal.Decimal(Vcell)**2
                    # Add up local fluid element momenta sums
                    if communicate:
                        Σmom_dim  = allreduce(Σmom_dim,  op=MPI.SUM)
                        Σmom2_dim = allreduce(Σmom2_dim, op=MPI.SUM)
                    # Compute global standard deviation
                    σ2mom_dim = Σmom2_dim/N_elements - (Σmom_dim/N_elements)**2
                    if σ2mom_dim < 0:
                        # Negative (about -machine_ϵ) σ² can happen due
                        # to round-off errors.
                        σ2mom_dim = 0
                    else:
                        σmom_dim = σ2mom_dim.sqrt()
                    # Pack results
                    Σmom[dim] = float(Σmom_dim)
                    σmom[dim] = float(σmom_dim)
        return Σmom, σmom
    # Fluid quantities
    elif quantity == 'ϱ':
        # Compute mean(ϱ), std(ϱ), min(ϱ)
        if component.representation == 'particles':
            # Particle components have no ϱ
            abort(
                f'The measure function was called with {component.name} and '
                f'quantity=\'ϱ\', but particle components do not have ϱ'
            )
        elif component.representation == 'fluid':
            # Total ϱ of all fluid elements
            Σϱ = np.sum(ϱ_noghosts)
            # Total ϱ² of all fluid elements
            Σϱ2 = np.sum(ϱ_noghosts**2)
            # Add up local sums
            if communicate:
                Σϱ  = allreduce(Σϱ,  op=MPI.SUM)
                Σϱ2 = allreduce(Σϱ2, op=MPI.SUM)
            # Compute mean value of ϱ
            ϱ_bar = Σϱ/N_elements
            # Compute global standard deviation
            σ2ϱ = Σϱ2/N_elements - ϱ_bar**2
            if σ2ϱ < 0:
                # Negative (about -machine_ϵ) σ² can happen due
                # to round-off errors.
                σ2ϱ = 0
            σϱ = sqrt(σ2ϱ)
            # Compute minimum value of ϱ
            ϱ_min = np.min(ϱ_noghosts)
            if communicate:
                ϱ_min = allreduce(ϱ_min, op=MPI.MIN)
        return ϱ_bar, σϱ, ϱ_min
    elif quantity == 'mass':
        if component.representation == 'particles':
            # Any change in the mass of particle a component is absorbed
            # into w_eff(a).
            Σmass = a**(-3*w_eff)*N*component.mass
        elif component.representation == 'fluid':
            # Total ϱ of all fluid elements
            Σϱ = np.sum(ϱ_noghosts)
            # Add up local sums
            if communicate:
                Σϱ = allreduce(Σϱ, op=MPI.SUM)
            # The total mass is
            # Σmass = (a**3*Vcell)*Σρ
            # where a**3*Vcell is the proper volume and Σρ is the sum of
            # proper densities. In terms of the fluid variable
            # ϱ = a**(3*(1 + w_eff))*ρ, the total mass is then
            # Σmass = a**(-3*w_eff)*Vcell*Σϱ.
            # Note that the total mass is generally constant.
            Σmass = a**(-3*w_eff)*Vcell*Σϱ
        return Σmass
    elif quantity == 'discontinuity':
        if component.representation == 'particles':
            # Particle components have no discontinuity
            abort(
                f'The measure function was called with {component.name} and '
                f'quantity=\'discontinuity\', which is not applicable to particle components'
            )
        elif component.representation == 'fluid':
            # Lists to store results which will be returned
            names = []
            Δdiff_max_normalized_list = []
            Δdiff_max_list = []
            # The grid spacing in physical units
            h = boxsize/component.gridsize
            # Find the maximum discontinuity in each fluid grid
            for fluidscalar in component.iterate_fluidscalars():
                # Store the name of the fluid scalar
                names.append(str(fluidscalar))
                # Communicate ghost points of the grid
                communicate_ghosts(fluidscalar.grid_mv, '=')
                # Differentiate the grid in all three directions via
                # both forward and backward difference. For each
                # direction, save the largest difference between
                # the two. Also save the largest differential in
                # each direction.
                Δdiff_max = empty(3, dtype=C2np['double'])
                diff_max = empty(3, dtype=C2np['double'])
                for dim in range(3):
                    # Do the differentiations
                    ᐁgrid_dim = diff_domaingrid(
                        fluidscalar.grid_mv, dim, 1, h, 0, direction='forward',
                    )
                    diff_forward = ᐁgrid_dim[
                        nghosts:(ᐁgrid_dim.shape[0] - nghosts),
                        nghosts:(ᐁgrid_dim.shape[1] - nghosts),
                        nghosts:(ᐁgrid_dim.shape[2] - nghosts),
                    ]
                    ᐁgrid_dim = diff_domaingrid(
                        fluidscalar.grid_mv, dim, 1, h, 1, direction='backward',
                    )
                    diff_backward = ᐁgrid_dim[
                        nghosts:(ᐁgrid_dim.shape[0] - nghosts),
                        nghosts:(ᐁgrid_dim.shape[1] - nghosts),
                        nghosts:(ᐁgrid_dim.shape[2] - nghosts),
                    ]
                    # Find the largest difference between the results of
                    # the forward and backward difference,
                    Δdiff_max_dim = 0
                    diff_max_dim = 0
                    for         i in range(ℤ[ϱ_noghosts.shape[0]]):
                        for     j in range(ℤ[ϱ_noghosts.shape[1]]):
                            for k in range(ℤ[ϱ_noghosts.shape[2]]):
                                # The maximum difference of the two differentials
                                Δdiff = abs(diff_forward[i, j, k] - diff_backward[i, j, k])
                                if Δdiff > Δdiff_max_dim:
                                    Δdiff_max_dim = Δdiff
                                # The maximum differential
                                diff_size = abs(diff_forward[i, j, k])
                                if diff_size > diff_max_dim:
                                    diff_max_dim = diff_size
                                diff_size = abs(diff_backward[i, j, k])
                                if diff_size > diff_max_dim:
                                    diff_max_dim = diff_size
                    # Use the global maxima
                    if communicate:
                        Δdiff_max_dim = allreduce(Δdiff_max_dim, op=MPI.MAX)
                        diff_max_dim  = allreduce(diff_max_dim,  op=MPI.MAX)
                    # Pack results into lists
                    Δdiff_max[dim] = Δdiff_max_dim
                    diff_max[dim] = diff_max_dim
                Δdiff_max_list.append(Δdiff_max)
                # Maximum discontinuity (difference between forward and
                # backward difference) normalized according to
                # the largest slope.
                Δdiff_max_normalized_list.append(asarray(
                    [
                        Δdiff_max[dim]/diff_max[dim] if Δdiff_max[dim] > 0 else 0
                        for dim in range(3)
                    ],
                    dtype=C2np['double'],
                ))
        return names, Δdiff_max_list, Δdiff_max_normalized_list
    elif master:
        abort(
            f'The measure function was called with '
            f'quantity=\'{quantity}\', which is not implemented'
        )

