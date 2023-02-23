# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2021 Jeppe Mosgaard Dakin.
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
cimport('from communication import communicate_ghosts, get_buffer')
cimport('from graphics import get_output_declarations, plot_powerspec')
cimport('from linear import get_linear_powerspec')
cimport(
    'from mesh import          '
    '    diff_domaingrid,      '
    '    fourier_loop,         '
    '    interpolate_upstream, '
)



# Top-level function for computing, plotting and saving power spectra
@cython.header(
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
        compute_powerspec_linear(declaration)
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
    declaration=object,  # PowerspecDeclaration
    declarations=list,
    index='Py_ssize_t',
    k2_max='Py_ssize_t',
    k_bin_centers='double[::1]',
    k_bin_indices='Py_ssize_t[::1]',
    n_modes='Py_ssize_t[::1]',
    n_modes_max='Py_ssize_t',
    power='double[::1]',
    power_linear='double[::1]',
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
    for index, declaration in enumerate(declarations):
        # Get bin information
        k2_max, k_bin_indices, k_bin_centers, n_modes, n_modes_max = get_powerspec_bins(
            declaration.gridsize,
            declaration.k_max,
            declaration.binsize,
        )
        # Allocate arrays for storing the power
        power = empty(bcast(k_bin_centers.shape[0] if master else None), dtype=C2np['double'])
        power_linear = (asarray(power).copy() if declaration.do_linear else None)
        # Replace old declaration with a new, fully populated one
        declaration = declaration._replace(
            k2_max=k2_max,
            k_bin_indices=k_bin_indices,
            k_bin_centers=k_bin_centers,
            n_modes=n_modes,
            n_modes_max=n_modes_max,
            power=power,
            power_linear=power_linear,
        )
        declarations[index] = declaration
    # Store declarations in cache and return
    powerspec_declarations_cache[cache_key] = declarations
    return declarations
# Cache used by the get_powerspec_declarations() function
cython.declare(powerspec_declarations_cache=dict)
powerspec_declarations_cache = {}
# Create the PowerspecDeclaration type
fields = (
    'components', 'do_data', 'do_linear', 'do_plot', 'gridsize',
    'interpolation', 'deconvolve', 'interlace',
    'k2_max', 'k_max', 'binsize', 'tophat', 'significant_figures',
    'k_bin_indices', 'k_bin_centers', 'n_modes', 'n_modes_max', 'power', 'power_linear',
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
    binsize=dict,
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
    k_magnitude='double',
    k_min='double',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    mask=object,  # boolean np.ndarray
    n_modes='Py_ssize_t[::1]',
    n_modes_fine='Py_ssize_t[::1]',
    n_modes_max='Py_ssize_t',
    nyquist='Py_ssize_t',
    powerspec_bins=tuple,
    θ='double',
    returns=tuple,
)
def get_powerspec_bins(gridsize, k_max, binsize):
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
    cache_key = (gridsize, k_max, tuple(binsize.items()))
    powerspec_bins = powerspec_bins_cache.get(cache_key)
    if powerspec_bins:
        return powerspec_bins
    # Minimum value of k (physical)
    k_min = ℝ[2*π/boxsize]
    # Maximum value of k (physical) and k² (grid units)
    nyquist = gridsize//2
    if isinstance(k_max, str):
        k_max = k_max.replace('nyquist' , f'({2*π/boxsize*nyquist})')
        k_max = k_max.replace('gridsize', f'({gridsize})')
        k_max = eval_unit(k_max, units_dict)
    if k_max < k_min:
        masterwarn(
            f'Power spectrum k_max was set to {k_max} {unit_length}⁻¹ '
            f'< k_min = 2π/boxsize = {k_min} {unit_length}⁻¹. '
            f'Setting k_max = k_min.'
        )
        k_max = k_min
    k2_max = int(round((k_max/ℝ[2*π/boxsize])**2))
    # No need for k2_max to be larger than the largest possible mode
    k2_max = pairmin(k2_max, 3*nyquist**2)
    # Correct k_max
    k_max = ℝ[2*π/boxsize]*sqrt(k2_max)
    # Get k of bin centres using a running bin size
    # as specified by binsize.
    k_bin_centers = construct_k_bin_centers(k_min, k_max, binsize, gridsize, nyquist)
    # Construct array mapping k2 (grid units) to bin index
    k_bin_indices = empty(1 + k2_max, dtype=C2np['Py_ssize_t'])
    k_bin_indices[0] = 0
    for k2 in range(1, k_bin_indices.shape[0]):
        k_magnitude = ℝ[2*π/boxsize]*sqrt(k2)
        # Find index of closest (in linear distance) bin centre
        index = np.searchsorted(k_bin_centers, k_magnitude, 'left')
        if index == ℤ[k_bin_centers.shape[0]]:
            index -= 1
        elif index != 0:
            dist_left  = k_magnitude - k_bin_centers[index - 1]
            dist_right = k_bin_centers[index] - k_magnitude
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
    n_modes_max = 0
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
            n_modes_max,
        )
        return powerspec_bins_cache[cache_key]
    # Redefine k_bin_centers so that each element is the mean of all the
    # k values that falls within the bin, using the multiplicity
    # (n_modes_fine) as weight. Simultaneously construct n_modes from
    # n_modes_fine, where n_modes is just like n_modes_fine, but
    # counting the multiplicity of the bins, rather than the individual
    # k² elements. Finally, we also find the n_modes_max, the largest
    # value in n_modes.
    k_bin_centers[:] = 0
    n_modes = zeros(k_bin_centers.shape[0], dtype=C2np['Py_ssize_t'])
    for k2 in range(1, n_modes_fine.shape[0]):
        if ℤ[n_modes_fine[k2]] == 0:
            continue
        k_magnitude = ℝ[2*π/boxsize]*sqrt(k2)
        k_bin_index = k_bin_indices[k2]
        n_modes[k_bin_index] += ℤ[n_modes_fine[k2]]
        k_bin_centers[k_bin_index] += ℤ[n_modes_fine[k2]]*k_magnitude
    for k_bin_index in range(k_bin_centers.shape[0]):
        if ℤ[n_modes[k_bin_index]] > 0:
            k_bin_centers[k_bin_index] /= ℤ[n_modes[k_bin_index]]
            if ℤ[n_modes[k_bin_index]] > n_modes_max:
                n_modes_max = ℤ[n_modes[k_bin_index]]
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
        n_modes_max,
    )
    powerspec_bins_cache[cache_key] = powerspec_bins
    return powerspec_bins
# Cache used by the get_powerspec_bins() function
cython.declare(powerspec_bins_cache=dict)
powerspec_bins_cache = {}

# Helper function to get_powerspec_bins()
def construct_k_bin_centers(k_min, k_max, binsize, gridsize, nyquist):
    import scipy.interpolate
    # A bin size below binsize_min is guaranteed to never bin
    # separate k² together in the same bin, and so binsize_min is the
    # smallest bin size allowed.
    binsize_min = (
        0.5*(1 - 1e-2)*ℝ[2*π/boxsize]
        *(sqrt(3*nyquist**2 + 1) - sqrt(3*nyquist**2))
    )
    # Evaluate str keys in the binsize dict
    binsize_float = {}
    for k, val in binsize.items():
        if isinstance(k, str):
            k = k.replace('nyquist' , f'({2*π/boxsize*nyquist})')
            k = k.replace('gridsize', f'({gridsize})')
            k = k.replace('k_min'   , f'({k_min})')
            k = k.replace('k_max'   , f'({k_max})')
            k = eval_unit(k, units_dict)
        binsize_float[k] = np.max((val, binsize_min))
    binsize = binsize_float
    if len(binsize) == 1:
        binsize.update({k + 1: val for k, val in binsize.items()})
    # Create linear spline mapping k to bin size
    binsize_interp = lambda k, *, f=scipy.interpolate.interp1d(
        tuple(binsize.keys()),
        tuple(binsize.values()),
        'linear',
        bounds_error=False,
        fill_value=(
            binsize[np.min(tuple(binsize.keys()))],
            binsize[np.max(tuple(binsize.keys()))],
        ),
    ): float(f(k))
    # Construct k_bin_centers using a running bin size
    k_bin_centers = []
    k_bin_size = binsize_interp(k_min)
    k_bin_right = k_min
    while k_bin_right < k_max:
        k_bin_left = k_bin_right
        k_bin_size = binsize_interp(k_bin_left + 0.5*k_bin_size)
        k_bin_right = k_bin_left + k_bin_size
        k_bin_center = 0.5*(k_bin_left + k_bin_right)
        k_bin_centers.append(k_bin_center)
    if not k_bin_centers:
        k_bin_centers.append(0.5*(k_min + k_max))
    k_bin_centers = asarray(k_bin_centers, dtype=C2np['double'])
    # Stretch and shift the array so that the end points match their
    # appropriate values exactly. Note that while the rightmost bin only
    # just includes k_max, half of the leftmost bin is less than k_min,
    # disfavouring binning of different modes into the very first bin.
    if len(k_bin_centers) > 1:
        k_bin_center_leftmost  = k_min
        k_bin_center_rightmost = k_max - 0.5*binsize_interp(k_max - 0.5*binsize_interp(k_max))
        k_bin_centers = (
            k_bin_center_leftmost + (k_bin_centers - k_bin_centers[0])*(
                (k_bin_center_rightmost - k_bin_center_leftmost)
                /(k_bin_centers[-1] - k_bin_centers[0])
            )
        )
    return k_bin_centers

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
    interlace='bint',
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
    k_bin_indices_ptr = cython.address(k_bin_indices[:])
    n_modes = declaration.n_modes
    if master:
        n_modes_ptr = cython.address(n_modes[:])
    power = declaration.power
    power_ptr = cython.address(power[:])
    # Begin progress message
    if len(components) == 1:
        component = components[0]
        masterprint(f'Computing power spectrum of {component.name} ...')
    else:
        components_str = ', '.join([component.name for component in components])
        masterprint(f'Computing power spectrum of {{{components_str}}} ...')
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
    slab_ptr = cython.address(slab[:, :, :])
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
    for k_bin_index in range(power.shape[0]):
        power_ptr[k_bin_index] *= normalization/n_modes_ptr[k_bin_index]
    # Done with the main power spectrum computation
    masterprint('done')

# Function which given a power spectrum declaration correctly populated
# with all fields will compute its linear CLASS power spectrum.
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
    if len(components) == 1:
        component = components[0]
        masterprint(f'Computing linear power spectrum of {component.name} ...')
    else:
        components_str = ', '.join([component.name for component in components])
        masterprint(f'Computing linear power spectrum of {{{components_str}}} ...')
    # Fill power_linear with values of the linear power spectrum.
    # Only the master will hold the values.
    get_linear_powerspec(components, k_bin_centers, power=power_linear)
    # Done with the linear power spectrum computation
    masterprint('done')

# Function for saving already computed power spectra
# to a single text file.
@cython.header(
    # Arguments
    declarations=list,
    filename=str,
    # Locals
    col='int',
    data='double[:, ::1]',
    declaration=object,  # PowerspecDeclaration
    declaration_group=list,
    declaration_groups=dict,
    header=str,
    header_info=object,  # PowerspecHeaderInfo
    k_bin_centers='double[::1]',
    n_cols='int',
    n_modes='Py_ssize_t[::1]',
    n_modes_float='double[::1]',
    n_rows='int',
    power='double[::1]',
    power_linear='double[::1]',
    size='Py_ssize_t',
    spectrum_plural=str,
    topline=str,
    σ='double',
    returns='void',
)
def save_powerspec(declarations, filename):
    if not master:
        return
    # Discard power spectrum declarations that should not be saved
    declarations = [declaration for declaration in declarations if declaration.do_data]
    if not declarations:
        return
    # Get header, format and delimiter specifier for the data file
    header_info = get_powerspec_header(declarations)
    header = header_info.header
    spectrum_plural = 'spectrum' if len(declarations) == 1 else 'spectra'
    masterprint(f'Saving power {spectrum_plural} to "{filename}" ...')
    # The top line of the header, stating general information
    header_significant_figures = np.max([
        declaration.significant_figures
        for declaration in declarations
    ])
    topline = unicode(
        f'Power {spectrum_plural} from CO𝘕CEPT job {jobid} at t = '
        + f'{{:.{header_significant_figures}g}} '.format(universals.t)
        + f'{unit_time}'
        + (
            f', a = ' + f'{{:.{header_significant_figures}g}}'.format(universals.a)
            if enable_Hubble else ''
        )
        + '.'
    )
    # The output data consists of a "k" column and a "modes" column for
    # each group, along with a "power" column for each power spectrum
    # and possibly another "power" if the linear power spectrum should
    # be outputted as well. The number of rows in a column depends on
    # the grid size, but to make it easier to read back in we make all
    # columns the same length by appending NaN's as required
    # (zeros for the modes).
    # Get a 2D array with the right size for storing all data.
    for declaration_group in header_info.declaration_groups.values():
        declaration = declaration_group[0]
        n_rows = declaration.k_bin_centers.shape[0]
        break
    n_cols = (
        2*len(header_info.declaration_groups)
        + len(declarations)
        + np.sum([
            declaration.do_linear
            for declaration in declarations
        ])
    )
    data = get_buffer((n_rows, n_cols))
    # Fill in data columns
    col = 0
    for declaration_group in header_info.declaration_groups.values():
        declaration = declaration_group[0]
        k_bin_centers = declaration.k_bin_centers
        n_modes       = declaration.n_modes
        size = k_bin_centers.shape[0]
        # New k
        data[:size, col] = k_bin_centers
        data[size:, col] = NaN
        col += 1
        # New modes
        n_modes_float = asarray(n_modes, dtype=C2np['double'])
        data[:size, col] = n_modes_float
        data[size:, col] = 0
        col += 1
        for declaration in declaration_group:
            # New power
            power = declaration.power
            data[:size, col] = power
            data[size:, col] = NaN
            col += 1
            # Compute the rms density variation
            # and insert it into the header.
            σ = compute_powerspec_σ(declaration)
            header = re.sub(r'= \{.+?\}', lambda m, σ=σ: m.group().format(σ), header, 1)
            # New linear power and rms density variation
            if declaration.do_linear:
                power_linear = declaration.power_linear
                data[:size, col] = power_linear
                data[size:, col] = NaN
                col += 1
                σ = compute_powerspec_σ(declaration, linear=True)
                header = re.sub(r'= \{.+?\}', lambda m, σ=σ: m.group().format(σ), header, 1)
    # Save data and header to text file
    np.savetxt(
        filename,
        data,
        fmt=header_info.fmt,
        delimiter=header_info.delimiter,
        header=f'{topline}\n{header}',
    )
    masterprint('done')

# Pure Python function for generating the header for a power spectrum
# data file, given a list of power spectrum declarations.
def get_powerspec_header(declarations):
    """Besides the header, this function also returns a list of data
    format specifiers, the delimiter needed between the data columns and
    a dict mapping power spectrum grid sizes to lists of
    power spectrum declarations.
    Importantly, the supplied list of power spectrum declarations should
    only contain declarations for which do_data is True.
    """
    # Look up in the cache
    cache_key = tuple([tuple(declaration.components) for declaration in declarations])
    header_info = powerspec_header_cache.get(cache_key)
    if header_info:
        # Cached result found.
        # Which components to include in the power spectrum
        # computation/plot may change over time due to components
        # changing their passive/active/terminated state, which in turn
        # change the passed declarations. As the cache key used above
        # depends on the components only, the resulting cached result
        # may hold outdated PowerspecDeclaration instances.
        # Update these before returning.
        for declaration_group in header_info.declaration_groups.values():
            for i, declaration_cached in enumerate(declaration_group):
                for declaration in declarations:
                    if declaration_cached.components == declaration.components:
                        declaration_group[i] = declaration
                        break
        return header_info
    # A column mapping each component to a number
    components = []
    for declaration in declarations:
        for component in declaration.components:
            if component not in components:
                components.append(component)
    longest_name_size = np.max([len(component.name) for component in components])
    column_components = ['Below, the following component mapping is used:']
    for i, component in enumerate(components):
        column_components.append(
            f'  {{:<{longest_name_size + 1}}} {i}'.format(f'{component.name}:')
        )
    # Group power spectrum declarations according to their grid size,
    # k2_max, number of unique modes (all in descending order), as well
    # as the exact k values at which the power spectrum is computed.
    declaration_groups_unordered = collections.defaultdict(list)
    for key, declarations_iter in itertools.groupby(
        declarations,
        key=(lambda declaration: (
            declaration.gridsize,
            declaration.k2_max,
            len(declaration.k_bin_centers),
            hashlib.sha1(asarray(declaration.k_bin_centers)).hexdigest(),
        )),
    ):
        declaration_groups_unordered[key] += list(declarations_iter)
    declaration_groups = {
        key: declaration_groups_unordered[key]
        for key in sorted(
            declaration_groups_unordered,
            key=(lambda t: (-t[0], -t[1], -t[2], t[3])),
        )
    }
    # The column headings
    column_headings = {
        'k': unicode(f'k [{unit_length}⁻¹]'),
        'modes': 'modes',
        'power': unicode(f'power [{unit_length}³]'),
    }
    # Helper function for obtaining the float format and width
    # given number of significant figures.
    def get_formatting(significant_figures):
        fmt_float = f'%-{{}}.{significant_figures - 1}e'
        width_float = significant_figures + n_chars_nonsignificant
        return fmt_float, width_float
    n_chars_nonsignificant = len(f'{1e+100:.1e}') - 2
    # The output data consists of a "k" column and a "modes" column for
    # each unique grid size, along with a "power" column for each power
    # spectrum.
    # Determine the headings for each column and their format specifier.
    group_spacing = 1
    group_delimiter = ' '*group_spacing
    col = 0
    components_heading = []
    σs_heading = []
    columns_heading = []
    fmt = []
    fmt_int = '%{}u'
    for (gridsize, *_), declaration_group in declaration_groups.items():
        if col > 0:
            # New group with new grid size begins. Insert additional
            # spacing by modifying the last elements of the
            # *s_heading and fmt.
            components_heading.append(components_heading.pop() + group_delimiter)
            σs_heading.append(σs_heading.pop() + group_delimiter)
            columns_heading.append(columns_heading.pop() + group_delimiter)
            fmt.append(fmt.pop() + group_delimiter)
        # New k
        col += 1
        column_heading = column_headings['k']
        fmt_float, width_float = get_formatting(
            np.max([declaration.significant_figures for declaration in declaration_group])
        )
        width = np.max((width_float, len(column_heading) + 2*(col == 1)))
        components_heading.append(' '*(width - 2*(col == 1)))
        σs_heading.append(' '*(width - 2*(col == 1)))
        extra_spacing = width - len(column_heading) - 2*(col == 1)
        columns_heading.append(
            ' '*(extra_spacing//2) + column_heading + ' '*(extra_spacing - extra_spacing//2)
        )
        fmt.append(fmt_float.format(width))
        # New modes
        col += 1
        column_heading = column_headings['modes']
        declaration = declaration_group[0]
        width = np.max((len(str(declaration.n_modes_max)), len(column_heading)))
        components_heading.append(' '*width)
        σs_heading.append(' '*width)
        extra_spacing = width - len(column_heading)
        columns_heading.append(
            ' '*(extra_spacing//2) + column_heading + ' '*(extra_spacing - extra_spacing//2)
        )
        fmt.append(fmt_int.format(width))
        for declaration in declaration_group:
            # Construct the rms density variation "σ₈" (or similar)
            # string. By convention, the unit is Mpc/h.
            σ_unit = ℝ[
                units.Mpc/(H0/(100*units.km/(units.s*units.Mpc))) if enable_Hubble else units.Mpc
            ]
            σ_str = ''.join([
                unicode('σ'),
                unicode_subscript(f'{declaration.tophat/σ_unit:.3g}'),
                ' = ',
            ])
            # New power and possibly linear power
            for power_type in range(1 + declaration.do_linear):
                col += 1
                if power_type == 0:  # Non-linear
                    component_heading = get_integerset_strrep([
                        components.index(component)
                        for component in declaration.components
                    ])
                    if len(declaration.components) == 1:
                        component_heading = f'component {component_heading}'
                    else:
                        component_heading = f'components {{{component_heading}}}'
                else:  # power_type == 1 (linear)
                    component_heading = '(linear)'
                fmt_float, width_float = get_formatting(declaration.significant_figures)
                σ_significant_figures = width_float - len(σ_str) - n_chars_nonsignificant
                if σ_significant_figures < 2:
                    σ_significant_figures = 2
                size1 = width_float - len(σ_str)
                size2 = len(f'{{:<{size1}.{σ_significant_figures - 1}e}}'.format(1e+100))
                size = np.max((size1, size2))
                σ_heading = σ_str + f'{{:<{size}.{σ_significant_figures - 1}e}}'
                column_heading = column_headings['power']
                width = np.max((
                    width_float,
                    len(column_heading),
                    len(component_heading),
                    len(σ_heading.format(1e+100)),
                ))
                extra_spacing = width - len(component_heading)
                components_heading.append(
                    ' '*(extra_spacing//2) + component_heading
                    + ' '*(extra_spacing - extra_spacing//2)
                )
                extra_spacing = width - len(σ_heading.format(1e+100))
                σs_heading.append(
                    ' '*(extra_spacing//2) + σ_heading
                    + ' '*(extra_spacing - extra_spacing//2)
                )
                extra_spacing = width - len(column_heading)
                columns_heading.append(
                    ' '*(extra_spacing//2) + column_heading
                    + ' '*(extra_spacing - extra_spacing//2)
                )
                extra_spacing = width - width_float
                fmt.append(' '*(extra_spacing//2) + fmt_float.format(width - extra_spacing//2))
    # Construct group header
    group_header_underlines = []
    delimiter = ' '*2
    for col, column_heading in enumerate(columns_heading):
        if column_heading.strip() == column_headings['k']:
            if col > 0:
                width -= group_spacing
                group_header_underlines.append(unicode('╭' + unicode('─')*(width - 2) + '╮'))
            width = len(column_heading)
        else:
            width += len(delimiter) + len(column_heading)
    group_header_underlines.append(unicode('╭' + unicode('─')*(width - 2) + '╮'))
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
    header = '\n'.join([
        '',
        *column_components,
        '',
        (delimiter + group_delimiter).join(group_headers),
        (delimiter + group_delimiter).join(group_header_underlines),
        delimiter.join(components_heading),
        delimiter.join(σs_heading),
        delimiter.join(columns_heading),
    ])
    # Store in cache and return
    header_info = PowerspecHeaderInfo(
        header, fmt, delimiter, declaration_groups,
    )
    powerspec_header_cache[cache_key] = header_info
    return header_info
# Cache and type used by the get_powerspec_header() function
cython.declare(powerspec_header_cache=dict)
powerspec_header_cache = {}
PowerspecHeaderInfo = collections.namedtuple(
    'PowerspecHeaderInfo',
    ('header', 'fmt', 'delimiter', 'declaration_groups'),
)

# Function which given a power spectrum declaration with the
# k_bin_centers and power fields correctly populated will compute the
# rms density variation of the power spectrum.
@cython.header(
    # Arguments
    declaration=object,  # PowerspecDeclaration
    linear='bint',
    # Locals
    W='double',
    i='Py_ssize_t',
    k_bin_centers='double[::1]',
    k_bin_index='Py_ssize_t',
    k_magnitude='double',
    kR='double',
    power='double[::1]',
    tophat='double',
    σ2='double',
    σ2_integrand='double[::1]',
    returns='double',
)
def compute_powerspec_σ(declaration, linear=False):
    tophat        = declaration.tophat
    k_bin_centers = declaration.k_bin_centers
    power         = declaration.power
    # If the σ to be computed is of the linear power spectrum,
    # we need to truncate k_bin_centers and power so that they do
    # not contain NaN's.
    if linear:
        power = declaration.power_linear
        power = asarray(power)[~np.isnan(power)]
        k_bin_centers = k_bin_centers[:power.shape[0]]
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

# Function which can measure different quantities of a passed component
@cython.header(
    # Arguments
    component='Component',
    quantity=str,
    communicate='bint',
    # Locals
    J_noghosts=object, # np.ndarray
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
    mom_i='double',
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
    Σmom_dim='double',
    Σmom2_dim='double',
    Σϱ='double',
    Σϱ2='double',
    ϱ='FluidScalar',
    ϱ_bar='double',
    ϱ_min='double',
    ϱ_mv='double[:, :, ::1]',
    ϱ_noghosts=object, # np.ndarray
    ϱ_ptr='double*',
    σ2mom_dim='double',
    σ2ϱ='double',
    σmom='double[::1]',
    σmom_dim='double',
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
        if component.representation == 'particles':
            # Total momentum of all particles, for each dimension
            for dim in range(3):
                Σmom_dim = Σmom2_dim = 0
                # Add up local particle momenta
                for indexˣʸᶻ in range(dim, 3*component.N_local, 3):
                    mom_i = mom[indexˣʸᶻ]
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
                σmom_dim = sqrt(σ2mom_dim)
                # Pack results
                Σmom[dim] = Σmom_dim
                σmom[dim] = σmom_dim
        elif component.representation == 'fluid':
            # Total momentum of all fluid elements, for each dimension.
            # Here the definition of momenta is chosen as
            # J*Vcell = (a**4*(ρ + c⁻²P))*Vcell
            #         = (V_phys*(ρ + c⁻²P))*a*u,
            # which reduces to mass*a*u for pressureless fluids and so
            # it is in correspondence with the momentum definition
            # for particles.
            for dim, fluidscalar in enumerate(component.J):
                J_noghosts = asarray(fluidscalar.grid_noghosts)
                # Total dim'th momentum of all fluid elements
                Σmom_dim = np.sum(J_noghosts)*Vcell
                # Total dim'th momentum squared of all fluid elements
                Σmom2_dim = np.sum(J_noghosts**2)*Vcell**2
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
                σmom_dim = sqrt(σ2mom_dim)
                # Pack results
                Σmom[dim] = Σmom_dim
                σmom[dim] = σmom_dim
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

