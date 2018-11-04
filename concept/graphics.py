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



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from communication import domain_size_x,  domain_size_y,  domain_size_z, '
                                  'domain_start_x, domain_start_y, domain_start_z,'
                                  'get_buffer,                                    '
)
cimport('from mesh import CIC_components2φ_general')

# Pure Python imports
from mpl_toolkits.mplot3d import proj3d  # Importing from mpl_toolkits.mplot3d enables 3D plotting



# Function for plotting power spectra
@cython.header(# Arguments
               k_bin_centers='double[::1]',
               power_dict=object,  # OrderedDict
               filename=str,
               powerspec_plot_select=dict,
               # Locals
               a_string=str,
               component_combination=tuple,
               component_combination_str=str,
               filename_combination=str,
               names_str=str,
               power='double[::1]',
               t_string=str,
               )
def plot_powerspec(k_bin_centers, power_dict, filename, powerspec_plot_select):
    """The power spectra are given in power_dict,
    which is an OrderedDict mapping component combinations
    (tuples of components) to arrays with the corresponding power.
    The matching k values are given by k_bin_centers.
    """
    # Only the master process takes part in the power spectra plotting
    if not master:
        return
    # Attach missing extension to filename
    if not filename.endswith('.png'):
        filename += '.png'
    # Plot each power spectrum given in power_dict,
    # if it has been selected for plotting in the
    # powerspec_select parameter. A refined version of this dict is
    # passed as powerspec_plot_select.
    for component_combination, power in power_dict.items():
        if not is_selected(component_combination, powerspec_plot_select):
            continue
        # The filename should reflect the individual
        # components/combinations, when several components/combinations
        # are being plotted.
        filename_combination = filename
        if len(power_dict) > 1:
            names_str = '_'.join([component.name.replace(' ', '-')
                                  for component in component_combination])
            if '_t=' in filename_combination:
                filename_combination = filename_combination.replace('_t=', f'_{names_str}_t=')
            elif '_a=' in filename_combination:
                filename_combination = filename_combination.replace('_a=', f'_{names_str}_a=')
            else:
                filename_combination = filename_combination.replace('.png', f'_{names_str}.png')
        if len(component_combination) == 1:
            component_combination_str = component_combination[0].name
        else:
            component_combination_str = '{{{}}}'.format(', '.join(
                [component.name for component in component_combination]
                                                                  )
                                                        )
        masterprint('Plotting power spectrum of {} and saving to "{}" ...'
                    .format(component_combination_str, filename_combination)
                    )
        # Plot power spectrum
        plt.figure()
        if np.any(asarray(power) != 0):
            plt.loglog(k_bin_centers, power, '-')
        else:
            # The odd case of no power at all
            plt.semilogx(k_bin_centers, power, '-')
        plt.xlabel(rf'$k$ $\mathrm{{[{unit_length}^{{-1}}]}}$', fontsize=14)
        plt.ylabel(rf'power $\mathrm{{[{unit_length}^3]}}$',    fontsize=14)
        t_string = (
            r'$t = {}\, \mathrm{{{}}}$'
            .format(significant_figures(universals.t, 4, fmt='tex'), unit_time)
        )
        a_string = ''
        if enable_Hubble:
            a_string = ', $a = {}$'.format(significant_figures(universals.a, 4, fmt='tex'))
        component_combination_str = (
            component_combination_str
            .replace('{', r'$\{$')
            .replace('}', r'$\}$')
            .replace(',', ',\n')
            )
        plt.title(
            f'{component_combination_str}\nat {t_string}{a_string}',
            fontsize=16,
            horizontalalignment='center',
            )
        plt.gca().tick_params(axis='both', which='major', labelsize=13)
        plt.gca().tick_params(axis='both', which='minor', labelsize=11)
        plt.tight_layout()
        plt.savefig(filename_combination)
        # Close the figure, leaving no trace in memory of the plot
        plt.close()
        # Finish progress message
        masterprint('done')

# Function for plotting detrended CLASS perturbations
@cython.pheader(
    # Arguments
    a_values='double[::1]',
    perturbations_detrended='double[::1]',
    transferfunction='TransferFunction',
    k='Py_ssize_t',
    # Locals
    a_values_raw=object,  # np.ndarray
    exponent=str,
    factor=str,
    filename=str,
    i='Py_ssize_t',
    k_str=str,
    key=str,
    loga_value='double',
    loga_values='double[::1]',
    loga_values_spline='double[::1]',
    perturbations_detrended_spline='double[::1]',
    perturbations_raw=object,  # np.ndarray
    skip='Py_ssize_t',
    specific_species='bint',
    spline='Spline',
    unit_latex=str,
    val=str,
    var_name_latex=str,
)
def plot_detrended_perturbations(a_values, perturbations_detrended, transferfunction, k):
    # All processes could carry out this work, but as it involved I/O,
    # we only allow the master process to do so.
    if not master:
        abort(f'rank {rank} called plot_detrended_perturbations()')
    loga_values = np.log(a_values)
    # Plot the detrended CLASS data
    plt.figure()
    plt.semilogx(a_values, perturbations_detrended, '.', markersize=3)
    # Plot the spline at values midway between the data points
    loga_values_spline             = empty(loga_values.shape[0] - 1, dtype=C2np['double'])
    perturbations_detrended_spline = empty(loga_values.shape[0] - 1, dtype=C2np['double'])
    spline = transferfunction.splines[k]
    skip = 0
    for i in range(loga_values_spline.shape[0]):
        loga_value = 0.5*(loga_values[i] + loga_values[i+1])
        if not (ℝ[spline.xmin] <= loga_value <= ℝ[spline.xmax]):
            skip += 1
            continue
        loga_values_spline[ℤ[i - skip]] = loga_value
        perturbations_detrended_spline[ℤ[i - skip]] = spline.eval(exp(loga_value))
    loga_values_spline = loga_values_spline[:ℤ[i - skip + 1]]
    perturbations_detrended_spline = perturbations_detrended_spline[:ℤ[i - skip + 1]]
    plt.semilogx(np.exp(loga_values_spline), perturbations_detrended_spline, '-',
        linewidth=1, zorder=0)
    # Decorate and save plot
    plt.xlabel('$a$', fontsize=14)
    var_name_latex = transferfunction.var_name
    for key, val in {
        'δ': r'{\delta}',
        'θ': r'{\theta}',
        'ρ': r'{\rho}',
        'σ': r'{\sigma}',
        'ʹ': r'^{\prime}',
    }.items():
        var_name_latex = var_name_latex.replace(key, val)
    unit_latex = {
        'δ'    : rf'',
        'θ'    : rf'[\mathrm{{{unit_time}}}^{{-1}}]',
        'δP': (
            rf'['
            rf'\mathrm{{{unit_mass}}}'
            rf'\mathrm{{{unit_length}}}^{{-1}}'
            rf'\mathrm{{{unit_time}}}^{{-2}}'
            rf']'
        ),
        'σ': rf'[\mathrm{{{unit_length}}}^2\mathrm{{{unit_time}}}^{{-2}}]',
        'hʹ': rf'[\mathrm{{{unit_time}}}^{{-1}}]',
        'H_Tʹ': rf'[\mathrm{{{unit_time}}}^{{-1}}]',
    }[transferfunction.var_name]
    unit_latex = (unit_latex
        .replace('(', '{')
        .replace(')', '}')
        .replace('**', '^')
        .replace('*', '')
        .replace('m_sun', r'm_{\odot}')
    )
    plt.ylabel(rf'$({var_name_latex} - \mathrm{{trend}})\, {unit_latex}$', fontsize=14)
    specific_species = transferfunction.var_name not in ('hʹ',)
    k_str = significant_figures(transferfunction.k_magnitudes[k], 3, fmt='tex', scientific=True)
    plt.title(
        (rf'{transferfunction.class_species}, ' if specific_species else '')
        + rf'$k = {k_str}\, \mathrm{{{unit_length}}}^{{-1}}$',
        fontsize=16,
        horizontalalignment='center',
    )
    plt.gca().tick_params(axis='x', which='major', labelsize=13)
    factor = significant_figures(transferfunction.factors[k], 6, fmt='tex', scientific=True)
    exponent = significant_figures(transferfunction.exponents[k], 6, scientific=False)
    plt.text(0.5, 0.8, rf'$\mathrm{{trend}} = {factor}{unit_latex.strip("[]")}a^{{{exponent}}}$',
        horizontalalignment='center',
        verticalalignment='center',
        transform=plt.gca().transAxes,
        fontsize=14,
    )
    plt.tight_layout()
    filename = output_dirs['powerspec'] + '/class_perturbations'
    filename += '/' + (var_name_latex
        .replace('\\', '')
        .replace('{', '')
        .replace('}', '')
        .replace('^', '')
        .replace('/', '_')
        .replace('sigma', 'shear')
        .replace('prime', '_prime')
    )
    if specific_species:
        filename += f'_{transferfunction.class_species}'
    os.makedirs(filename, exist_ok=True)
    filename += f'/{k}.png'
    plt.savefig(filename)
    plt.close()

# Function for plotting processed CLASS perturbations
@cython.pheader(
    # Arguments
    a_values='double[::1]',
    k_magnitudes='double[::1]',
    transfer='double[:, ::1]',
    var_name=str,
    class_species=str,
    n_plots_in_figure='Py_ssize_t',
    # Locals
    a='double',
    dirname=str,
    i='Py_ssize_t',
    i_figure='Py_ssize_t',
    key=str,
    nfigs='Py_ssize_t',
    unit_latex=str,
    val=str,
    var_name_ascii=str,
    var_name_latex=str,
)
def plot_processed_perturbations(a_values, k_magnitudes, transfer, var_name, class_species,
    n_plots_in_figure=10):
    """The 2D transfer array is the tabulated transfer function values,
    indexed as transfer[a, k], with the values of a and k given by
    a_values and k_magnitudes.
    """
    # All processes could carry out this work, but as it involved I/O,
    # we only allow the master process to do so.
    if not master:
        abort(f'rank {rank} called plot_processed_perturbations()')
    masterprint(
        f'Plotting {var_name} {class_species} transfer functions ...'
    )
    var_name_latex = var_name
    for key, val in {
        'δ': r'{\delta}',
        'θ': r'{\theta}',
        'ρ': r'{\rho}',
        'σ': r'{\sigma}',
        'ʹ': r'^{\prime}',
    }.items():
        var_name_latex = var_name_latex.replace(key, val)
    var_name_ascii = (var_name_latex
        .replace('\\', '')
        .replace('{', '')
        .replace('}', '')
        .replace('^', '')
        .replace('/', '_')
        .replace('sigma', 'shear')
        .replace('prime', '_prime')
    )
    dirname = '/'.join([
        output_dirs['powerspec'],
        'class_perturbations_processed',
        f'{var_name_ascii}_{class_species}'
    ])
    os.makedirs(dirname, exist_ok=True)
    unit_latex = {
        'δ'    : rf'',
        'θ'    : rf'[\mathrm{{{unit_time}}}^{{-1}}]',
        'δP': (
            rf'['
            rf'\mathrm{{{unit_mass}}}'
            rf'\mathrm{{{unit_length}}}^{{-1}}'
            rf'\mathrm{{{unit_time}}}^{{-2}}'
            rf']'
        ),
        'σ': rf'[\mathrm{{{unit_length}}}^2\mathrm{{{unit_time}}}^{{-2}}]',
        'hʹ': rf'[\mathrm{{{unit_time}}}^{{-1}}]',
    }[var_name]
    unit_latex = (unit_latex
        .replace('(', '{')
        .replace(')', '}')
        .replace('**', '^')
        .replace('*', '')
        .replace('m_sun', r'm_{\odot}')
    )
    nfigs = int(log10(a_values.shape[0])) + 1
    i_figure = 0
    plt.figure()
    for i in range(a_values.shape[0]):
        a = a_values[i]
        plt.semilogx(k_magnitudes, transfer[i, :],
            label='$a={}$'.format(significant_figures(a, nfigs, fmt='tex')))
        if ((i + 1)%n_plots_in_figure == 0) or i == ℤ[a_values.shape[0] - 1]:
            plt.legend()
            plt.xlabel(rf'$k\,[\mathrm{{{unit_length}}}^{{-1}}]$', fontsize=14)
            plt.ylabel(rf'${var_name_latex}\, {unit_latex}$', fontsize=14)
            plt.gca().tick_params(axis='x', which='major', labelsize=13)
            plt.tight_layout()
            plt.savefig(f'{dirname}/{i_figure}.png')
            i_figure += 1
            plt.cla()
    plt.close()
    masterprint('done')

# This function produces 2D renders of the density fields of single
# and sets of components.
@cython.header(
    # Arguments
    components=list,
    filename=str,
    # Locals
    L='double',
    N_bins='Py_ssize_t',
    N_data_outputs='Py_ssize_t',
    N_image_outputs='Py_ssize_t',
    a='double',
    axis=str,
    bin_edges='double[::1]',
    bins='Py_ssize_t[::1]',
    blurrinesses=list,
    color_truncation_factor_lower='double',
    color_truncation_factor_upper='double',
    colormap=str,
    colornumber='int',
    component='Component',
    component_combination=tuple,
    component_combination_str=str,
    component_combinations=object,  # generator
    critical_blurriness_ratio='double',
    data_coordinates='double[::1]',
    domain_start_i='Py_ssize_t',
    domain_start_j='Py_ssize_t',
    domain_start_k='Py_ssize_t',
    enhance='bint',
    exponent='double',
    exponent_lower='double',
    exponent_max='double',
    exponent_min='double',
    exponent_upper='double',
    ext=str,
    extend=tuple,
    filename_combination=str,
    i='Py_ssize_t',
    i_center='Py_ssize_t',
    i_max='Py_ssize_t',
    i_min='Py_ssize_t',
    index_m='Py_ssize_t',
    index_n='Py_ssize_t',
    interpolation_coordinates_m='double[::1]',
    interpolation_coordinates_n='double[::1]',
    interpolation_quantities=list,
    j='Py_ssize_t',
    k='Py_ssize_t',
    names_str=str,
    occupation='Py_ssize_t',
    terminal_projection='double[:, ::1]',
    terminal_projection_ANSI=list,
    terminal_projection_candidates=list,
    projection='double[:, ::1]',
    projection_enhanced='double[:, ::1]',
    projection_max='double',
    projection_min='double',
    shifting_factor='double',
    terminal_resolution='Py_ssize_t',
    value='double',
    vmin='double',
    vmax='double',
    x='double',
    y='double',
    z='double',
    Σbins='Py_ssize_t',
    φ='double[:, :, ::1]',
)
def render2D(components, filename):
    """This function will produce 2D renders of the passed components.
    A slab of the density field will be projected onto a plane.
    The details of this projection is specified in the render2D_options
    user parameter. Before the projection, the density field within the
    slab will be constructed using CIC interpolation.
    """
    # Remove any extension on the filename
    for ext in ('.hdf5', '.png'):
        if filename.endswith(ext):
            filename = filename[:len(filename) - len(ext)]
            break
    # Always use the current value of the scale factor
    a = universals.a
    # Generator yielding tuples of all possible combinations
    # of the passed components.
    component_combinations = itertools.chain.from_iterable(
        [itertools.combinations(components, i) for i in range(1, len(components) + 1)]
    )
    # Count the number of output files
    N_data_outputs = 0
    N_image_outputs = 0
    for component_combination in component_combinations:
        if is_selected(component_combination, render2D_data_select):
            N_data_outputs += 1
        if is_selected(component_combination, render2D_image_select):
            N_image_outputs += 1
    # Rebuild the generator
    component_combinations = itertools.chain.from_iterable(
        [itertools.combinations(components, i) for i in range(1, len(components) + 1)]
    )
    # Produce 2D renders of each combination of components,
    # if they have been selected for in the render2D_select parameter.
    for component_combination in component_combinations:
        if not (
                is_selected(component_combination, render2D_data_select)
            or  is_selected(component_combination, render2D_image_select)
            or  is_selected(component_combination, render2D_terminal_image_select)
            ):
            continue
        component_combination_str = ', '.join(
            [component.name for component in component_combination]
        )
        if len(component_combination) > 1:
            component_combination_str = f'{{{component_combination_str}}}'
        masterprint(f'Rendering 2D projection of {component_combination_str} ...')
        # Extract some options for this component combination
        # from the render2D_options user parameter.
        axis   = is_selected(component_combination, render2D_options['axis'])
        extend = is_selected(component_combination, render2D_options['extend'])
        # We now do the CIC interpolation of the components onto the
        # φ grid. We choose to interpolate the mass of each
        # component onto the grid.
        # Since all particles have the same mass, the mass contribution
        # from a single particle is Σmass/component.N,
        # which equals component.mass.
        # For fluids, each fluid element contributes to the mass by
        # an amount (a*L_cell)**3*ρ(x)
        #         = (a*boxsize/component.gridsize)**3*ρ(x)
        #         = (boxsize/component.gridsize)**3*a**(-3*w_eff)*ϱ(x).
        # Note that for fluids the total amount of mass is not constant
        # but depends on w_eff(a).
        interpolation_quantities = [
            # Particle components
            ('particles', [component.mass for component in component_combination]),
            # Fluid components
            ('ϱ', [(boxsize/component.gridsize)**3*a**(-3*component.w_eff(a=a))
                   for component in component_combination]),
            ]
        φ = CIC_components2φ_general(
            list(component_combination),
            interpolation_quantities,
            add_particles_and_fluids=True,
        )
        domain = φ[2:(φ.shape[0] - 3), 2:(φ.shape[1] - 3), 2:(φ.shape[0] - 3)]
        # The array storing the projected render. This is allocated
        # in full on every process.
        projection = get_buffer((φ_gridsize, )*2, 'projection', nullify=True)
        # Fill up the local part of the projection array.
        # When axis is 'x' the projection will be onto the yz plane
        # with y right and z up.
        # When axis is 'y' the projection will be onto the xz plane
        # with x right and z up.
        # When axis is 'z' the projection will be onto the xy plane
        # with x right and y up.
        # The rightward and upward directions will be referred to as
        # m and n, respectively. Because the 2D projection array is C
        # contiguoues it is in row-major order, and so m should be the
        # second dimension while n should be the first dimension.
        # Also, since rows are counted downwards, the n dimension should
        # be indexed backwards.
        L = boxsize/φ_gridsize
        domain_start_i = int(round(domain_start_x/L))
        domain_start_j = int(round(domain_start_y/L))
        domain_start_k = int(round(domain_start_z/L))
        for i in range(ℤ[domain.shape[0]]):
            x = domain_start_x + i*L
            with unswitch(1):
                if 𝔹[axis == 'x']:
                    if ℝ[extend[0]] <= x:
                        weight = (ℝ[extend[1]] - x)/L
                        if weight > 1:
                            weight = 1
                    else:
                        weight = 1 - (ℝ[extend[0]] - x)/L
                    if weight <= 0:
                        continue
                else:
                    index_m = domain_start_i + i
            for j in range(ℤ[domain.shape[1]]):
                y = domain_start_y + j*L
                with unswitch(2):
                    if 𝔹[axis == 'x']:
                        index_m = domain_start_j + j
                    elif 𝔹[axis == 'y']:
                        if ℝ[extend[0]] <= y:
                            weight = (ℝ[extend[1]] - y)/L
                            if weight > 1:
                                weight = 1
                        else:
                            weight = 1 - (ℝ[extend[0]] - y)/L
                        if weight <= 0:
                            continue
                    else:
                        index_n = ℤ[φ_gridsize - 1 - domain_start_j] - j
                for k in range(ℤ[domain.shape[2]]):
                    z = domain_start_z + k*L
                    with unswitch(3):
                        if not 𝔹[axis == 'z']:
                            index_n = ℤ[φ_gridsize - 1 - domain_start_k] - k
                        else:
                            if ℝ[extend[0]] <= z:
                                weight = (ℝ[extend[1]] - z)/L
                                if weight > 1:
                                    weight = 1
                            else:
                                weight = 1 - (ℝ[extend[0]] - z)/L
                            if weight <= 0:
                                continue
                    # Include this grid cell
                    projection[index_n, index_m] += weight*domain[i, j, k]
        # Sum up contributions from all processes into the master,
        # after which only the master process should carry on.
        Reduce(sendbuf=(MPI.IN_PLACE if master else projection),
               recvbuf=(projection   if master else None),
               op=MPI.SUM,
               )
        if not master:
            continue
        # Store projected image as an hdf5 file
        if is_selected(component_combination, render2D_data_select):
            # The filename should reflect the component combination
            filename_combination = filename + '.hdf5'
            if N_data_outputs > 1:
                names_str = '_'.join(
                    [component.name.replace(' ', '-') for component in component_combination]
                )
                if '_t=' in filename_combination:
                    filename_combination = (
                        filename_combination.replace('_t=', f'_{names_str}_t=')
                    )
                elif '_a=' in filename_combination:
                    filename_combination = (
                        filename_combination.replace('_a=', f'_{names_str}_a=')
                    )
                else:
                    filename_combination = (
                        filename_combination.replace('.hdf5', f'_{names_str}.hdf5')
                    )
            masterprint(f'Saving data to "{filename_combination}" ...')
            with open_hdf5(filename_combination, mode='w') as hdf5_file:
                # Save used base unit
                hdf5_file.attrs['unit time'  ] = unit_time
                hdf5_file.attrs['unit length'] = unit_length
                hdf5_file.attrs['unit mass'  ] = unit_mass
                # Save attributes
                hdf5_file.attrs['boxsize'              ] = boxsize
                hdf5_file.attrs['component combination'] = component_combination_str
                hdf5_file.attrs['axis'                 ] = axis
                hdf5_file.attrs['extend'               ] = extend
                if enable_Hubble:
                    hdf5_file.attrs['a'] = a
                hdf5_file.attrs['t'    ] = universals.t
                # Store the 2D projection
                dset = hdf5_file.create_dataset(
                    'data',
                    asarray(projection).shape,
                    dtype=C2np['double']
                )
                dset[...] = projection
            masterprint('done')
        # If no colorized image should be produced from the projection,
        # skip the following.
        if not (
               is_selected(component_combination, render2D_image_select)
            or is_selected(component_combination, render2D_terminal_image_select)
            ):
            masterprint('done')
            continue
        # Extract further options for this component combination
        # from the render2D_options user parameter.
        colormap = is_selected(component_combination, render2D_options['colormap'])
        enhance  = is_selected(component_combination, render2D_options['enhance'])
        # Enhance the projected image by applying a non-linear
        # transformation of the form
        # projection → projection**exponent.
        # We want to find a value for the exponent which leads to a nice
        # distribution of the values in the projection. We take this to
        # be the case when the histogram of these values is "centered"
        # at the value specified by the shifting_factor variable.
        # A shifting_factor of 0.5 implies that the histogram of the
        # pixel values is "centered" in the middle of the axis, with the
        # same distance to the first and last bin. For Gaussian data,
        # this require a value of the exponent tending to 0. Thus,
        # the shifting factor should be below 0.5. A shifting_factor
        # between 0 and 0.5 shifts the center of the histogram to be at
        # the location of shifting_factor, measured relative to the
        # histogram axis. Here, the center is defined to be the point
        # which partitions the histogram into two parts which integrate
        # to the same value.
        if enhance:
            masterprint(f'Enhancing image ...')
            shifting_factor = 0.28
            # Enforce all pixel values to be between 0 and 1
            projection_min = np.min(projection)
            if projection_min != 0:
                projection = asarray(projection) - projection_min
            projection_max = np.max(projection)
            if projection_max not in (0, 1):
                projection = asarray(projection)*(1/projection_max)
            # Find a good value for the exponent using a binary search
            exponent_min = 1e-2
            exponent_max = 1e+2
            exponent_lower = exponent_min
            exponent_upper = exponent_max
            exponent = 1
            i_min = -4
            i_max = -2
            N_bins = np.max([25, φ_gridsize**2//100])
        while enhance:
            # Construct histogram over projection**exponent
            projection_enhanced = asarray(projection)**exponent
            bins, bin_edges = np.histogram(projection_enhanced, N_bins)
            # Compute the sum of all bins. This is equal to the sum of
            # values in the projection. However, we skip bins[0] since
            # sometimes empty cells results in a large spike there.
            Σbins = ℤ[φ_gridsize**2] - bins[0]
            # Find the position of the center of the histogram,
            # defined by the sums of bins being the same on both
            # sides of this center. We again skip bins[0].
            occupation = 0
            for i in range(1, N_bins):
                occupation += bins[i]
                if occupation >= ℤ[Σbins//2]:
                    i_center = i
                    break
            else:
                masterwarn(
                    'Something went wrong during image enhancement. '
                    'The image will be generated from the raw data.'
                    )
                enhance = False
                masterprint('done')
                break
            if i_center < ℤ[N_bins*shifting_factor]:
                # The exponent should be decreased
                exponent_upper = exponent
                i_min = i_center
            elif i_center > ℤ[N_bins*shifting_factor]:
                # The exponent should be increased
                exponent_lower = exponent
                i_max = i_center
            else:
                # Good choice of exponent found
                break
            # The current value of the exponent does not place the
            # "center" of the histogram at the desired location
            # specified by shifting_factor.
            # Check if the the binary seach has (almost) converged on
            # some other value.
            if i_max >= i_min and i_max - i_min <= 1:
                break
            # Check if the exponent is very close
            # to one of the extreme values.
            if exponent/exponent_min < 1.001:
                exponent = exponent_min
                break
            elif exponent_max/exponent < 1.001:
                exponent = exponent_max
                break
            # Update the exponent. As the range of the exponent is
            # large, the binary step is done in logarithmic space.
            exponent = sqrt(exponent_lower*exponent_upper)
        # Set color limits
        if enhance:
            # Apply the image enhancement
            projection = projection_enhanced
            # To further enhance the image, we set the color limits
            # so as to truncate the color space at both ends,
            # saturating pixels with very little or very high intensity.
            # The color limits vmin and vmax are determined based on
            # color_truncation_factor_lower and
            # color_truncation_factor_upper, respectively.
            # These specify the accumulated fraction of Σbins at which
            # the histogram should be truncated, for the lower and
            # upper intensity ends. For images with a lot of structure
            # the best results are obtained by giving the lower color
            # truncation quite a large value (this effectively removes
            # the background), while giving the higher color truncation
            # a small value, so that small very overdense regions
            # appear clearly.
            color_truncation_factor_lower = 0.001
            color_truncation_factor_upper = 0.00002
            occupation = 0
            for i in range(1, N_bins):
                occupation += bins[i]
                if occupation >= ℝ[color_truncation_factor_lower*Σbins]:
                    vmin = bin_edges[i - 1]
                    break
            occupation = 0
            for i in range(N_bins - 1, 0, -1):
                occupation += bins[i]
                if occupation >= ℝ[color_truncation_factor_upper*Σbins]:
                    vmax = bin_edges[i + 1]
                    break
            masterprint('done')
        else:
            vmin = np.min(projection)
            vmax = np.max(projection)
        # Draw projected image in the terminal
        if is_selected(component_combination, render2D_terminal_image_select):
            # Construct a version of the projection with the resolution
            # specified as the terminal resolution for this
            # component combination. Since each character in the
            # terminal is rectangular with about double the height
            # compared to the width, the terminal projection will only
            # have half as many rows as it has columns.
            terminal_resolution = is_selected(
                component_combination,
                render2D_options['terminal resolution'],
            )
            if terminal_resolution == φ_gridsize:
                # When the terminal resolution matches that of the
                # φ grid, simply use the φ grid (projection) as is.
                terminal_projection = np.ascontiguousarray(projection[::2, :])
            else:
                # Coordinate mapping between the original
                # data (projection) and the interpolated
                # data (terminal_projection).
                data_coordinates = linspace(
                    0,
                    terminal_resolution - 1,
                    φ_gridsize,
                )
                interpolation_coordinates_m = linspace(
                    0,
                    terminal_resolution - 1,
                    terminal_resolution,
                )
                interpolation_coordinates_n = linspace(
                    0,
                    terminal_resolution - 1,
                    (terminal_resolution + 1)//2,
                )
                # Interpolate original data (projection) to the new
                # image format. Sometimes (e.g. when φ_gridsize is
                # larger than the cube root of the number of particles
                # and particles are placed in close to perfect mesh
                # alignment) the resulting interpolated image gets very
                # blurry and few structural features remain. This can
                # (perhaps surprisingly) be fixed by slightly smoothing
                # the data before doing the interpolation. As this
                # smothing does not noticeably alter the image in the
                # normal case, we always perform this smoothing.
                # In the end, this is a result of the fact that we are
                # using the φ grid to generate the projected image,
                # and so the image depends on φ_grid.
                terminal_projection = np.ascontiguousarray(
                    scipy.interpolate.interp2d(
                        data_coordinates,
                        data_coordinates,
                        scipy.ndimage.filters.gaussian_filter(
                            projection, sigma=1, truncate=3, mode='wrap',
                        ),
                        'cubic',
                    )(interpolation_coordinates_m, interpolation_coordinates_n)
                )
            # Apply the colormap
            # specified for this component combination.
            set_terminal_colormap(colormap)
            # Construct list of strings, each string being a space
            # prepended with an ANSI/VT100 control sequences which sets
            # the background color. When printed together, these strings
            # produce an ANSI image of the terminal projection.
            # We need to map the values between vmin and vmax to
            # the 238 higher integer color numbers 18–255 (the lowest 18
            # color numbers are already occupied).
            terminal_projection_ANSI = []
            for     i in range(ℤ[terminal_projection.shape[0]]):
                for j in range(ℤ[terminal_projection.shape[1]]):
                    value = terminal_projection[i, j]
                    if value > vmax:
                        value = vmax
                    elif value < vmin:
                        value = vmin
                    colornumber = 18 + int(round((value - vmin)*ℝ[237/(vmax - vmin)]))
                    # Insert a space with colored background
                    terminal_projection_ANSI.append(f'{ANSI_ESC}[48;5;{colornumber}m ')
                # Insert newline with no background color
                terminal_projection_ANSI.append(f'{ANSI_ESC}[0m\n')
            # Print the ANSI image to the terminal
            masterprint(''.join(terminal_projection_ANSI), end='', indent=-1, wrap=False)
        # Save colorized image to disk
        if is_selected(component_combination, render2D_image_select):
            # The filename should reflect the component combination
            filename_combination = filename + '.png'
            if N_image_outputs > 1:
                names_str = '_'.join(
                    [component.name.replace(' ', '-') for component in component_combination]
                )
                if '_t=' in filename_combination:
                    filename_combination = filename_combination.replace('_t=', f'_{names_str}_t=')
                elif '_a=' in filename_combination:
                    filename_combination = filename_combination.replace('_a=', f'_{names_str}_a=')
                else:
                    filename_combination = filename_combination.replace('.png', f'_{names_str}.png')
            masterprint(f'Saving image to "{filename_combination}" ...')
            plt.imsave(
                filename_combination,
                projection,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
            )
            masterprint('done')
        # Done with the entire rendering process
        # for this component combination.
        masterprint('done')
# Construct the render2D_data_select, render2D_image_select
# and render2D_terminal_image_select dicts from
# the render2D_data_select parameter.
cython.declare(
    render2D_data_select=dict,
    render2D_image_select=dict,
    render2D_terminal_image_select=dict,
)
render2D_data_select = {
    key: val['data' ] for key, val in render2D_select.items()
}
render2D_image_select = {
    key: val['image'] for key, val in render2D_select.items()
}
render2D_terminal_image_select = {
    key: val['terminal image'] for key, val in render2D_select.items()
}

# Function for chancing the colormap of the terminal
def set_terminal_colormap(colormap):
    """This function constructs and apply a terminal colormap with
    256 - 16 - 2 = 238 ANSI/VT100 control sequences, remapping the 238
    higher color numbers. The 16 + 2 = 18 lowest are left alone in order
    not to mess with standard terminal coloring and the colors used for
    the CO𝘕CEPT logo at startup.
    If the specified colormap is already in use, nothing is done.
    """
    global current_terminal_colormap
    if not master:
        return
    if current_terminal_colormap == colormap:
        return
    current_terminal_colormap = colormap
    colormap_ANSI = getattr(matplotlib.cm, colormap)(linspace(0, 1, 238))[:, :3]
    for i, rgb in enumerate(colormap_ANSI):
        colorhex = matplotlib.colors.rgb2hex(rgb)
        masterprint(
            f'{ANSI_ESC}]4;{18 + i};rgb:'
            f'{colorhex[1:3]}/{colorhex[3:5]}/{colorhex[5:]}{ANSI_ESC}\\',
            end='',
            indent=-1,
            wrap=False,
            ensure_newline_after_ellipsis=False,
        )
# Global variable used to keep track of the currently applied
# terminal colormap, used by the set_terminal_colormap function.
cython.declare(current_terminal_colormap=str)
current_terminal_colormap = None

# Function for 3D renderings of the components
@cython.header(# Arguments
               components=list,
               filename=str,
               cleanup='bint',
               tmp_dirname=str,
               # Locals
               N='Py_ssize_t',
               N_local='Py_ssize_t',
               a_str=str,
               artists_text=dict,
               color='double[::1]',
               component='Component',
               component_dict=dict,
               figname=str,
               filename_component=str,
               filename_component_alpha=str,
               filename_component_alpha_part=str,
               filenames_component_alpha=list,
               filenames_component_alpha_part=list,
               filenames_components=list,
               i='Py_ssize_t',
               index='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               label_props=list,
               label_spacing='double',
               name=str,
               names=tuple,
               part='int',
               posx_mv='double[::1]',
               posy_mv='double[::1]',
               posz_mv='double[::1]',
               render3D_dir=str,
               rgbα='double[:, ::1]',
               scatter_size='double',
               size='Py_ssize_t',
               size_i='Py_ssize_t',
               size_j='Py_ssize_t',
               size_k='Py_ssize_t',
               t_str=str,
               xi='double',
               yj='double',
               zk='double',
               α='double',
               α_factor='double',
               α_homogeneous='double',
               α_min='double',
               ϱ_noghosts='double[:, :, :]',
               ϱbar_component='double',
               )
def render3D(components, filename, cleanup=True, tmp_dirname='.renders3D'):
    global render3D_image
    # Do not 3D render anything if
    # render3D_select does not contain any True values.
    if not any(render3D_select.values()):
        return
    # Attach missing extension to filename
    if not filename.endswith('.png'):
        filename += '.png'
    # The directory for storing the temporary 3D renders
    render3D_dir = '{}/{}'.format(os.path.dirname(filename), tmp_dirname)
    # Initialize figures by building up render3D_dict, if this is the
    # first time this function is called.
    if not render3D_dict:
        masterprint('Initializing 3D renders ...')
        # Make cyclic default colors as when doing multiple plots in
        # one figure. Make sure that none of the colors are identical
        # to the background color.
        default_colors = itertools.cycle([to_rgb(prop['color'])
                                          for prop in matplotlib.rcParams['axes.prop_cycle']
                                          if not all(to_rgb(prop['color']) == render3D_bgcolor)])
        for component in components:
            if not is_selected(component, render3D_select):
                continue
            # This component should be 3D rendered.
            # Prepare a figure for the 3D render of the i'th component.
            figname = 'render3D_{}'.format(component.name)
            dpi = 100  # This only affects the font size relative to the figure
            fig = plt.figure(figname, figsize=[render3D_resolution/dpi]*2, dpi=dpi)
            ax = fig.gca(projection='3d', facecolor=render3D_bgcolor)
            # The color and α (of a homogeneous column through the
            # entire box) of this component.
            if component.name.lower() in render3D_colors:
                # This component is given a specific color by the user
                color, α_homogeneous = render3D_colors[component.name.lower()]
            elif 'all' in render3D_colors:
                # All components are given the same color by the user
                color, α_homogeneous = render3D_colors['all']
            else:
                # No color specified for this particular component.
                # Assign the next color from the default cyclic colors.
                color = next(default_colors)
                α_homogeneous = 0.2
            # Alpha values below this small value appear completely
            # invisible, for whatever reason.
            α_min = 0.0059
            # The artist for the component
            if component.representation == 'particles':
                # The particle size on the figure.
                # The size is chosen such that the particles stand side
                # by side in a homogeneous universe (more or less).
                N = component.N
                scatter_size = 1550*np.prod(fig.get_size_inches())/N**ℝ[2/3]
                # Determine the α value which ensures that a homogeneous
                # column through the entire box will result in a
                # combined α value of α_homogeneous. Alpha blending
                # is non-linear, but via the code given in
                # https://stackoverflow.com/questions/28946400
                # /is-it-possible-for-matplotlibs-alpha-transparency
                # -values-to-sum-to-1
                # I have found that 4/∛N is a good approximation to
                # the α value needed to make the combined α equal to 1.
                α = α_homogeneous*4/cbrt(N)
                # Alpha values lower than α_min are not allowed.
                # Shrink the scatter size to make up for the larger α.
                if α < α_min:
                    scatter_size *= α/α_min
                    α = α_min
                # Apply size and alpha
                artist_component = ax.scatter(0, 0, 0,
                                              alpha=α,
                                              c=np.expand_dims(color, 0),
                                              s=scatter_size,
                                              depthshade=False,
                                              lw=0,
                                              )
            elif component.representation == 'fluid':
                # To 3D render fluid elements, their explicit positions
                # are needed. In the following, these are computed and
                # stored in the variables posx_mv, posy_mv and posz_mv.
                size_i = component.shape_noghosts[0] - 1
                size_j = component.shape_noghosts[1] - 1
                size_k = component.shape_noghosts[2] - 1
                # Number of local fluid elements
                size = size_i*size_j*size_k
                # Allocate arrays for storing grid positions
                posx_mv = empty(size, dtype='double')
                posy_mv = empty(size, dtype='double')
                posz_mv = empty(size, dtype='double')
                # Fill the arrays
                index = 0
                for i in range(size_i):
                    xi = domain_start_x + i*ℝ[domain_size_x/size_i]
                    for j in range(size_j):
                        yj = domain_start_y + j*ℝ[domain_size_y/size_j]
                        for k in range(size_k):
                            zk = domain_start_z + k*ℝ[domain_size_z/size_k]
                            posx_mv[index] = xi
                            posy_mv[index] = yj
                            posz_mv[index] = zk
                            index += 1
                # 2D array with rgbα rows, one row for each
                # fluid element. This is the only array which will be
                # updated for each new 3D render, and only the α column
                # will be updated.
                rgbα = np.empty((size, 4), dtype=C2np['double'])
                for i in range(size):
                    for dim in range(3):
                        rgbα[i, dim] = color[dim]
                    rgbα[i, 3] = 1
                # The particle (fluid element) size on the figure.
                # The size is chosen such that the particles stand side
                # by side in a homogeneous universe (more or less).
                N = component.gridsize**3
                scatter_size = 1550*np.prod(fig.get_size_inches())/N**ℝ[2/3]
                # Determine the α multiplication factor which ensures
                # that a homogeneous column through the entire box will
                # result in an α value of α_homogeneous. Alpha blending
                # is non-linear, but via the code given in
                # https://stackoverflow.com/questions/28946400
                # /is-it-possible-for-matplotlibs-alpha-transparency
                # -values-to-sum-to-1
                # I have found that 4/∛N is a good approximation to
                # the α value needed to make the combined α equal to 1.
                α_factor = α_homogeneous*4/cbrt(N)
                # An α_factor below α_min are not allowed.
                # Shrink the scatter size to make up for the larger α.
                if α_factor < α_min:
                    scatter_size *= α_factor/α_min
                    α_factor = α_min
                # Plot the fluid elements as a 3D scatter plot
                artist_component = ax.scatter(posx_mv, posy_mv, posz_mv,
                                              c=rgbα,
                                              s=scatter_size,
                                              depthshade=False,
                                              lw=0,
                                              )
                # The set_facecolors method on the artist can be used
                # to update the α values on the plot. This function is
                # called internally my matplotlib with wrong arguments,
                # cancelling the α updates. For this reason, we
                # replace this method with a dummy method, while
                # keeping the original as _set_facecolors (though we
                # do not use this, as we set the _facecolors attribute
                # manually instead).
                artist_component._set_facecolors = artist_component.set_facecolors
                artist_component.set_facecolors = dummy_func
            # The artists for the cosmic time and scale factor text
            artists_text = {}
            label_spacing = 0.07
            label_props = [(label_spacing,     label_spacing, 'left'),
                           (1 - label_spacing, label_spacing, 'right')]
            artists_text['t'] = ax.text2D(label_props[0][0],
                                          label_props[0][1],
                                          '',
                                          fontsize=16,
                                          horizontalalignment=label_props[0][2],
                                          transform=ax.transAxes,
                                          )
            if enable_Hubble:
                artists_text['a'] = ax.text2D(label_props[1][0],
                                              label_props[1][1],
                                              '',
                                              fontsize=16,
                                              horizontalalignment=label_props[1][2],
                                              transform=ax.transAxes,
                                              )
            # Configure axis options
            ax.set_aspect('equal')
            ax.dist = 9  # Zoom level
            ax.set_xlim(0, boxsize)
            ax.set_ylim(0, boxsize)
            ax.set_zlim(0, boxsize)
            ax.axis('off')  # Remove panes, gridlines, axes, ticks, etc.
            for spine in ax.spines.values():
                # Needed due to bug in matplotlib 3.0.0
                spine.set_visible(False)
            plt.tight_layout(pad=-1)  # Extra tight layout, to prevent white frame
            proj3d.persp_transformation = orthographic_proj  # Use orthographic 3D projection
            # Store the figure, axes and the component
            # and text artists in the render3D_dict.
            render3D_dict[component.name] = {'fig': fig,
                                             'ax': ax,
                                             'artist_component': artist_component,
                                             'artists_text': artists_text,
                'α_factor': (α_factor if component.representation == 'fluid' else None),
                'rgbα'    : (rgbα     if component.representation == 'fluid' else None),
                                           }
        # Create the temporary 3D render directory if necessary
        if not (nprocs == 1 == len(render3D_dict)):
            if master:
                os.makedirs(render3D_dir, exist_ok=True)
            Barrier()
        masterprint('done')
        # Return if no component is to be 3D rendered
        if not render3D_dict:
            return
    # Print out progress message
    names = tuple(render3D_dict.keys())
    if len(names) == 1:
        masterprint('Rendering {} in 3D and saving to "{}" ...'.format(names[0], filename))
    else:
        filenames_components = []
        for name in names:
            name = name.replace(' ', '-')
            filename_component = filename
            if '_t=' in filename:
                filename_component = filename.replace('_t=', '_{}_t='.format(name))
            elif '_a=' in filename:
                filename_component = filename.replace('_a=', '_{}_a='.format(name))
            else:
                filename_component = filename.replace('.png', '_{}.png'.format(name))
            filenames_components.append('"{}"'.format(filename_component))
        masterprint('3D rendering {} and saving to {} ...'
                    .format(', '.join(names), ', '.join(filenames_components)))
    # 3D render each component separately
    for component in components:
        if component.name not in render3D_dict:
            continue
        # Switch to the render3D figure
        figname = 'render3D_{}'.format(component.name)
        plt.figure(figname)
        # Extract figure elements
        component_dict   = render3D_dict[component.name]
        fig              = component_dict['fig']
        ax               = component_dict['ax']
        artist_component = component_dict['artist_component']
        artists_text     = component_dict['artists_text']
        if component.representation == 'particles':
            # Update particle positions on the figure
            N_local = component.N_local
            artist_component._offsets3d = (component.posx_mv[:N_local],
                                           component.posy_mv[:N_local],
                                           component.posz_mv[:N_local])
        elif component.representation == 'fluid':
            rgbα     = component_dict['rgbα']
            α_factor = component_dict['α_factor']
            # Measure the mean value of the ϱ grid
            ϱ_noghosts = component.ϱ.grid_noghosts
            ϱbar_component = allreduce(np.sum(ϱ_noghosts[:(ϱ_noghosts.shape[0] - 1),
                                                         :(ϱ_noghosts.shape[1] - 1),
                                                         :(ϱ_noghosts.shape[2] - 1)]),
                                       op=MPI.SUM)/component.gridsize**3
            # Update the α values in rgbα array based on the values of
            # ϱ at each grid point. The rgb-values remain the same for
            # all 3D renders of this component.
            index = 0
            for         i in range(ℤ[ϱ_noghosts.shape[0] - 1]):
                for     j in range(ℤ[ϱ_noghosts.shape[1] - 1]):
                    for k in range(ℤ[ϱ_noghosts.shape[2] - 1]):
                        α = ℝ[α_factor/ϱbar_component]*ϱ_noghosts[i, j, k]
                        if α > 1:
                            α = 1
                        rgbα[index, 3] = α
                        index += 1
            # Apply the new α values to the artist.
            # We do this by setting the attribute _facecolors,
            # which is much faster than using the set_facecolors
            # method.
            artist_component._facecolors = rgbα
        # Print the current cosmic time and scale factor on the figure
        if master:
            t_str = a_str = ''
            t_str = '$t = {}\, \mathrm{{{}}}$'.format(significant_figures(universals.t, 4, 'tex'),
                                                      unit_time)
            artists_text['t'].set_text(t_str)
            if enable_Hubble:
                a_str = '$a = {}$'.format(significant_figures(universals.a, 4, 'tex'))
                artists_text['a'].set_text(a_str)
            # Make the text color black or white,
            # dependent on the background color.
            for artist_text in artists_text.values():
                if sum(render3D_bgcolor) < 1:
                    artist_text.set_color('white')
                else:
                    artist_text.set_color('black')
        # Save the 3D render
        if nprocs == 1:
            filename_component_alpha_part = ('{}/{}_alpha.png'
                                              .format(render3D_dir,
                                                      component.name.replace(' ', '-')))
        else:
            filename_component_alpha_part = ('{}/{}_alpha_{}.png'
                                             .format(render3D_dir,
                                                     component.name.replace(' ', '-'),
                                                     rank))
        if nprocs == 1 == len(render3D_dict):
            # As this is the only 3D render which should be done, it can
            # be saved directly in its final, non-transparent state.
            plt.savefig(filename, transparent=False)
            masterprint('done')
        else:
            # Save transparent 3D render
            plt.savefig(filename_component_alpha_part, transparent=True)
    # All 3D rendering done
    Barrier()
    # The partial 3D renders will now be combined into full 3D renders,
    # stored in the 'render3D_image', variable. Partial 3D renders of
    # the j'th component will be handled by the process with rank j.
    if not (nprocs == 1 == len(render3D_dict)):
        # Loop over components designated to each process
        for i in range(1 + len(render3D_dict)//nprocs):
            # Break out when there is no more work for this process
            j = rank + nprocs*i
            if j >= len(names):
                break
            name = names[j].replace(' ', '-')
            if nprocs == 1:
                # Simply load the already fully constructed image
                filename_component_alpha = '{}/{}_alpha.png'.format(render3D_dir, name)
                render3D_image = plt.imread(filename_component_alpha)
            else:
                # Create list of filenames for the partial 3D renders
                filenames_component_alpha_part = ['{}/{}_alpha_{}.png'
                                                  .format(render3D_dir, name, part)
                                                  for part in range(nprocs)]
                # Read in the partial 3D renders and blend
                # them together into the render3D_image variable.
                blend(filenames_component_alpha_part)
                # Save combined 3D render of the j'th component
                # with transparency. Theese are then later combined into
                # a 3D render containing all components.
                if len(names) > 1:
                    filename_component_alpha = '{}/{}_alpha.png'.format(render3D_dir, name)
                    plt.imsave(filename_component_alpha, render3D_image)
            # Add opaque background to render3D_image
            add_background()
            # Save combined 3D render of the j'th component
            # without transparency.
            filename_component = filename
            if len(names) > 1:
                if '_t=' in filename:
                    filename_component = filename.replace('_t=', '_{}_t='.format(name))
                elif '_a=' in filename:
                    filename_component = filename.replace('_a=', '_{}_a='.format(name))
                else:
                    filename_component = filename.replace('.png', '_{}.png'.format(name))
            plt.imsave(filename_component, render3D_image)
        Barrier()
        masterprint('done')
        # Finally, combine the full 3D renders of individual components
        # into a total 3D render containing all components.
        if master and len(names) > 1:
            masterprint('Combining component 3D renders and saving to "{}" ...'.format(filename))
            filenames_component_alpha = ['{}/{}_alpha.png'.format(render3D_dir,
                                                                  name.replace(' ', '-'))
                                         for name in names]
            blend(filenames_component_alpha)
            # Add opaque background to render3D_image and save it
            add_background()
            plt.imsave(filename, render3D_image)
            masterprint('done')
    # Remove the temporary directory, if cleanup is requested
    if master and cleanup and not (nprocs == 1 == len(render3D_dict)):
        shutil.rmtree(render3D_dir)
# Declare global variables used in the render3D function
cython.declare(render3D_dict=object,  # OrderedDict
               render3D_image='float[:, :, ::1]',
               )
# (Ordered) dictionary containing the figure, axes, component
# artist and text artist for each component.
render3D_dict = collections.OrderedDict()
# The array storing the 3D render
render3D_image = empty((render3D_resolution, render3D_resolution, 4), dtype=C2np['float'])
# Dummy function
def dummy_func(*args, **kwargs):
    return None

# Transformation function for orthographic projection
def orthographic_proj(zfront, zback):
    """This function is taken from
    http://stackoverflow.com/questions/23840756
    To replace the default 3D persepctive projection with
    3D orthographic perspective, simply write
    proj3d.persp_transformation = orthographic_proj
    where proj3d is imported from mpl_toolkits.mplot3d.
    """
    a = (zfront + zback)/(zfront - zback)
    b = -2*(zfront*zback)/(zfront - zback)
    return asarray([[1, 0,  0   , 0    ],
                    [0, 1,  0   , 0    ],
                    [0, 0,  a   , b    ],
                    [0, 0, -1e-6, zback],
                    ])

# Function which takes in a list of filenames of images and blend them
# together into the global render3D_image array.
@cython.header(# Arguments
               filenames=list,
               # Locals
               alpha_A='float',
               alpha_B='float',
               alpha_tot='float',
               i='int',
               j='int',
               rgb='int',
               rgbα='int',
               tmp_image='float[:, :, ::1]',
               )
def blend(filenames):
    # Make render3D_image black and transparent
    render3D_image[...] = 0
    for filename in filenames:
        tmp_image = plt.imread(filename)
        for     i in range(render3D_resolution):
            for j in range(render3D_resolution):
                # Pixels with 0 alpha has (r, g, b) = (1, 1, 1)
                # (this is a defect of plt.savefig).
                # These should be disregarded completely.
                alpha_A = tmp_image[i, j, 3]
                if alpha_A != 0:
                    # Combine render3D_image with tmp_image by
                    # adding them together, using their alpha values
                    # as weights.
                    alpha_B = render3D_image[i, j, 3]
                    alpha_tot = alpha_A + alpha_B - alpha_A*alpha_B
                    for rgb in range(3):
                        render3D_image[i, j, rgb] = (
                            (alpha_A*tmp_image[i, j, rgb] + alpha_B*render3D_image[i, j, rgb])
                            /alpha_tot
                        )
                    render3D_image[i, j, 3] = alpha_tot
    # Some pixel values in the combined 3D render may have overflown.
    # Clip at saturation value.
    for     i in range(render3D_resolution):
        for j in range(render3D_resolution):
            for rgbα in range(4):
                if render3D_image[i, j, rgbα] > 1:
                    render3D_image[i, j, rgbα] = 1

# Add background color to render3D_image
@cython.header(# Locals
               alpha='float',
               i='int',
               j='int',
               rgb='int',
               )
def add_background():
    for     i in range(render3D_resolution):
        for j in range(render3D_resolution):
            alpha = render3D_image[i, j, 3]
            # Add background using "A over B" alpha blending
            for rgb in range(3):
                render3D_image[i, j, rgb] = (
                    alpha*render3D_image[i, j, rgb] + (1 - alpha)*render3D_bgcolor[rgb]
                )
                render3D_image[i, j, 3] = 1
