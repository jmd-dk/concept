# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2019 Jeppe Mosgaard Dakin.
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
cimport('from mesh import diff_domain')
cimport('from communication import communicate_domain, get_buffer')
cimport('from graphics import plot_powerspec')
cimport('from mesh import CIC_components2φ, fft, slab_decompose')



# Function for computing power spectra of sets of components
@cython.pheader(# Arguments
                components=list,
                filename=str,
                # Locals
                W='double',
                a='double',
                any_fluid='bint',
                any_particles='bint',
                column_components=list,
                column_width_normal='Py_ssize_t',
                column_widths=list,
                component='Component',
                component_combination=tuple,
                component_combination_str=str,
                component_combinations=object,  # generator
                component_index='Py_ssize_t',
                component_indices_str=str,
                component_mapping=object,  # OrderedDict
                deconv_ijk='double',
                delimiter=str,
                fill_n_modes='bint',
                fmt=list,
                header=list,
                i='Py_ssize_t',
                index_largest_mode='Py_ssize_t',
                interpolation_quantities=list,
                j='Py_ssize_t',
                j_global='Py_ssize_t',
                k='Py_ssize_t',
                k_bin_index='Py_ssize_t',
                k_magnitude='double',
                k2='Py_ssize_t',
                kR='double',
                ki='Py_ssize_t',
                kj='Py_ssize_t',
                kj2='Py_ssize_t',
                kk='Py_ssize_t',
                longest_name_size='Py_ssize_t',
                max_n_modes='Py_ssize_t',
                nyquist='Py_ssize_t',
                power='double[::1]',
                power_dict=object,  # OrderedDict
                power_jik='double',
                reciprocal_sqrt_deconv_ij='double',
                reciprocal_sqrt_deconv_ijk='double',
                reciprocal_sqrt_deconv_j='double',
                representation=str,
                row_components=list,
                row_headings=list,
                row_σ=list,
                save_powerspecs='bint',
                size_i='Py_ssize_t',
                size_j='Py_ssize_t',
                size_k='Py_ssize_t',
                slab='double[:, :, ::1]',
                slab_dict=dict,
                slab_fluid='double[:, :, ::1]',
                slab_fluid_jik='double*',
                slab_particles='double[:, :, ::1]',
                slab_particles_jik='double*',
                spectrum_plural=str,
                symmetry_multiplicity='int',
                topline=list,
                Σmass='double',
                σ_dict=object,  # OrderedDict
                σ_str=str,
                φ='double[:, :, ::1]',
                φ_dict=dict,
                )
def powerspec(components, filename):
    # Always produce the powerspectrum at the current time
    a = universals.a
    # Ordered dicts storing the power and rms density variation,
    # with the component names as keys.
    power_dict = collections.OrderedDict()
    σ_dict = collections.OrderedDict()
    # Generator yielding tuples of all possible combinations
    # of the passed components.
    component_combinations = itertools.chain.from_iterable([itertools.combinations(components, i)
                                                            for i in range(1, len(components) + 1)
                                                            ])
    # Compute power spectrum for each combination of components,
    # if they have been selected for power spectrum compuation
    # (either of 'data' or 'plot') in the powerspec_select parameter.
    save_powerspecs = False
    for component_combination in component_combinations:
        if is_selected(component_combination, powerspec_data_select):
            save_powerspecs = True
        elif not is_selected(component_combination, powerspec_plot_select):
            # Neither dump data or plot for this combination
            continue
        component_combination_str = ', '.join(
            [component.name for component in component_combination]
            )
        if len(component_combination) > 1:
            component_combination_str = f'{{{component_combination_str}}}'
        masterprint(f'Computing power spectrum of {component_combination_str} ...')
        # Grab a designated buffer for the power spectrum
        # of this component, and store it in the power dict.
        power = get_buffer(k_bin_centers.shape[0], f'powerspec_{component_combination_str}',
                           nullify=True)
        power_dict[component_combination] = power
        # We now do the CIC interpolation of the component onto a grid
        # and perform the FFT on this grid. Here the φ grid is used.
        # We choose to interpolate the mass of each component onto
        # the grid. For both particle and fluid components, the total
        # mass can be computed by
        # Σmass = (a*boxsize)**3*ρ_bar
        #       = boxsize**3*a**(-3*w_eff)*ϱ_bar.
        # Since all particles have the same mass, the mass contribution
        # from a single particle is Σmass/component.N. For particles of
        # constant mass, this is exactly component.mass. In general,
        # we have Σmass/component.N = a**(-3*w_eff)*component.mass.
        # For fluids, each fluid element contributes to the mass by
        # an amount (a*L_cell)**3*ρ(x)
        #         = (a*boxsize/component.gridsize)**3*ρ(x)
        #         = (boxsize/component.gridsize)**3*a**(-3*w_eff)*ϱ(x).
        Σmass = ℝ[boxsize**3]*np.sum(
            [a**(-3*component.w_eff(a=a))*component.ϱ_bar for component in component_combination]
        )
        interpolation_quantities = [
            # Particle components
            ('particles', [
                component.mass*a**(-3*component.w_eff(a=a))
                for component in component_combination]
            ),
            # Fluid components
            ('ϱ', [
                (boxsize/component.gridsize)**3*a**(-3*component.w_eff(a=a))
                for component in component_combination]
            ),
        ]
        φ_dict = CIC_components2φ(list(component_combination), interpolation_quantities)
        # Flags specifying whether any
        # fluid/particle components are present.
        any_particles = ('particles' in φ_dict)
        any_fluid     = ('fluid'     in φ_dict)
        # Slab decompose the grids
        slab_dict = {
            representation: slab_decompose(φ, f'φ_{representation}_slab', prepare_fft=True)
            for representation, φ in φ_dict.items()
            }
        if any_fluid:
            slab_fluid = slab_dict['fluid']
        if any_particles:
            slab_particles = slab_dict['particles']
        # Do a forward in-place Fourier transform of the slabs
        for slab in slab_dict.values():
            fft(slab, 'forward')
            size_j, size_i, size_k = slab.shape[0], slab.shape[1], slab.shape[2]
        # Flag specifying whether or not n_modes has been computed
        fill_n_modes = (n_modes[0] == -1)
        if fill_n_modes:
            n_modes[0] = 0
        # Begin loop over slabs. As the first and second dimensions
        # are transposed due to the FFT, start with the j-dimension.
        nyquist = φ_gridsize//2
        for j in range(size_j):
            # The j-component of the wave vector (grid units).
            # Since the slabs are distributed along the j-dimension,
            # an offset must be used.
            j_global = ℤ[size_j*rank] + j
            if j_global > ℤ[φ_gridsize//2]:
                kj = j_global - φ_gridsize
            else:
                kj = j_global
            kj2 = kj**2
            # Reciprocal square root of the
            # j-component of the deconvolution.
            with unswitch(1):
                if any_particles:
                    reciprocal_sqrt_deconv_j = sinc(kj*ℝ[π/φ_gridsize])
            # Loop over the entire first dimension
            for i in range(φ_gridsize):
                # The i-component of the wave vector
                if i > ℤ[φ_gridsize//2]:
                    ki = i - φ_gridsize
                else:
                    ki = i
                # Reciprocal square root of the product of the i-
                # and the j-component of the deconvolution.
                with unswitch(2):
                    if any_particles:
                        reciprocal_sqrt_deconv_ij = (
                            sinc(ki*ℝ[π/φ_gridsize])*reciprocal_sqrt_deconv_j
                        )
                # Loop over the entire last dimension in steps of two,
                # as contiguous pairs of elements are the real and
                # imaginary part of the same complex number.
                for k in range(0, size_k, 2):
                    # The k-component of the wave vector
                    kk = k//2
                    # The squared magnitude of the wave vector
                    k2 = ℤ[ki**2 + kj2] + kk**2
                    # Skip the DC component.
                    # For some reason, the k = k_max mode is
                    # highly uncertain. Skip this as well.
                    if k2 == 0 or k2 == k2_max:
                        continue
                    # Get the bin index of this k²
                    k_bin_index = k_bin_indices[k2]
                    # Do the CIC deconvolution of the particles slab
                    with unswitch(3):
                        if any_particles:
                            # Pointer to the [j, i, k]'th element of the
                            # particles slab.
                            # The complex number is then given as
                            # Re = slab_particles_jik[0],
                            # Im = slab_particles_jik[1].
                            slab_particles_jik = cython.address(slab_particles[j, i, k:])
                            # Reciprocal square root of the product of
                            # all components of the deconvolution.
                            reciprocal_sqrt_deconv_ijk = (
                                reciprocal_sqrt_deconv_ij*sinc(kk*ℝ[π/φ_gridsize])
                            )
                            # The total factor
                            # for a complete deconvolution.
                            deconv_ijk = 1/reciprocal_sqrt_deconv_ijk**2
                            # Carry out the deconvolution
                            slab_particles_jik[0] *= deconv_ijk  # Real part
                            slab_particles_jik[1] *= deconv_ijk  # Imag part
                    # Get the total power of the [j, i, k]'th
                    # element of the slabs.
                    with unswitch(3):
                        if any_particles and any_fluid:
                            # Pointers to the [j, i, k]'th element of
                            # the particles and the fluid slab.
                            # The complex numbers are then given as e.g.
                            # Re = slab_particles_jik[0],
                            # Im = slab_particles_jik[1].
                            slab_particles_jik = cython.address(slab_particles[j, i, k:])
                            slab_fluid_jik     = cython.address(slab_fluid    [j, i, k:])
                            power_jik = (
                                  (slab_particles_jik[0] + slab_fluid_jik[0])**2
                                + (slab_particles_jik[1] + slab_fluid_jik[1])**2
                            )
                        elif any_particles:
                            # Pointer to the [j, i, k]'th element of the
                            # particles slab.
                            # The complex number is then given as
                            # Re = slab_particles_jik[0],
                            # Im = slab_particles_jik[1].
                            slab_particles_jik = cython.address(slab_particles[j, i, k:])
                            power_jik = slab_particles_jik[0]**2 + slab_particles_jik[1]**2
                        elif any_fluid:
                            # Pointer to the [j, i, k]'th element of the
                            # fluid slab.
                            # The complex number is then given as
                            # Re = slab_fluid_jik[0],
                            # Im = slab_fluid_jik[1].
                            slab_fluid_jik = cython.address(slab_fluid[j, i, k:])
                            power_jik = slab_fluid_jik[0]**2 + slab_fluid_jik[1]**2
                    # Because of the complex-conjugate symmetry,
                    # the slabs only contain the half with
                    # positive kk frequencies. Including this
                    # missing half lead to truer statistics,
                    # altering the binned power spectrum.
                    # Below, the symmetry_multiplicity
                    # variable counts the number of times this
                    # grid point should be counted.
                    if kk == 0 or kk == nyquist:
                        symmetry_multiplicity = 1
                    else:
                        symmetry_multiplicity = 2
                    # If the number of modes in each k bin has not
                    # been computed, do this now.
                    with unswitch(3):
                        if fill_n_modes:
                            # Increase the multiplicity
                            n_modes[k_bin_index] += symmetry_multiplicity
                    # Increase the power in this bin.
                    # For now, power holds the sum of powers.
                    power[k_bin_index] += symmetry_multiplicity*power_jik
        # Sum power into the master process
        Reduce(sendbuf=(MPI.IN_PLACE if master else power),
               recvbuf=(power        if master else None),
               op=MPI.SUM,
               )
        # If n_modes has just been computed,
        # sum the individual results into the master process.
        if fill_n_modes:
            Reduce(sendbuf=(MPI.IN_PLACE if master else n_modes),
                   recvbuf=(n_modes      if master else None),
                   op=MPI.SUM,
                   )
            # The maximm n_modes is used for formatting the output.
            # Store this as the additional, last element.
            n_modes[n_modes.shape[0] - 1] = max(n_modes)
        # The master process now holds all the information needed
        if not master:
            continue
        # Find the index of the largest populated mode. It is either
        # n_modes.shape[0] - 2 or n_modes.shape[0] - 3.
        for i in range(n_modes.shape[0] - 2, -1, -1):
            if n_modes[i] != 0:
                index_largest_mode = i
                break
        # We need to transform power from being the sum to being the
        # mean, by dividing by n_modes.
        # We want to normalize the power spectrum with respect to the
        # box volume. Since we interpolated the mass to the grid and
        # then square each grid value to compute the power,
        # the normalization will be boxsize**3/Σmass**2.
        for i in range(ℤ[index_largest_mode + 1]):
            power[i] *= ℝ[boxsize**3/Σmass**2]/n_modes[i]
        # Compute the rms density variation σ_R_tophat (usually σ₈).
        # This is given by
        # σ² = ∫d³k/(2π)³ W² power
        #    = 1/(2π)³∫_0^∞ dk 4πk² W² power
        #    = 1/(2π²)∫_0^∞ dk k² W² power,
        # where W = 3(sin(kR) - kR*cos(kR))/(kR)³.
        # Note that below, the factor 3² = 9 has been moved
        # outside of the integral (loop), and so W is really W/3.
        for i in range(ℤ[index_largest_mode + 1]):
            k_magnitude = k_bin_centers[i]
            kR = k_magnitude*R_tophat
            if kR < 1e-3:
                # In the limit of vanishing kR, W/3 tends to 1/3
                W = ℝ[1/3]
            else:
                W = (sin(kR) - kR*cos(kR))/kR**3
            σ2_integrand[i] = (k_magnitude*W)**2*power[i]
        # The integrand above starts from k = k_min, which means that
        # the interval from 0 to k_min has been left out. At k = 0,
        # the integrand vanishes. According to the trapezoidal rule,
        # this means that the full integral is missing the area of the
        # triangle with vertices (0, 0), (k_min, 0),
        # (k_min, σ2_integrand[0]), with k_min = k_bin_centers[0].
        σ_dict[component_combination] = np.sqrt(
            ℝ[9/(2*π**2)]*(  np.trapz(σ2_integrand [:ℤ[index_largest_mode + 1]],
                                      k_bin_centers[:ℤ[index_largest_mode + 1]])
                           + 0.5*k_bin_centers[0]*σ2_integrand[0]
                           )
            )
        # Done computing this power spectrum and its
        # associated rms density variation.
        masterprint('done')
    # If no power spectra has been computed, return now
    if not power_dict:
        return
    # Only the master process should write
    # power spectra to disk and do plotting.
    if not master:
        return
    # Trim the arrays inside power_dict so that they stop
    # at the largest populated mode.
    power_dict = {key: arr[:ℤ[index_largest_mode + 1]] for key, arr in power_dict.items()}
    # Regardless of the values in powerspec_data_select, all power
    # spectra are saved. The exception is when nothing should be saved,
    # in which case we really do not save anything.
    if save_powerspecs:
        # We want to save all power spectra to a single text file.
        # First we generate the header.
        spectrum_plural = 'spectrum' if len(power_dict) == 1 else 'spectra'
        masterprint(f'Saving power {spectrum_plural} to "{filename}" ...')
        # The top line of the header, stating general information
        topline = [
            f'Power {spectrum_plural} from CO𝘕CEPT job {jobid} '
            f'at t = {universals.t:.6g} {unit_time}, '
        ]
        if enable_Hubble:
            topline += [f'a = {a:.6g}, ']
        topline += [f'computed with a grid of linear size {φ_gridsize}.']
        # A column mapping each component to a number
        component_index = 0
        component_mapping = collections.OrderedDict()
        for component_combination in power_dict.keys():
            for component in component_combination:
                if component not in component_mapping:
                    component_mapping[component] = component_index
                    component_index += 1
        longest_name_size = np.max([len(component.name) for component in component_mapping])
        column_components = ['Below, the following component mapping is used:']
        for component, component_index in component_mapping.items():
            column_components.append(f'  {{:<{longest_name_size + 1}}} {component_index}'
                                     .format(component.name + ':')
                                     )
        # A row of component specifications
        row_components = ['', '']
        for component_combination in power_dict.keys():
            component_indices_str = get_integerset_strrep([component_mapping[component]
                                                           for component in component_combination])
            if len(component_combination) == 1:
                row_components.append(f'component {component_indices_str}')
            else:
                row_components.append(f'components {{{component_indices_str}}}')
        # A row of σ (rms density variation) values
        σ_unit = units.Mpc/(H0/(100*units.km/(units.s*units.Mpc))) if enable_Hubble else units.Mpc
        σ_str = ''.join([
            unicode('σ'),
            unicode_subscript(f'{R_tophat/σ_unit:.3g}'),
            ' = {:.6g}',
        ])
        row_σ = ['', '', *[σ_str.format(σ) for σ in σ_dict.values()]]
        # A row of column headings
        row_headings = [# Note: The extra spaces are used to counteract the inserted "# "
                        unicode(f'k [{unit_length}⁻¹]  '),
                        'modes',
                        *[unicode(f'power [{unit_length}³]')]*len(power_dict),
                        ]
        # Adjust rows based on the column widths
        max_n_modes = n_modes[n_modes.shape[0] - 1]
        column_width_normal = len(f'{0:.16e}')
        column_widths = [# k [Mpc⁻¹]
                         np.max([column_width_normal - 2, len(row_headings[0])]),
                         # n_modes
                         np.max([len(str(max_n_modes)), len(row_headings[1])]),
                         # power [Mpc⁻³]
                         *[np.max([column_width_normal,
                                   len(row_components[i]),
                                   len(row_headings  [i]),
                                   len(row_σ         [i]),
                                   ])
                           for i in range(2, 2 + len(power_dict))],
                         ]
        for i in range(len(row_components)):
            row_components[i] = f'{{:^{column_widths[i]}}}'.format(row_components[i])
            row_σ         [i] = f'{{:^{column_widths[i]}}}'.format(row_σ         [i])
            row_headings  [i] = f'{{:^{column_widths[i]}}}'.format(row_headings  [i])
        # Assemble the header from its pieces
        delimiter = '  '
        header = [unicode(line) for line in [
            ''.join(topline),
            *column_components,
            '',
            delimiter.join(row_components),
            delimiter.join(row_σ),
            delimiter.join(row_headings),
        ]]
        # Save header and power spectra data to text file
        fmt = [f'%-{column_width}{"u" if i == 1 else ".16e"}'
               for i, column_width in enumerate(column_widths)]
        np.savetxt(
            filename,
            asarray([
                k_bin_centers[:ℤ[index_largest_mode + 1]],
                n_modes      [:ℤ[index_largest_mode + 1]],
                *power_dict.values(),
                ]).transpose(),
            fmt=fmt,
            delimiter=delimiter,
            header='\n'.join(header),
            )
        masterprint('done')
    # Plot the power spectra
    plot_powerspec(
        k_bin_centers[:ℤ[index_largest_mode + 1]],
        power_dict,
        filename,
        powerspec_plot_select,
    )
# Initialize variables used for the power spectrum computation
# at import time, if such computation should ever take place.
cython.declare(i='Py_ssize_t',
               k_bin_center='double',
               k_bin_centers='double[::1]',
               k_bin_indices='Py_ssize_t[::1]',
               k_bin_size='double',
               k_max='double',
               k_min='double',
               k_magnitude='double',
               k2='Py_ssize_t',
               k2_max='Py_ssize_t',
               n_modes='Py_ssize_t[::1]',
               powerspec_data_select=dict,
               powerspec_plot_select=dict,
               σ2_integrand='double[::1]',
               )
if any(powerspec_times.values()) or special_params.get('special') == 'powerspec':
    # Construct the powerspec_data_select and powerspec_plot_select
    # dicts from the powerspec_select parameter.
    powerspec_data_select = {key: val['data'] for key, val in powerspec_select.items()}
    powerspec_plot_select = {key: val['plot'] for key, val in powerspec_select.items()}
    # Maximum value of k² (grid units)
    k2_max = 3*(φ_gridsize//2)**2
    # Maximum and minum k values
    k_min = ℝ[2*π/boxsize]
    k_max = ℝ[2*π/boxsize]*sqrt(k2_max)
    # Construct linear k bins, each with a size of k_min
    k_bin_size = k_min
    k_bin_centers = np.arange(k_min, k_max + k_bin_size, k_bin_size)
    # Construct array mapping k2 (grid units) to bin index
    k_bin_indices = empty(k2_max + 1, dtype=C2np['Py_ssize_t'])
    k_bin_indices[0] = 0
    i = 1
    for k2 in range(1, k_bin_indices.shape[0]):
        k_magnitude = ℝ[2*π/boxsize]*sqrt(k2)
        # Find index of closest bin center
        for i in range(i, ℤ[k_bin_centers.shape[0]]):
            k_bin_center = k_bin_centers[i]
            if k_bin_center > k_magnitude:
                # k2 belongs to either bin (i - 1) or bin i
                if k_magnitude - k_bin_centers[ℤ[i - 1]] < k_bin_center - k_magnitude:
                    k_bin_indices[k2] = ℤ[i - 1]
                else:
                    k_bin_indices[k2] = i
                break
    # Array counting the multiplicity (number of modes) in the bins.
    # One additional element is allocated, which will be used to store
    # the largest of all the other numbers.
    n_modes = zeros(k_bin_centers.shape[0] + 1, dtype=C2np['Py_ssize_t'])
    # The multiplicity of each bin is the same for all components and
    # constant throughout time. We therefore only need to compute
    # this once. Flag the first element so that we know it has not
    # been computed yet.
    n_modes[0] = -1
    # Array used for storing the integrand of σ²,
    # the squared rms density variation σ_R_tophat (usually σ₈).
    σ2_integrand = empty(k_bin_centers.shape[0], dtype=C2np['double'])

# Function which can measure different quantities of a passed component
@cython.header(
    # Arguments
    component='Component',
    quantity=str,
    # Locals
    J_arr=object, # np.ndarray
    J_noghosts='double[:, :, :]',
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
    j='Py_ssize_t',
    k='Py_ssize_t',
    mom='double*',
    mom2='double',
    mom2_max='double',
    mom2_i='double',
    momx='double*',
    momy='double*',
    momz='double*',
    mom_i='double',
    names=list,
    previous_result=tuple,
    v_rms='double',
    v_max='double',
    w='double',
    w_eff='double',
    Δdiff='double',
    Δdiff_max='double[::1]',
    Δdiff_max_dim='double',
    Δdiff_max_list=list,
    Δdiff_max_normalized_list=list,
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
    ΣJ_over_ϱ_plus_𝒫_2='double',
    Σmass='double',
    Σmom='double[::1]',
    Σmom_dim='double',
    Σmom2_dim='double',
    Σϱ='double',
    Σϱ2='double',
    ϱ='FluidScalar',
    ϱ_arr=object,  # np.ndarray
    ϱ_bar='double',
    ϱ_min='double',
    ϱ_mv='double[:, :, ::1]',
    ϱ_noghosts='double[:, :, :]',
    ϱ_ptr='double*',
    σ2mom_dim='double',
    σ2ϱ='double',
    σmom='double[::1]',
    σmom_dim='double',
    σϱ='double',
    𝒫_mv='double[:, :, ::1]',
    𝒫_ptr='double*',
    returns=object,  # double or tuple
)
def measure(component, quantity):
    """Implemented quantities are:
    'v_max'
    'v_rms'
    'momentum'
    'ϱ'              (fluid quantity)
    'mass'           (fluid quantity)
    'discontinuity'  (fluid quantity)
    """
    a = universals.a
    # Check cache for result
    previous_result = measurements.get((component, quantity), ())
    if previous_result and previous_result[0] == a:
        return previous_result[1]
    # Extract variables
    N = component.N
    N_elements = component.gridsize**3
    Vcell = boxsize**3/N_elements
    w     = component.w    (a=a)
    w_eff = component.w_eff(a=a)
    ϱ = component.ϱ
    ϱ_noghosts = ϱ.grid_noghosts
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
        # In the case of decyaing (matter) particles, the mass at time a
        # is really a**(-3*w_eff)*mass, and so we get
        # v = mom/(a**(2 - 3*w_eff)*mass)
        if component.representation == 'particles':
            mom2_max = 0
            momx = component.momx
            momy = component.momy
            momz = component.momz
            for i in range(component.N_local):
                mom2_i = momx[i]**2 + momy[i]**2 + momz[i]**2
                if mom2_i > mom2_max:
                    mom2_max = mom2_i
            mom2_max = allreduce(mom2_max, op=MPI.MAX)
            v_max = sqrt(mom2_max)/(a**(2 - 3*w_eff)*component.mass)
        elif component.representation == 'fluid':
            if (    component.boltzmann_order == -1
                or (component.boltzmann_order == 0 and component.boltzmann_closure == 'truncate')
                ):
                # Without J as a fluid variable, no velocity exists
                # and so no Courant limit needs to be set.
                v_max = 0
            elif component.boltzmann_order == 0 and component.boltzmann_closure == 'class':
                # With J as a linear fluid variable, we only need to
                # consider one of its components. Also, the P = wρ
                # approximation is guaranteed to be enabled.
                ϱ_ptr  = component.ϱ .grid
                Jx_ptr = component.Jx.grid
                J_over_ϱ_2_max = 0
                for i in range(component.size):
                    J_over_ϱ_2_i = (Jx_ptr[i]/ϱ_ptr[i])**2
                    if J_over_ϱ_2_i > J_over_ϱ_2_max:
                        J_over_ϱ_2_max = J_over_ϱ_2_i
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
                for i in range(component.size):
                    J_over_ϱ_plus_𝒫_2_i = (
                        (Jx_ptr[i]**2 + Jy_ptr[i]**2 + Jz_ptr[i]**2)
                        /(ϱ_ptr[i] + ℝ[light_speed**(-2)]*𝒫_ptr[i])**2
                    )
                    if J_over_ϱ_plus_𝒫_2_i > J_over_ϱ_plus_𝒫_2_max:
                        J_over_ϱ_plus_𝒫_2_max = J_over_ϱ_plus_𝒫_2_i
                J_over_ϱ_plus_𝒫_2_max = allreduce(J_over_ϱ_plus_𝒫_2_max, op=MPI.MAX)
                v_max = a**(3*w_eff - 2)*sqrt(J_over_ϱ_plus_𝒫_2_max)
                # Add the sound speed. When the P=wρ approxiamation is
                # False, the sound speed is non-global and given by the
                # square root of δ𝒫/δϱ. However, constructing δ𝒫/δϱ
                # locally from the ϱ and 𝒫 grids leads to large
                # numerical errors. Regardless of whether the P=wρ
                # approximation is used or not, we simply use the
                # global sound speed.
                v_max += light_speed*sqrt(w)/a
        # Store result in cache and return
        measurements[component, quantity] = (a, v_max)
        return v_max
    elif quantity == 'v_rms':
        if component.representation == 'particles':
            mom2 = 0
            momx = component.momx
            momy = component.momy
            momz = component.momz
            for i in range(component.N_local):
                mom2 += momx[i]**2 + momy[i]**2 + momz[i]**2
            mom2 = allreduce(mom2, op=MPI.SUM)
            v_rms = sqrt(mom2/component.N)/(a**(2 - 3*component.w_eff(a=a))*component.mass)
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
                for         i in range(2, ℤ[component.shape[0] - 2 - 1]):
                    for     j in range(2, ℤ[component.shape[1] - 2 - 1]):
                        for k in range(2, ℤ[component.shape[2] - 2 - 1]):
                            ΣJ_over_ϱ_plus_𝒫_2 += 3*(Jx_mv[i, j, k]/(ϱ_mv[i, j, k]*(1 + w)))**2
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
                for         i in range(2, ℤ[component.shape[0] - 2 - 1]):
                    for     j in range(2, ℤ[component.shape[1] - 2 - 1]):
                        for k in range(2, ℤ[component.shape[2] - 2 - 1]):
                            ΣJ_over_ϱ_plus_𝒫_2 += (
                                (Jx_mv[i, j, k]**2 + Jy_mv[i, j, k]**2 + Jz_mv[i, j, k]**2)
                                /(ϱ_mv[i, j, k] + ℝ[light_speed**(-2)]*𝒫_mv[i, j, k])**2
                            )
                ΣJ_over_ϱ_plus_𝒫_2 = allreduce(ΣJ_over_ϱ_plus_𝒫_2, op=MPI.SUM)
                v_rms = a**(3*w_eff - 2)*sqrt(ΣJ_over_ϱ_plus_𝒫_2/N_elements)
                # Add the sound speed. When the P=wρ approxiamation is
                # False, the sound speed is non-global and given by the
                # square root of δ𝒫/δϱ. However, constructing δ𝒫/δϱ
                # locally from the ϱ and 𝒫 grids leads to large
                # numerical errors. Regardless of whether the P=wρ
                # approximation is used or not, we simply use the
                # global sound speed.
                v_rms += light_speed*sqrt(w)/a
        # Store result in cache and return
        measurements[component, quantity] = (a, v_rms)
        return v_rms
    elif quantity == 'momentum':
        Σmom = empty(3, dtype=C2np['double'])
        σmom = empty(3, dtype=C2np['double'])
        if component.representation == 'particles':
            # Total momentum of all particles, for each dimension
            for dim in range(3):
                mom = component.mom[dim]
                Σmom_dim = Σmom2_dim = 0
                # Add up local particle momenta
                for i in range(component.N_local):
                    mom_i = mom[i]
                    Σmom_dim  += mom_i
                    Σmom2_dim += mom_i**2
                # Add up local particle momenta sums
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
            # it is in correspondance with the momentum definition
            # for particles.
            for dim, fluidscalar in enumerate(component.J):
                # NumPy array of local part of J with no pseudo points
                J_noghosts = fluidscalar.grid_noghosts
                J_arr = asarray(J_noghosts[:(J_noghosts.shape[0] - 1),
                                           :(J_noghosts.shape[1] - 1),
                                           :(J_noghosts.shape[2] - 1)])
                # Total dim'th momentum of all fluid elements
                Σmom_dim = np.sum(J_arr)*Vcell
                # Total dim'th momentum squared of all fluid elements
                Σmom2_dim = np.sum(J_arr**2)*Vcell**2
                # Add up local fluid element momenta sums
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
        # Store result in cache and return
        measurements[component, quantity] = (a, (Σmom, σmom))
        return Σmom, σmom
    # Fluid quantities
    elif quantity == 'ϱ':
        # Compute mean(ϱ), std(ϱ), min(ϱ)
        if component.representation == 'particles':
            # Particle components have no ϱ
            abort('The measure function was called with the "{}" component with '
                  'quantity=\'ϱ\', but particle components do not have ϱ.'
                  .format(component.name)
                  )
        elif component.representation == 'fluid':
            # NumPy array of local part of ϱ with no pseudo points
            ϱ_arr = asarray(ϱ_noghosts[:(ϱ_noghosts.shape[0] - 1),
                                       :(ϱ_noghosts.shape[1] - 1),
                                       :(ϱ_noghosts.shape[2] - 1)])
            # Total ϱ of all fluid elements
            Σϱ = np.sum(ϱ_arr)
            # Total ϱ² of all fluid elements
            Σϱ2 = np.sum(ϱ_arr**2)
            # Add up local sums
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
            ϱ_min = allreduce(np.min(ϱ_arr), op=MPI.MIN)
        # Store result in cache and return
        measurements[component, quantity] = (a, (ϱ_bar, σϱ, ϱ_min))
        return ϱ_bar, σϱ, ϱ_min
    elif quantity == 'mass':
        if component.representation == 'particles':
            # Any change in the mass of particle a component is absorbed
            # into w_eff(a).
            Σmass = a**(-3*w_eff)*component.N*component.mass
        elif component.representation == 'fluid':
            # NumPy array of local part of ϱ with no pseudo points
            ϱ_arr = asarray(ϱ_noghosts[:(ϱ_noghosts.shape[0] - 1),
                                       :(ϱ_noghosts.shape[1] - 1),
                                       :(ϱ_noghosts.shape[2] - 1)])
            # Total ϱ of all fluid elements
            Σϱ = np.sum(ϱ_arr)
            # Add up local sums
            Σϱ = allreduce(Σϱ, op=MPI.SUM)
            # The total mass is
            # Σmass = (a**3*Vcell)*Σρ
            # where a**3*Vcell is the proper volume and Σρ is the sum of
            # proper densities. In terms of the fluid variable
            # ϱ = a**(3*(1 + w_eff))*ρ, the total mass is then
            # Σmass = a**(-3*w_eff)*Vcell*Σϱ.
            # Note that the total mass is generally constant.
            Σmass = a**(-3*w_eff)*Vcell*Σϱ
        # Store result in cache and return
        measurements[component, quantity] = (a, Σmass)
        return Σmass
    elif quantity == 'discontinuity':
        if component.representation == 'particles':
            # Particle components have no discontinuity
            abort('The measure function was called with the "{}" component with '
                  'quantity=\'discontinuity\', which is not applicable to particle components.'
                  .format(component.name)
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
                # Communicate pseudo and ghost points of the grid
                communicate_domain(fluidscalar.grid_mv, mode='populate')
                # Differentiate the grid in all three directions via
                # both forward and backward difference. For each
                # direction, save the largest difference between
                # the two. Also save the largest differential in
                # each direction.
                Δdiff_max = empty(3, dtype=C2np['double'])
                diff_max = empty(3, dtype=C2np['double'])
                for dim in range(3):
                    # Do the differentiations
                    diff_forward  = diff_domain(fluidscalar.grid_mv, dim, h, 0, order=1, direction='forward')
                    diff_backward = diff_domain(fluidscalar.grid_mv, dim, h, 1, order=1, direction='backward')
                    # Find the largest difference between the results of the
                    # forward and backward difference,
                    Δdiff_max_dim = 0
                    diff_max_dim = 0
                    for         i in range(ℤ[ϱ_noghosts.shape[0] - 1]):
                        for     j in range(ℤ[ϱ_noghosts.shape[1] - 1]):
                            for k in range(ℤ[ϱ_noghosts.shape[2] - 1]):
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
                    Δdiff_max_dim = allreduce(Δdiff_max_dim, op=MPI.MAX)
                    diff_max_dim  = allreduce(diff_max_dim,  op=MPI.MAX)
                    # Pack results into lists
                    Δdiff_max[dim] = Δdiff_max_dim
                    diff_max[dim] = diff_max_dim
                Δdiff_max_list.append(Δdiff_max)
                # Maximum discontinuity (difference between forward and
                # backward difference) normalized accoring to
                # the largest slope.
                Δdiff_max_normalized_list.append(np.array([Δdiff_max[dim]/diff_max[dim]
                                                           if Δdiff_max[dim] > 0 else 0
                                                           for dim in range(3)
                                                           ], dtype=C2np['double'],
                                                          )
                                                 )
        # Store result in cache and return
        measurements[component, quantity] = (a, (names, Δdiff_max_list, Δdiff_max_normalized_list))
        return names, Δdiff_max_list, Δdiff_max_normalized_list
    elif master:
        abort('The measure function was called with quantity=\'{}\', which is not implemented'
              .format(quantity))
# Cache used by the measure function
cython.declare(measurements=dict)
measurements = {}

# Function for doing debugging analysis
@cython.header(# Arguments
               components=list,
               # Locals
               component='Component',
               dim='int',
               name=str,
               w_eff='double',
               Δdiff_max='double[::1]',
               Δdiff_max_normalized='double[::1]',
               Σmom='double[::1]',
               Σmom_prev_dim='double',
               ϱ_bar='double',
               ϱ_min='double',
               σmom='double[::1]',
               σϱ='double',
               )
def debug(components):
    """This function will compute many different quantities from the
    component data and print out the results. Warnings will be given for
    obviously erroneous results.
    """
    # Componentwise analysis
    for component in components:
        w_eff = component.w_eff()
        # sum(momentum) and std(momentum) in each dimension
        Σmom, σmom = measure(component, 'momentum')
        for dim in range(3):
            debug_print('total {}-momentum'.format('xyz'[dim]),
                        component,
                        Σmom[dim],
                        'm☉ Mpc Gyr⁻¹',
                        )
            debug_print('standard deviation of {}-momentum'.format('xyz'[dim]),
                        component,
                        σmom[dim],
                        'm☉ Mpc Gyr⁻¹',
                        )
        # Warn if sum(momentum) does not agree with previous measurement
        if component.name in Σmom_prev:
            for dim in range(3):
                Σmom_prev_dim = Σmom_prev[component.name][dim]
                if not isclose(Σmom_prev_dim, Σmom[dim],
                               rel_tol=1e-6,
                               abs_tol=1e-6*σmom[dim],
                               ):
                    masterwarn('Previously the "{}" had a '
                               'total {}-momentum of {} m☉ Mpc Gyr⁻¹'
                               .format(component.name,
                                       'xyz'[dim],
                                       significant_figures(Σmom_prev_dim
                                                           /(units.m_sun*units.Mpc/units.Gyr),
                                                           12,
                                                           fmt='unicode',
                                                           incl_zeros=False,
                                                           scientific=True,
                                                           ),
                                       )
                               )
        Σmom_prev[component.name] = asarray(Σmom).copy()
        # mean(ϱ), std(ϱ) and min(ϱ)
        if component.representation == 'fluid':
            ϱ_bar, σϱ, ϱ_min = measure(component, 'ϱ')
            debug_print('mean ϱ',
                        component,
                        ϱ_bar,
                        'm☉ Mpc⁻³',
                        )
            debug_print('standard deviation of ϱ',
                        component,
                        σϱ,
                        'm☉ Mpc⁻³',
                        )
            debug_print('minimum ϱ',
                        component,
                        ϱ_min,
                        'm☉ Mpc⁻³',
                        )
            # Warn if any densities are negative
            if ϱ_min < 0:
                masterwarn('Negative density occured in "{}"'.format(component.name))
            # Warn if mean(ϱ) differs from the correct, constant result
            if not isclose(ϱ_bar, cast(component.ϱ_bar, 'double'), rel_tol=1e-6):
                masterwarn('The "{}" ought to have a mean ϱ of {} m☉ Mpc⁻³'
                           .format(component.name,
                                   significant_figures(component.ϱ_bar/(units.m_sun/units.Mpc**3),
                                                       12,
                                                       fmt='unicode',
                                                       incl_zeros=False,
                                                       scientific=True,
                                                       ),
                                   )
                    )
        # The maximum discontinuities in the fluid scalars,
        # for each dimension. Here, a discontinuity means a difference
        # in forward and backward difference.
        if component.representation == 'fluid':
            for name, Δdiff_max, Δdiff_max_normalized in zip(*measure(component, 'discontinuity')):
                for dim in range(3):
                    debug_print('maximum            {}-discontinuity in {}'.format('xyz'[dim], name),
                                component,
                                Δdiff_max[dim],
                                'Mpc⁻¹',
                                )
                    debug_print('maximum normalized {}-discontinuity in {}'.format('xyz'[dim], name),
                                component,
                                Δdiff_max_normalized[dim],
                                )
# Dict storing sum of momenta for optained in previous call to the
# debug function, for all components.
cython.declare(Σmom_prev=dict)
Σmom_prev = {}

# Function for printing out debugging info,
# used in the debug function above.
@cython.header(# Arguments
               quantity=str,
               component='Component',
               value='double',
               unit_str=str,
               # Locals
               text=str,
               unit='double',
               value_str=str,
               )
def debug_print(quantity, component, value, unit_str='1'):
    unit = eval_unit(unit_str)
    value_str = significant_figures(value/unit,
                                    12,
                                    fmt='unicode',
                                    incl_zeros=False,
                                    scientific=True,
                                    )
    text = '{} {}({}) = {}{}'.format(terminal.bold_cyan('Debug info:'),
                                     quantity[0].upper() + quantity[1:],
                                     component.name,
                                     value_str,
                                     ' ' + unit_str if unit_str != '1' else '',
                                     )
    masterprint(text)
