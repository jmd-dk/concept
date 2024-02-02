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
    'from mesh import          '
    '    domain_decompose,     '
    '    fft,                  '
    '    fill_slab_padding,    '
    '    interpolate_upstream, '
    '    resize_grid,          '
    '    slab_loop,            '
)

# Pure Python imports
from communication import get_domain_info



# Function for plotting an already computed power spectrum
# and saving an image file to disk.
def plot_powerspec(declaration, filename):
    if not master:
        return
    # Recursive dispatch
    if isinstance(declaration, list):
        declarations = declaration
    else:
        declarations = [declaration]
    declarations = [
        declaration
        for declaration in declarations
        if declaration.do_plot
    ]
    if not declarations:
        return
    if len(declarations) > 1:
        for declaration in declarations:
            # Since we have multiple plots --- one for each
            # set of components --- we augment each filename
            # with this information.
            plot_powerspec(
                declaration,
                augment_filename(
                    filename,
                    '_'.join([
                        component.name.replace(' ', '-')
                        for component in declaration.components
                    ]),
                    '.png',
                )
            )
        return
    declaration = declarations[0]
    # Fetch Matplotlib
    plt = get_matplotlib().pyplot
    # Ensure correct filename extension
    if not filename.endswith('.png'):
        filename += '.png'
    # Extract variables
    components      = declaration.components
    k_bin_centers   = asarray(declaration.k_bin_centers)
    power           = asarray(declaration.power)
    power_corrected = declaration.power_corrected
    power_linear    = declaration.power_linear
    if power_corrected is not None:
        power_corrected = asarray(power_corrected)
    if power_linear is not None:
        power_linear = asarray(power_linear)
    # Begin progress message
    if len(components) == 1:
        components_str = components[0].name
    else:
        components_str = '{{{}}}'.format(
            ', '.join([component.name for component in components])
        )
    masterprint(
        f'Plotting power spectrum of {components_str} and saving to "{filename}" ...'
    )
    # Plot power spectrum in new figure
    fig, ax = plt.subplots()
    label = 'simulation'
    if np.any(power):
        ax.loglog(k_bin_centers, power, '-', label=label)
    else:
        # The odd case of no power at all
        ax.semilogx(k_bin_centers, power, '-', label=label)
    nlines = 1
    # Also plot corrected power spectrum, if specified
    if power_corrected is not None:
        nlines += 1
        label = 'corrected'
        if np.any(power_corrected) and np.any(~np.isnan(power_corrected)):
            ax.loglog(k_bin_centers, power_corrected, '--', label=label)
        else:
            # The odd case of no corrected power at all
            ax.semilogx(k_bin_centers, power_corrected, '--', label=label)
    # Also plot linear power spectra, if specified
    if power_linear is not None:
        nlines += 1
        label = 'linear'
        label_attrs = []
        gauge = collections.Counter(
            [component.realization_options['gauge'] for component in components]
        ).most_common(1)[0][0]
        gauge = {
            'nbody'      : r'$N$-body',
            'synchronous': r'synchronous',
            'newtonian'  : r'Newtonian',
        }.get(gauge, gauge)
        label_attrs.append(f'{gauge} gauge')
        backscale = collections.Counter(
            [component.realization_options['backscale'] for component in components]
        ).most_common(1)[0][0]
        if backscale:
            label_attrs.append('back-scaled')
        if label_attrs:
            label += ' ({})'.format(', '.join(label_attrs))
        ylim = ax.get_ylim()
        if np.any(power_linear) and np.any(~np.isnan(power_linear)):
            ax.loglog(k_bin_centers, power_linear, 'k--', lw=1, label=label)
        else:
            # The odd case of no linear power at all
            ax.semilogx(k_bin_centers, power_linear, 'k--', lw=1, label=label)
        ax.set_ylim(ylim)
    if nlines > 1:
        ax.legend(fontsize=14)
    ax.set_xlim(k_bin_centers[0], k_bin_centers[-1])
    # Finishing touches
    ax.set_xlabel(rf'$k$ $[\mathrm{{{unit_length}}}^{{-1}}]$', fontsize=14)
    ax.set_ylabel(rf'$P$ $[\mathrm{{{unit_length}}}^3]$',      fontsize=14)
    t_str = (
        rf'$t = {{}}\, \mathrm{{{{{unit_time}}}}}$'
        .format(significant_figures(universals.t, 4, fmt='TeX'))
    )
    a_str = ''
    if enable_Hubble:
        a_str = ', $a = {}$'.format(significant_figures(universals.a, 4, fmt='TeX'))
    components_str = (
        components_str
        .replace('{', r'$\{$')
        .replace('}', r'$\}$')
    )
    ax.set_title(f'{components_str}\n{t_str}{a_str}', fontsize=16, horizontalalignment='center')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    # Done with this plot.
    # Close the figure, leaving no trace in memory of the plot.
    plt.close(fig)
    masterprint('done')

# Function for plotting an already computed bispectrum
# and saving an image file to disk.
def plot_bispec(declaration, filename):
    if not master:
        return
    # Recursive dispatch
    if isinstance(declaration, list):
        declarations = declaration
    else:
        declarations = [declaration]
    declarations = [
        declaration
        for declaration in declarations
        if declaration.do_plot
    ]
    if not declarations:
        return
    if len(declarations) > 1:
        for declaration in declarations:
            # Since we have multiple plots --- one for each
            # set of components --- we augment each filename
            # with this information.
            plot_bispec(
                declaration,
                augment_filename(
                    filename,
                    '_'.join([
                        component.name.replace(' ', '-')
                        for component in declaration.components
                    ]),
                    '.png',
                )
            )
        return
    declaration = declarations[0]
    do_treelevel = declaration.do_treelevel
    do_reduced = declaration.do_reduced
    # Fetch Matplotlib
    matplotlib = get_matplotlib()
    plt = matplotlib.pyplot
    # Ensure correct filename extension
    if not filename.endswith('.png'):
        filename += '.png'
    # Begin progress message
    components = declaration.components
    if len(components) == 1:
        components_str = components[0].name
    else:
        components_str = '{{{}}}'.format(
            ', '.join([component.name for component in components])
        )
    masterprint(
        f'Plotting bispectrum of {components_str} and saving to "{filename}" ...'
    )
    # Extract data and mask out NaN values
    bpower = asarray(declaration.bpower)
    bpower[bpower == 0] = NaN
    mask = ~np.isnan(bpower)
    bpower = asarray(bpower)[mask]
    if bpower.shape[0] == 0:
        masterwarn('No bispectrum data to be plotted')
        masterprint('done')
        return
    k = asarray([bin.k for bin in declaration.bins])[mask]
    t = asarray([bin.t for bin in declaration.bins])[mask]
    μ = asarray([bin.μ for bin in declaration.bins])[mask]
    if do_treelevel:
        bpower_treelevel = asarray(declaration.bpower_treelevel)
        mask_treelevel = ~np.isnan(bpower_treelevel)
        bpower_treelevel = bpower_treelevel[mask_treelevel]
        if bpower_treelevel.shape[0] == 0:
            do_treelevel = False
        else:
            k_treelevel = asarray([bin.k for bin in declaration.bins])[mask_treelevel]
            t_treelevel = asarray([bin.t for bin in declaration.bins])[mask_treelevel]
            μ_treelevel = asarray([bin.μ for bin in declaration.bins])[mask_treelevel]
    if do_reduced:
        bpower_reduced = asarray(declaration.bpower_reduced)[mask]
        if do_treelevel:
            bpower_reduced_treelevel = asarray(
                declaration.bpower_reduced_treelevel
            )[mask_treelevel]
    # Specifications for each bispectrum parameter
    bins_data = {
        'k': k,
        't': t,
        'μ': μ,
    }
    if do_treelevel:
        bins_data_treelevel = {
            'k': k_treelevel,
            't': t_treelevel,
            'μ': μ_treelevel,
        }
    axis_scales = {
        'k': 'log',
        't': 'linear',
        'μ': 'linear',
    }
    axis_labels = {
        'k': rf'$k$ $[\mathrm{{{unit_length}}}^{{-1}}]$',
        't': r'$t$',
        'μ': r'$\mu$',
        'B': rf'$B$ $[\mathrm{{{unit_length}}}^6]$',
        'Q': rf'$Q$',
    }
    # Labels
    label = 'simulation'
    if do_treelevel:
        gauge = collections.Counter(
            [component.realization_options['gauge'] for component in components]
        ).most_common(1)[0][0]
        gauge = {
            'nbody'      : r'$N$-body',
            'synchronous': r'synchronous',
            'newtonian'  : r'Newtonian',
        }.get(gauge, gauge)
        backscale = collections.Counter(
            [component.realization_options['backscale'] for component in components]
        ).most_common(1)[0][0]
        backscale_str = ', back-scaled'*backscale
        label_treelevel = f'tree-level ({gauge} gauge{backscale_str})'
    # Helper functions
    def add_param_text(ax, *param_names):
        if degenerate_3D:
            return
        text = []
        for other_param_name, other_param_vals in bins_data.items():
            if other_param_name in param_names:
                continue
            val_str = significant_figures(np.mean(other_param_vals), 3, fmt='TeX')
            text.append(
                re.subn(
                    r'(\$ )|(\$$)',
                    rf' = {val_str}$ '.replace('\\', '\\\\'),
                    axis_labels[other_param_name],
                    1,
                )[0]
            )
        ax.text(
            0.05, 0.05, '\n'.join(text),
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=14,
        )
    def get_logc(data, fac=0.85, threshold=1e+2):
        data_min, data_max = np.min(data), np.max(data)
        data = data[(data_min/fac < data) & (data < data_max*fac)]
        if len(data) < 3:
            return False
        data_min, data_max = np.min(data), np.max(data)
        return (data_max/data_min > threshold)
    def get_logticks(xdata, ydata):
        fig_tmp, ax_tmp = plt.subplots()
        if axis_scales[dimensions[0]] == 'log':
            xdata = 10**xdata
        if axis_scales[dimensions[1]] == 'log':
            ydata = 10**ydata
        ax_tmp.tripcolor(xdata, ydata, bpower)
        if axis_scales[dimensions[0]] == 'log':
            ax_tmp.set_xscale('log')
        if axis_scales[dimensions[1]] == 'log':
            ax_tmp.set_yscale('log')
        x_min, x_max = np.min(xdata), np.max(xdata)
        y_min, y_max = np.min(ydata), np.max(ydata)
        ticks = {
            'x': [x for x in ax_tmp.get_xticks() if x_min <= x <= x_max],
            'y': [y for y in ax_tmp.get_yticks() if y_min <= y <= y_max],
        }
        plt.close(fig_tmp)
        for key, val in ticks.items():
            if not val:
                continue
            ticks[key] = correct_float(np.concatenate(([1e-1*val[0]], val, [1e+1*val[-1]])))
        return ticks
    def log_tick_formatter(val, pos=None):
        """Log scaling does not work for 3D axes nor for the Gouraud
        shading. In these cases we instead explicitly plot log10(k).
        To now display the tick labels properly, we need a custom
        tick formatter function (this function).
        For the 3D axis issue, see
          https://github.com/matplotlib/matplotlib/issues/209
        """
        ticklabel = significant_figures(10**val, 2, fmt='TeX', incl_zeros=False)
        return f'${ticklabel}$'
    # Determine what to plot
    dimensions = []
    degenerate_3D = False
    if bpower.shape[0] > 1:
        if not np.all(np.isclose(k, k[0], 1e-4, 0)):
            dimensions.append('k')
        if not np.all(np.isclose(t, t[0], 1e-4, 0)):
            dimensions.append('t')
        if not np.all(np.isclose(μ, μ[0], 1e-4, 0)):
            dimensions.append('μ')
        # Though all of k, t, μ vary, t and μ might be strictly related,
        # in which case we should not treat both as independent.
        if bpower.shape[0] > 2 and len(dimensions) == 3 and np.min(t) >= 0 and np.min(μ) >= 0:
            tμ_pos_mask = (t > 0) & (μ > 0)
            if np.sum(tμ_pos_mask) > 2:
                t_pos = t[tμ_pos_mask]
                μ_pos = μ[tμ_pos_mask]
                poly_coeffs = np.polyfit(np.log(t_pos), np.log(μ_pos), 1)
                if np.max(np.abs(np.exp(poly_coeffs[0]*np.log(t_pos) + poly_coeffs[1]) - μ_pos)) < 1e-4:
                    degenerate_3D = True
                    if bispec_plot_prefer == 't':
                        dimensions.remove('μ')
                    else:
                        dimensions.remove('t')
    # Plot bispectrum in new figure
    ticklabelsizefac = 1
    if len(dimensions) == 0:
        # Single dot
        fig, axes = plt.subplots(1 + do_reduced, 1, sharex=True)
        axes = any2list(axes)
        param_name = 'k'
        xdata = bins_data[param_name]
        if do_treelevel:
            xdata_treelevel = bins_data_treelevel[param_name]
        xmin = np.min((np.min(xdata), 2*π/boxsize))
        xmax = np.max((np.max(xdata), 2*π/boxsize*sqrt(3)*(declaration.gridsize//2)))
        axes[0].plot(xdata, bpower, '.', label=label)
        if do_treelevel:
            axes[0].plot(xdata_treelevel, bpower_treelevel, 'k.', label=label_treelevel)
        if do_reduced:
            axes[1].plot(xdata, bpower_reduced, '.', label=label)
            if do_treelevel:
                axes[1].plot(xdata_treelevel, bpower_reduced_treelevel, 'k.', label=label_treelevel)
        axes[ 0].set_xscale(axis_scales[param_name])
        axes[ 0].set_xlim(xmin, xmax)
        axes[-1].set_xlabel(axis_labels[param_name], fontsize=14)
        axes[ 0].set_ylabel(axis_labels['B'], fontsize=14)
        if do_reduced:
            axes[1].set_ylabel(axis_labels['Q'], fontsize=14)
        add_param_text(axes[0], param_name)
    elif len(dimensions) == 1:
        # 1D line plot
        fig, axes = plt.subplots(1 + do_reduced, 1, sharex=True)
        axes = any2list(axes)
        param_name = dimensions[0]
        xdata = bins_data[param_name]
        sorting = np.argsort(xdata)
        xdata = xdata[sorting]
        xmin, xmax = np.min(xdata), np.max(xdata)
        if do_treelevel:
            xdata_treelevel = bins_data_treelevel[param_name]
            sorting_treelevel = np.argsort(xdata_treelevel)
            xdata_treelevel = xdata_treelevel[sorting_treelevel]
        bpower_negnan = bpower.copy()
        bpower_negnan[bpower < 0] = NaN
        axes[0].semilogy(xdata, np.abs(bpower[sorting]), '--')
        axes[0].set_prop_cycle(None)
        axes[0].semilogy(xdata, np.abs(bpower_negnan[sorting]), '-', label=label)
        if do_treelevel:
            ylim = axes[0].get_ylim()
            axes[0].semilogy(
                xdata_treelevel, bpower_treelevel[sorting_treelevel], 'k--',
                label=label_treelevel,
            )
            axes[0].set_ylim(ylim)
        if do_reduced:
            axes[1].plot(xdata, bpower_reduced[sorting], '-', label=label)
            ylim = axes[1].get_ylim()
            if do_treelevel:
                axes[1].plot(
                    xdata_treelevel, bpower_reduced_treelevel[sorting_treelevel], 'k--',
                    label=label_treelevel,
                )
            axes[1].plot(
                [xmin, xmax],
                [0]*2,
                '-',
                color='grey',
                lw=1,
                zorder=-1,
            )
            axes[1].set_ylim(ylim)
        axes[-1].set_xlabel(axis_labels[param_name], fontsize=14)
        axes[ 0].set_xscale(axis_scales[param_name])
        axes[ 0].set_xlim(xmin, xmax)
        axes[ 0].set_ylabel(axis_labels['B'], fontsize=14)
        if do_reduced:
            axes[1].set_ylabel(axis_labels['Q'], fontsize=14)
        add_param_text(axes[0], param_name)
    elif len(dimensions) == 2:
        # 2D contour plot
        fig, axes = plt.subplots(1 + do_reduced, 1, sharex=True)
        axes = any2list(axes)
        def ensuretri(xdata, ydata, bpower, bpower_reduced):
            if len(xdata) > 2:
                return xdata, ydata, bpower, bpower_reduced
            xdata = asarray([
                xdata[0],
                np.mean(xdata)*(1 - 1e-3),
                np.mean(xdata)*(1 + 1e-3),
                xdata[1],
            ])
            ydata = asarray([
                ydata[0],
                np.mean(ydata)*(1 - 1e-3),
                np.mean(ydata)*(1 + 1e-3),
                ydata[1],
            ])
            bpower = asarray([
                bpower[0],
                np.mean(bpower),
                np.mean(bpower),
                bpower[1],
            ])
            if bpower_reduced is not None:
                bpower_reduced = asarray([
                    bpower_reduced[0],
                    np.mean(bpower_reduced),
                    np.mean(bpower_reduced),
                    bpower_reduced[1],
                ])
            return xdata, ydata, bpower, bpower_reduced
        xdata, ydata = bins_data[dimensions[0]], bins_data[dimensions[1]]
        if axis_scales[dimensions[0]] == 'log':
            xdata = np.log10(xdata)
        if axis_scales[dimensions[1]] == 'log':
            ydata = np.log10(ydata)
        xdata, ydata, bpower, bpower_reduced = ensuretri(
            xdata, ydata, bpower,
            (bpower_reduced if do_reduced else None),
        )
        if do_treelevel:
            xdata_treelevel, ydata_treelevel = (
                bins_data_treelevel[dimensions[0]], bins_data_treelevel[dimensions[1]],
            )
            (
                xdata_treelevel,
                ydata_treelevel,
                bpower_treelevel,
                bpower_reduced_treelevel,
            ) = ensuretri(
                xdata_treelevel, ydata_treelevel, bpower_treelevel,
                (bpower_reduced_treelevel if do_reduced else None),
            )
            if axis_scales[dimensions[0]] == 'log':
                xdata_treelevel = np.log10(xdata_treelevel)
            if axis_scales[dimensions[1]] == 'log':
                ydata_treelevel = np.log10(ydata_treelevel)
        logc = get_logc(np.abs(bpower))
        tri = matplotlib.tri.Triangulation(xdata, ydata)
        tri.set_mask((bpower[tri.triangles] < 0).any(axis=1))
        pc = axes[0].tripcolor(
            tri, np.abs(bpower),
            norm=(matplotlib.colors.LogNorm() if logc else None),
            shading='gouraud',
        )
        pcs = [pc]
        if (bpower < 0).sum() > 2:
            α = 0.4
            cmap = getattr(
                matplotlib.cm, matplotlib.rcParams['image.cmap'],
            )(linspace(0, 1, 256))
            cmap = (1 - α) + α*cmap
            axes[0].tripcolor(
                xdata, ydata, np.abs(bpower),
                norm=pc.norm,
                cmap=matplotlib.colors.ListedColormap(cmap),
                shading='gouraud',
                zorder=-1,
            )
        def create_legend_marker(ax):
            markerfacecolors = getattr(
                matplotlib.cm, matplotlib.rcParams['image.cmap'],
            )([0.2, 0.8])[:, :3]
            ylim = ax.get_ylim()
            ax.plot(
                np.mean(xdata),
                np.max((2*np.max(ydata) - np.min(ydata), np.max(ydata)**2/np.min(ydata))),
                's',
                markersize=13, markeredgecolor='none', fillstyle='top',
                markerfacecolor=markerfacecolors[1], markerfacecoloralt=markerfacecolors[0],
                label=label,
            )
            ax.set_ylim(ylim)
        create_legend_marker(axes[0])
        if do_treelevel:
            logc_treelevel = get_logc(bpower_treelevel)
            axes[0].tricontour(
                xdata_treelevel, ydata_treelevel, bpower_treelevel,
                norm=pc.norm,
                linewidths=1,
            )
            tcs = [
                axes[0].tricontour(
                    xdata_treelevel, ydata_treelevel, bpower_treelevel,
                    norm=pc.norm,
                    colors='k',
                    linestyles='dashed',
                    linewidths=1,
                )
            ]
            if logc_treelevel != logc:
                for α in range(2):
                    tc = axes[0].tricontour(
                        xdata_treelevel, ydata_treelevel, bpower_treelevel,
                        norm=(matplotlib.colors.LogNorm() if logc_treelevel else None),
                        colors='k',
                        linestyles='dashed',
                        linewidths=1,
                        alpha=α,
                    )
                    if α:
                        break
                    axes[0].tricontour(
                        xdata_treelevel, ydata_treelevel, bpower_treelevel,
                        norm=pc.norm,
                        linewidths=1,
                        levels=tc.levels,
                    )
                tcs.append(tc)
            def get_fmt_treelevel(tc):
                nfigs = 2
                vmin = np.min(tc.levels)
                if vmin > 0:
                    x = np.max(tc.levels)/vmin - 1
                    if x > 0:
                        nfigs = np.max((nfigs, int(2 - log10(x))))
                fmt_treelevel = lambda x: '${}$'.format(
                    significant_figures(
                        x, nfigs, fmt='TeX', incl_zeros=False,
                    )
                )
                return fmt_treelevel
            for tc in tcs:
                axes[0].clabel(tc, tc.levels, fontsize=10, fmt=get_fmt_treelevel(tc))
            def create_legend_marker_treelevel(ax):
                ax.plot(
                    np.mean(xdata_treelevel), np.mean(ydata_treelevel), 'k--',
                    lw=1, label=label_treelevel,
                )
            create_legend_marker_treelevel(axes[0])
        if do_reduced:
            pc = axes[1].tripcolor(xdata, ydata, bpower_reduced, shading='gouraud')
            pcs.append(pc)
            create_legend_marker(axes[1])
            if do_treelevel:
                axes[1].tricontour(
                    xdata_treelevel, ydata_treelevel, bpower_reduced_treelevel,
                    norm=pc.norm,
                    linewidths=1,
                )
                tc = axes[1].tricontour(
                    xdata_treelevel, ydata_treelevel, bpower_reduced_treelevel,
                    colors='k',
                    linestyles='dashed',
                    linewidths=1,
                )
                axes[1].clabel(tc, tc.levels, fontsize=10, fmt=get_fmt_treelevel(tc))
                create_legend_marker_treelevel(axes[1])
        axes[-1].set_xlabel(axis_labels[dimensions[0]], fontsize=14)
        ticks = get_logticks(xdata, ydata)
        for ax in axes:
            for i, c in enumerate('xy'):
                if axis_scales[dimensions[i]] == 'log':
                    getattr(ax, f'set_{c}ticks')(np.log10(ticks[c]))
                    getattr(ax, f'{c}axis').set_major_formatter(
                        matplotlib.ticker.FuncFormatter(log_tick_formatter)
                    )
                    getattr(ax, f'{c}axis').set_minor_locator(
                        matplotlib.ticker.AutoMinorLocator()
                    )
        add_param_text(axes[0], *dimensions)
        for ax in axes:
            ax.set_xlim(np.min(xdata), np.max(xdata))
            ax.set_ylim(np.min(ydata), np.max(ydata))
        for ax, pc, bq in zip(axes, pcs, ['B', 'Q']):
            ax.set_ylabel(axis_labels[dimensions[1]], fontsize=14)
            cbar = fig.colorbar(pc, ax=ax)
            cbar.ax.tick_params(labelsize=ticklabelsizefac*13)
            cbar.set_label(axis_labels[bq], fontsize=14)
            if logc and bq == 'B':
                cbar.ax.set_yscale('log')
    elif len(dimensions) == 3:
        # 3D scatter plot
        fig = plt.figure(figsize=(6.4, 4.8*(1 + 0.9*do_reduced)))
        axes = [fig.add_subplot(1 + do_reduced, 1, 1, projection='3d')]
        if do_reduced:
            axes.append(fig.add_subplot(1 + do_reduced, 1, 2, projection='3d'))
        ticklabelsizefac *= 0.8
        marker_size = 1.8e+3*bpower.shape[0]**(-2.0/3.0)
        marker_size = np.min((500, marker_size))
        marker_size = np.max(( 10, marker_size))
        logc = get_logc(np.abs(bpower))
        logk = np.log10(k)
        bpower_neg = (bpower < 0)
        fillstyle = ('top' if do_treelevel else 'full')
        if bpower_neg.any():
            axes[0].scatter(
                logk[bpower_neg], t[bpower_neg], μ[bpower_neg],
                s=marker_size,
                c=np.abs(bpower[bpower_neg]),
                marker=matplotlib.markers.MarkerStyle('X', fillstyle=fillstyle),
                norm=(matplotlib.colors.LogNorm() if logc else None),
            )
        bpower_pos = (bpower > 0)
        arts = [None]
        if bpower_pos.any():
            arts[0] = axes[0].scatter(
                logk[bpower_pos], t[bpower_pos], μ[bpower_pos],
                s=marker_size,
                c=bpower[bpower_pos],
                marker=matplotlib.markers.MarkerStyle('o', fillstyle=fillstyle),
                norm=(matplotlib.colors.LogNorm() if logc else None),
                label=label,
            )
        if do_treelevel:
            logk_treelevel = np.log10(k_treelevel)
            vmin, vmax = np.min(np.abs(bpower)), np.max(np.abs(bpower))
            bpower_treelevel_truncated = bpower_treelevel.copy()
            bpower_treelevel_truncated[bpower_treelevel_truncated < vmin] = vmin
            bpower_treelevel_truncated[bpower_treelevel_truncated > vmax] = vmax
            bpower_treelevel_truncated = np.concatenate((
                bpower_treelevel_truncated, [vmin, vmax],
            ))
            logk_treelevel_truncated = np.concatenate((logk_treelevel, [np.mean(logk_treelevel)]*2))
            t_treelevel_truncated    = np.concatenate((t_treelevel,    [np.mean(t_treelevel)]*2))
            μ_treelevel_truncated    = np.concatenate((μ_treelevel,    [np.mean(μ_treelevel)]*2))
            fillstyle_treelevel = 'bottom'
            linewidth_treelevel = 2*bpower.shape[0]**(-1.0/3.0)
            hatch_linewidth_ori = matplotlib.rcParams['hatch.linewidth']
            matplotlib.rcParams['hatch.linewidth'] = linewidth_treelevel
            axes[0].scatter(
                logk_treelevel_truncated, t_treelevel_truncated, μ_treelevel_truncated,
                s=([marker_size]*bpower_treelevel.shape[0] + [0, 0]),
                c=bpower_treelevel_truncated,
                marker=matplotlib.markers.MarkerStyle('o', fillstyle=fillstyle_treelevel),
                norm=(matplotlib.colors.LogNorm() if logc else None),
                hatch='/'*5,
                edgecolor='k',
                linewidth=linewidth_treelevel,
            )
            def create_legend_marker_treelevel(ax):
                ax.scatter(
                    logk_treelevel[0], t_treelevel[0], μ_treelevel[0],
                    s=marker_size,
                    facecolor='none',
                    marker=matplotlib.markers.MarkerStyle('o', fillstyle=fillstyle_treelevel),
                    hatch='/'*5,
                    edgecolor='k',
                    linewidth=linewidth_treelevel,
                    label=label_treelevel,
                )
            create_legend_marker_treelevel(axes[0])
        if do_reduced:
            arts.append(
                axes[1].scatter(
                    logk, t, μ,
                    s=marker_size,
                    c=bpower_reduced,
                    marker=matplotlib.markers.MarkerStyle('o', fillstyle=fillstyle),
                    label=label,
                )
            )
            if do_treelevel:
                vmin, vmax = np.min(bpower_reduced), np.max(bpower_reduced)
                bpower_reduced_treelevel_truncated = bpower_reduced_treelevel.copy()
                bpower_reduced_treelevel_truncated[bpower_reduced_treelevel_truncated < vmin] = vmin
                bpower_reduced_treelevel_truncated[bpower_reduced_treelevel_truncated > vmax] = vmax
                bpower_reduced_treelevel_truncated = np.concatenate((
                    bpower_reduced_treelevel_truncated, [vmin, vmax],
                ))
                axes[1].scatter(
                    logk_treelevel_truncated, t_treelevel_truncated, μ_treelevel_truncated,
                    s=([marker_size]*bpower_reduced_treelevel.shape[0] + [0, 0]),
                    c=bpower_reduced_treelevel_truncated,
                    marker=matplotlib.markers.MarkerStyle('o', fillstyle=fillstyle_treelevel),
                    norm=(matplotlib.colors.LogNorm() if logc else None),
                    hatch='/'*5,
                    edgecolor='k',
                    linewidth=linewidth_treelevel,
                )
                create_legend_marker_treelevel(axes[1])
        ticks = get_logticks(logk, t)
        for ax, art, bq in zip(axes, arts, ['B', 'Q']):
            ax.set_xlabel(axis_labels[dimensions[0]], fontsize=14)
            ax.set_ylabel(axis_labels[dimensions[1]], fontsize=14)
            ax.set_zlabel(axis_labels[dimensions[2]], fontsize=14)
            xlim = ax.get_xlim()
            ax.set_xticks(np.log10(ticks['x']))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(log_tick_formatter))
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.set_xlim(xlim)
            if art is not None:
                cbar = fig.colorbar(art, ax=ax, location='left')
            cbar.ax.tick_params(labelsize=ticklabelsizefac*13)
            cbar.set_label(axis_labels[bq], fontsize=14)
            if logc and bq == 'B':
                cbar.ax.set_yscale('log')
    # Finishing touches
    t_str = (
        rf'$t = {{}}\, \mathrm{{{{{unit_time}}}}}$'
        .format(significant_figures(universals.t, 4, fmt='TeX'))
    )
    a_str = ''
    if enable_Hubble:
        a_str = ', $a = {}$'.format(significant_figures(universals.a, 4, fmt='TeX'))
    components_str = (
        components_str
        .replace('{', r'$\{$')
        .replace('}', r'$\}$')
    )
    axes[0].set_title(f'{components_str}\n{t_str}{a_str}', fontsize=16)
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=ticklabelsizefac*13)
        ax.tick_params(axis='both', which='minor', labelsize=ticklabelsizefac*11)
    if do_treelevel:
        axes[-1].legend(fontsize=(14 - 2*(do_reduced or len(dimensions) == 3)))
    fig.tight_layout()
    if do_reduced and len(dimensions) != 3:
        fig.subplots_adjust(hspace=(0.1 if len(dimensions) == 3 else 0.15))
    fig.savefig(filename, dpi=150)
    if len(dimensions) == 3 and do_treelevel:
        matplotlib.rcParams['hatch.linewidth'] = hatch_linewidth_ori
    # Done with this plot.
    # Close the figure, leaving no trace in memory of the plot.
    plt.close(fig)
    masterprint('done')

# Function for plotting detrended CLASS perturbations
@cython.pheader(
    # Arguments
    k='Py_ssize_t',
    k_magnitude='double',
    transferfunction_info=object,  # TransferFunctionInfo
    class_species=str,
    factors='double[::1]',
    exponents='double[::1]',
    splines=object,  # np.ndarray of dtype object
    largest_trusted_k_magnitude='double',
    crossover='int',
    # Locals
    exponent='double',
    exponent_str=str,
    factor='double',
    factor_str=str,
    filename=str,
    i='Py_ssize_t',
    k_str=str,
    key=str,
    loga_value='double',
    loga_values='double[::1]',
    loga_values_spline='double[::1]',
    n='Py_ssize_t',
    perturbations_detrended_spline='double[::1]',
    skip='Py_ssize_t',
    spline='Spline',
    val=str,
)
def plot_detrended_perturbations(k, k_magnitude, transferfunction_info, class_species,
    factors, exponents, splines, largest_trusted_k_magnitude, crossover):
    # Fetch Matplotlib
    plt = get_matplotlib().pyplot
    # All processes could carry out this work, but as it involves I/O,
    # we only allow the master process to do so.
    if not master:
        abort(f'rank {rank} called plot_detrended_perturbations()')
    n_subplots = 0
    for spline in splines:
        if spline is None:
            break
        n_subplots += 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(6*n_subplots + 0.4, 4.8))
    axes = any2list(axes)
    k_str = significant_figures(k_magnitude, 3, fmt='TeX', force_scientific=True)
    fig.suptitle(
        ('' if transferfunction_info.total else rf'{class_species}, ')
        + rf'$k = {k_str}\, \mathrm{{{unit_length}}}^{{-1}}$',
        fontsize=16,
        horizontalalignment='center',
    )
    for n, ax in enumerate(axes):
        factor, exponent, spline = factors[n], exponents[n], splines[n]
        a_values, perturbations_detrended = spline.x, spline.y
        index_left = 0
        if n != 0:
            index_left += crossover
        index_right = a_values.shape[0]
        if n != n_subplots - 1:
            index_right -= crossover
        a_values = a_values[index_left:index_right]
        a_min = significant_figures(
            a_values[0], 4, fmt='TeX', force_scientific=True,
        )
        a_max = significant_figures(
            a_values[a_values.shape[0] - 1], 4, fmt='TeX', force_scientific=True,
        )
        perturbations_detrended = perturbations_detrended[index_left:index_right]
        # Plot the detrended CLASS data
        ax.semilogx(asarray(a_values), asarray(perturbations_detrended), '.', markersize=3)
        # Plot the spline at values midway between the data points
        loga_values = np.log(a_values)
        loga_values_spline             = empty(loga_values.shape[0] - 1, dtype=C2np['double'])
        perturbations_detrended_spline = empty(loga_values.shape[0] - 1, dtype=C2np['double'])
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
        ax.semilogx(
            np.exp(loga_values_spline), asarray(perturbations_detrended_spline), '-',
            linewidth=1, zorder=0,
        )
        ax.set_xlim(a_values[0], a_values[a_values.shape[0] - 1])
        # Decorate plot
        if n == 0:
            ax.set_ylabel(
                rf'$({transferfunction_info.name_latex} - \mathrm{{trend}})\, '
                rf'[{transferfunction_info.units_latex}]$'
                if transferfunction_info.units_latex else
                rf'${transferfunction_info.name_latex} - \mathrm{{trend}}$',
                fontsize=14,
            )
        ax.set_xlabel(rf'$a \in [{a_min}, {a_max}]$', fontsize=14)
        factor_str = significant_figures(factor, 6, fmt='TeX', force_scientific=True)
        exponent_str = significant_figures(exponent, 6, force_scientific=False)
        trend_str = (
            rf'$\mathrm{{trend}} = 0$'
            if factor == 0 else
            rf'$\mathrm{{trend}} = {factor_str}'
            rf'{transferfunction_info.units_latex}a^{{{exponent_str}}}$'
        )
        ax.text(0.5, 0.8,
            trend_str,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=14,
        )
        if k_magnitude > largest_trusted_k_magnitude:
            ax.text(0.5, 0.65,
                rf'(using data from $k = {largest_trusted_k_magnitude}\, '
                rf'\mathrm{{{unit_length}}}^{{-1}}$)',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=14,
            )
    # Finalise and save plot
    fig.subplots_adjust(wspace=0, hspace=0)
    filename = '/'.join([
        output_dirs['powerspec'],
        'class_perturbations',
        transferfunction_info.name_ascii.format(class_species),
    ])
    os.makedirs(filename, exist_ok=True)
    filename += f'/{k}.png'
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()

# Function for plotting processed CLASS perturbations
@cython.pheader(
    # Arguments
    a_values='double[::1]',
    k_magnitudes='double[::1]',
    transfer='double[:, ::1]',
    transferfunction_info=object,  # TransferFunctionInfo
    class_species=str,
    n_plots_in_figure='Py_ssize_t',
    # Locals
    a='double',
    dirname=str,
    i='Py_ssize_t',
    i_figure='Py_ssize_t',
    key=str,
    nfigs='Py_ssize_t',
    val=str,
)
def plot_processed_perturbations(
    a_values, k_magnitudes, transfer, transferfunction_info, class_species,
    n_plots_in_figure=10,
):
    """The 2D transfer array is the tabulated transfer function values,
    indexed as transfer[a, k], with the values of a and k given by
    a_values and k_magnitudes.
    """
    # Fetch Matplotlib
    plt = get_matplotlib().pyplot
    # All processes could carry out this work, but as it involved I/O,
    # we only allow the master process to do so.
    if not master:
        abort(f'rank {rank} called plot_processed_perturbations()')
    if transferfunction_info.total:
        masterprint(f'Plotting processed {transferfunction_info.name} transfer functions ...')
    else:
        masterprint(
            f'Plotting processed {transferfunction_info.name} {class_species} '
            f'transfer functions ...'
        )
    dirname = '/'.join([
        output_dirs['powerspec'],
        'class_perturbations_processed',
        transferfunction_info.name_ascii.format(class_species),
    ])
    os.makedirs(dirname, exist_ok=True)
    nfigs = int(log10(a_values.shape[0])) + 1
    i_figure = 0
    fig, ax = plt.subplots()
    for i in range(a_values.shape[0]):
        a = a_values[i]
        ax.semilogx(
            asarray(k_magnitudes), asarray(transfer[i, :]),
            label='$a={}$'.format(significant_figures(a, nfigs, fmt='TeX')),
        )
        if ((i + 1)%n_plots_in_figure == 0) or i == ℤ[a_values.shape[0] - 1]:
            ax.legend()
            ax.set_xlabel(rf'$k\,[\mathrm{{{unit_length}}}^{{-1}}]$', fontsize=14)
            ax.set_ylabel(
                rf'${transferfunction_info.name_latex}\, [{transferfunction_info.units_latex}]$'
                if transferfunction_info.units_latex else
                rf'${transferfunction_info.name_latex}$',
                fontsize=14,
            )
            if not transferfunction_info.total:
                ax.set_title(
                    class_species,
                    fontsize=16,
                    horizontalalignment='center',
                )
            ax.tick_params(axis='x', which='major', labelsize=13)
            fig.tight_layout()
            fig.savefig(f'{dirname}/{i_figure}.png', dpi=150)
            i_figure += 1
            ax.cla()
    plt.close(fig)
    masterprint('done')

# Top-level function for computing, rendering and saving 2D renders
@cython.pheader(
    # Arguments
    components=list,
    filename=str,
    # Locals
    component='Component',
    components_str=str,
    declaration=object,  # Render2DDeclaration
    declarations=list,
    n_dumps='int',
    returns='void',
)
def render2D(components, filename):
    # Get render2D declarations
    declarations = get_render2D_declarations(components)
    # Count up number of 2D renders to be dumped to disk
    n_dumps = 0
    for declaration in declarations:
        if declaration.do_data or declaration.do_image:
            n_dumps += 1
    # Compute 2D render for each declaration
    for declaration in declarations:
        components_str = ', '.join([component.name for component in declaration.components])
        if len(declaration.components) > 1:
            components_str = f'{{{components_str}}}'
        masterprint(f'Rendering 2D projection of {components_str} ...')
        # Compute the 2D render. In the case of both normal
        # and terminal 2D renders, both of these will be computed.
        # The results are stored in declaration.projections.
        # Only the master process holds the full 2D renders.
        compute_render2D(declaration)
        # Save 2D render data to an HDF5 file on disk, if specified
        save_render2D_data(declaration, filename, n_dumps)
        # Enhance the normal and terminal 2D render, if specified
        enhance_render2D(declaration)
        # Rescale the 2D render values so that they lie in [0, 1]
        rescale_render2D(declaration)
        # Save 2D render image to a PNG file on disk, if specified
        save_render2D_image(declaration, filename, n_dumps)
        # Display terminal render, if specified
        display_terminal_render(declaration)
        # Done with the entire 2D rendering process for this declaration
        masterprint('done')

# Function for getting generic output declarations
@cython.header(
    # Arguments
    output_type=str,
    components=list,
    selections=dict,
    options=dict,
    Declaration=object,  # collections.namedtuple
    # Locals
    cache_key=tuple,
    component_combination=list,
    component_combinations=list,
    declaration=object,
    declarations=list,
    do=dict,
    gridsize='Py_ssize_t',
    key=str,
    selected=dict,
    specifications=dict,
    returns=list,
)
def get_output_declarations(output_type, components, selections, options, Declaration):
    # Look up declarations in cache.
    # We would like to include the selections dict in the cache key.
    # As this is not hashable, we use its str representation instead.
    # This is not reliable if the dict is mutated between calls to this
    # function, but the penalty is just that the result is not found in
    # cache, the correct result will be returned regardless.
    cache_key = (output_type, tuple(components), str(selections))
    declarations = output_declarations_cache.get(cache_key)
    if declarations:
        return declarations
    # Generate list of lists storing all possible (unordered)
    # combinations of the passed components.
    component_combinations = list(map(
        list,
        itertools.chain.from_iterable(
            [itertools.combinations(components, i) for i in range(1, len(components) + 1)]
        ),
    ))
    # Construct dicts to be used with the is_selected() function
    selected = {
        key: {selections_key: val[key] for selections_key, val in selections.items()}
        for key in selections['default'].keys()
    }
    # Construct declarations
    declarations = []
    for component_combination in component_combinations:
        # Check whether any output is specified
        # for this component combination.
        do = {
            key: is_selected(component_combination, selection)
            for key, selection in selected.items()
        }
        if not any(do.values()):
            continue
        # Output is to be generated for this component combination
        specifications = {}
        # If this declaration type makes use of a 'global gridsize' and
        # none is set, use the maximum of the individual upstream grid
        # sizes of the components instead.
        if 'global gridsize' in options:
            gridsize = is_selected(component_combination, options['global gridsize'], default=-1)
            if gridsize == -1:
                gridsize = np.max([
                    getattr(component, f'{output_type}_upstream_gridsize')
                    for component in component_combination
                ])
            specifications['gridsize'] = gridsize
        # Look up the rest of the specifications
        specifications |= {
            key.replace(' ', '_'): is_selected(component_combination, option)
            for key, option in options.items()
            if key not in {'upstream gridsize', 'global gridsize'}
        }
        # Instantiate declaration
        declaration = Declaration(
            components=component_combination,
            **{f'do_{key}'.replace(' ', '_'): val for key, val in do.items()},
            **specifications,
        )
        declarations.append(declaration)
    # Store declarations in cache and return
    output_declarations_cache[cache_key] = declarations
    return declarations
# Cache used by the get_output_declarations() function
cython.declare(output_declarations_cache=dict)
output_declarations_cache = {}

# Function for getting declarations for all needed 2D renders,
# given a list of components.
@cython.header(
    # Arguments
    components=list,
    # Locals
    cache_key=tuple,
    chunk=object,  # np.ndarray
    declaration=object,  # Render2DDeclaration
    declarations=list,
    gridsize='Py_ssize_t',
    index='Py_ssize_t',
    iteration=str,
    key=str,
    projection='double[:, ::1]',
    projections=dict,
    size='Py_ssize_t',
    terminal_resolution='Py_ssize_t',
    returns=list,
)
def get_render2D_declarations(components):
    """Note that due to the global reallocation (chunk.resize()),
    this function uses no cache. The projection field of all
    declarations must be replaced at every call.
    """
    # Get declarations with basic fields populated
    declarations = get_output_declarations(
        'render2D',
        components,
        render2D_select,
        render2D_options,
        Render2DDeclaration,
    )
    # Add missing declaration fields.
    # We need to do the reallocation of chunk for all grid sizes of the
    # declarations before we start wrapping it, as these reallocations
    # may move the memory. To this end we perform the below loop twice,
    # with the first iteration taking care of reallocation only.
    for iteration in ('reallocate', 'wrap'):
        for index, declaration in enumerate(declarations):
            # Set terminal resolution if unset
            terminal_resolution = declaration.terminal_resolution
            if terminal_resolution == -1:
                # Set the terminal resolution equal to the gridsize,
                # though no larger than the terminal width.
                terminal_resolution = np.min((declaration.gridsize, terminal_width))
                # As the terminal render is obtained through FFTs,
                # the terminal resolution must be divisible by
                # the number of processes and be even.
                terminal_resolution = terminal_resolution//nprocs*nprocs
                if terminal_resolution == 0:
                    terminal_resolution = nprocs
                if terminal_resolution%2:
                    terminal_resolution *= 2
            # Create needed 2D projection arrays.
            # Here we always make use of the same globally allocated
            # memory, which we reallocate if necessary. We can then
            # never have two 2D renders simultaneously in memory.
            projections = {}
            for key, chunk in projection_chunks.items():
                if not getattr(declaration, f'do_{key}'):
                    continue
                gridsize = declaration.gridsize
                if key == 'terminalimage':
                    gridsize = terminal_resolution
                size = gridsize**2
                with unswitch(2):
                    if iteration == 'reallocate':
                        if chunk.size < size:
                            chunk.resize(size, refcheck=False)
                        continue
                    else:  # iteration == 'wrap'
                        projection = chunk[:size].reshape([gridsize]*2)
                        projections[key] = projection
            # Replace old declaration with a new, fully populated one
            with unswitch(1):
                if iteration == 'wrap':
                    declaration = declaration._replace(
                        terminal_resolution=terminal_resolution,
                        projections=projections,
                    )
                    declarations[index] = declaration
    # Return declarations without caching
    return declarations
# Global memory chunks for storing projections (2D render data).
# The 'image' and 'data' projection are not distinct.
cython.declare(projection_chunks=dict)
projection_chunks = {
    'image'        : empty(1, dtype=C2np['double']),
    'terminalimage': empty(1, dtype=C2np['double']),
}
projection_chunks['data'] = projection_chunks['image']
# Create the Render2DDeclaration type
fields = (
    'components', 'do_data', 'do_image', 'do_terminalimage', 'gridsize',
    'terminal_resolution', 'interpolation', 'deconvolve', 'interlace',
    'axis', 'extent', 'colormap', 'enhance',
    'projections',
)
Render2DDeclaration = collections.namedtuple(
    'Render2DDeclaration', fields, defaults=[None]*len(fields),
)

# Function which given a 2D render declaration correctly populated
# with all fields will compute its render and terminal render.
@cython.header(
    # Arguments
    declaration=object,  # Render2DDeclaration
    # Locals
    axis=str,
    component='Component',
    components=list,
    deconvolve='bint',
    extent=tuple,
    grid='double[:, :, ::1]',
    grid_terminal='double[:, :, ::1]',
    gridsize='Py_ssize_t',
    gridsizes_upstream=list,
    i='Py_ssize_t',
    interlace=str,
    interpolation='int',
    j='Py_ssize_t',
    key=str,
    projection='double[:, ::1]',
    projections=dict,
    row='Py_ssize_t',
    slab='double[:, :, ::1]',
    termsize='Py_ssize_t',
    returns='void',
)
def compute_render2D(declaration):
    # Extract some variables from the 2D render declaration
    components    = declaration.components
    gridsize      = declaration.gridsize
    termsize      = declaration.terminal_resolution
    interpolation = declaration.interpolation
    deconvolve    = declaration.deconvolve
    interlace     = declaration.interlace
    axis          = declaration.axis
    extent        = declaration.extent
    projections   = declaration.projections
    # Interpolate the components onto global Fourier slabs by first
    # interpolating onto individual upstream grids, Fourier transforming
    # these and adding them together.
    # We choose to interpolate the physical density ρ.
    gridsizes_upstream = [
        component.render2D_upstream_gridsize
        for component in components
    ]
    slab = interpolate_upstream(
        components, gridsizes_upstream, gridsize, 'ρ', interpolation,
        deconvolve=deconvolve, interlace=interlace, output_space='Fourier',
    )
    # If a terminal image is to be produced, construct a copy of the
    # slab, resized appropriately. Obtain the result in real space.
    if 'terminalimage' in projections:
        grid_terminal = resize_grid(
            slab, termsize,
            input_space='Fourier', output_space='real',
            output_grid_or_buffer_name='grid_terminal',
            output_slab_or_buffer_name='slab_terminal',
            inplace=False, do_ghost_communication=False,
        )
    # Transform the slab to real space
    fft(slab, 'backward')
    grid = domain_decompose(slab, 'grid_global', do_ghost_communication=False)
    # Get projected 2D grid for main 2D render data/image
    for key, projection in projections.items():
        if key in {'data', 'image'}:
            project_render2D(grid, projection, axis, extent)
            break
    # Get projected 2D grid for terminal render
    projection = projections.get('terminalimage')
    if projection is not None:
        project_render2D(grid_terminal, projection, axis, extent)
        # Since each monospaced character cell in the terminal is
        # rectangular with about double the height compared to the
        # width, the terminal projection should only have half as many
        # rows as it has columns. Below we average together consecutive
        # pairs of rows. Though the terminal projection still has shape
        # (termsize, termsize), you should then only make use of the
        # first termsize//2 rows after this.
        if termsize%2 != 0:
            abort(f'Cannot produce terminal render with odd resolution {termsize}')
        row = -1
        for i in range(0, termsize, 2):
            row += 1
            for j in range(termsize):
                projection[row, j] = 0.5*(projection[i, j] + projection[i + 1, j])

# Function for converting a distributed 3D domain grid
# into a 2D projection grid.
@cython.header(
    # Arguments
    grid='double[:, :, ::1]',
    projection='double[:, ::1]',
    axis=str,
    extent=tuple,
    # Locals
    cellsize='double',
    dim='int',
    dim_axis='int',
    dims='int[::1]',
    domain_bgn_indices='Py_ssize_t[::1]',
    float_index_global_bgn='double',
    float_index_global_end='double',
    frac='double',
    frac_bgn='double',
    frac_end='double',
    gridshape_local='Py_ssize_t[::1]',
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    i2='Py_ssize_t',
    indices_2D_bgn='Py_ssize_t[::1]',
    indices_2D_end='Py_ssize_t[::1]',
    indices_global_bgn='Py_ssize_t[::1]',
    indices_global_end='Py_ssize_t[::1]',
    indices_local='Py_ssize_t[::1]',
    indices_local_bgn='Py_ssize_t[::1]',
    indices_local_end='Py_ssize_t[::1]',
    j='Py_ssize_t',
    participate='bint',
    projection_arr=object,  # np.ndarray
    slices=list,
    returns='double[:, ::1]',
)
def project_render2D(grid, projection, axis, extent):
    """The passed 2D projection array will be mutated in-place.
    Only the projection on the master process will be complete.
    """
    # Get global grid size of the grid off of the 2D projection array,
    # which is fully allocated on every process.
    gridsize = projection.shape[0]
    # Get global index range into the grids, specifying the chunk
    # that should be used for the projection.
    indices_global_bgn = asarray([0       ]*3, dtype=C2np['Py_ssize_t'])
    indices_global_end = asarray([gridsize]*3, dtype=C2np['Py_ssize_t'])
    cellsize = boxsize/gridsize
    dim_axis = 'xyz'.index(axis)
    float_index_global_bgn = extent[0]/cellsize
    float_index_global_end = extent[1]/cellsize
    if isint(float_index_global_bgn):
        float_index_global_bgn = round(float_index_global_bgn)
    if isint(float_index_global_end):
        float_index_global_end = round(float_index_global_end)
    indices_global_bgn[dim_axis] = int(float_index_global_bgn)
    indices_global_end[dim_axis] = int(ceil(float_index_global_end))
    # If the extent is chosen such that it divides the grid cells,
    # only the corresponding fraction of cells (of the first and last
    # planes along the axis) will enter the projection. These fractions
    # are computed here.
    frac_bgn = 1 - (float_index_global_bgn - indices_global_bgn[dim_axis])
    frac_end = 1 - (indices_global_end[dim_axis] - float_index_global_end)
    # Convert the global indices to local indices,
    # disregarding ghost points, for now.
    domain_bgn_indices = asarray(
        [
            int(round(domain_bgn_x/cellsize)),
            int(round(domain_bgn_y/cellsize)),
            int(round(domain_bgn_z/cellsize)),
        ],
        dtype=C2np['Py_ssize_t'],
    )
    gridshape_local = asarray(
        asarray(asarray(grid).shape) - ℤ[2*nghosts],
        dtype=C2np['Py_ssize_t'],
    )
    participate = True
    if participate:
        indices_local_bgn = asarray(indices_global_bgn) - asarray(domain_bgn_indices)
        for dim in range(3):
            if indices_local_bgn[dim] < 0:
                indices_local_bgn[dim] = 0
                if dim == dim_axis:
                    frac_bgn = 0
            elif indices_local_bgn[dim] > gridshape_local[dim]:
                participate = False
                break
    if participate:
        indices_local_end = asarray(indices_global_end) - asarray(domain_bgn_indices)
        for dim in range(3):
            if indices_local_end[dim] < 0:
                participate = False
                break
            elif indices_local_end[dim] > gridshape_local[dim]:
                indices_local_end[dim] = gridshape_local[dim]
                if dim == dim_axis:
                    frac_end = 0
    if participate:
        for dim in range(3):
            if indices_local_bgn[dim] == indices_local_end[dim]:
                participate = False
                break
    # Fill in the local part of the projection on each process
    projection[...] = 0
    projection_arr = asarray(projection)
    if participate:
        # Redefine the global indices so that they correspond to the
        # local chunk, but indexing into a global grid.
        indices_global_bgn = asarray(indices_local_bgn) + asarray(domain_bgn_indices)
        indices_global_end = asarray(indices_local_end) + asarray(domain_bgn_indices)
        # Get indices into the projection
        dims = asarray(
            {
                'x': (1, 2),  # The projection will be onto the yz plane with y right and z up
                'y': (0, 2),  # The projection will be onto the xz plane with x right and z up
                'z': (0, 1),  # The projection will be onto the xy plane with x right and y up
            }[axis],
            dtype=C2np['int'],
        )
        indices_2D_bgn = asarray(
            [indices_global_bgn[dims[0]], indices_global_bgn[dims[1]]],
            dtype=C2np['Py_ssize_t'],
        )
        indices_2D_end = asarray(
            [indices_global_end[dims[0]], indices_global_end[dims[1]]],
            dtype=C2np['Py_ssize_t'],
        )
        # Construct slices indexing into the grid,
        # except the first and last plane along the specified axis.
        slices = [slice(nghosts + bgn, nghosts + end) for bgn, end in zip(indices_local_bgn, indices_local_end)]
        slices[dim_axis] = slice(
            nghosts + indices_local_bgn[dim_axis] + (frac_bgn > 0),
            nghosts + indices_local_end[dim_axis] - (frac_end > 0),
        )
        # Sum the contributions from the grid along the axis
        projection_arr[
            indices_2D_bgn[0]:indices_2D_end[0],
            indices_2D_bgn[1]:indices_2D_end[1],
        ] += np.sum(asarray(grid)[tuple(slices)], dim_axis)
        # If the extent is over a single plane of cells, the above sum
        # is empty. Furthermore we need to not double count this single
        # plane, i.e. reduce frac_bgn and frac_end to a single
        # non-zero fraction.
        frac = float_index_global_end - float_index_global_bgn
        if (
                0 < frac_bgn
            and 0 < frac_end
            and 0 < frac <= 1
            and slices[dim_axis].start >= slices[dim_axis].stop
        ):
            frac_bgn = frac
            frac_end = 0
        # Add the missing contributions from the first and last plane.
        # Only a fraction (0 to 1) of these are used, corresponding to
        # only accounting for a fraction of a cell.
        indices_local_end[dim_axis] -= 1
        for frac, indices_local in zip(
            (frac_bgn, frac_end),
            (indices_local_bgn, indices_local_end),
        ):
            if frac > 0:
                slices[dim_axis] = nghosts + indices_local[dim_axis]
                projection_arr[
                    indices_2D_bgn[0]:indices_2D_end[0],
                    indices_2D_bgn[1]:indices_2D_end[1],
                ] += frac*asarray(grid)[tuple(slices)]
    # Sum up contributions from all processes into the master process,
    # after which this process holds the full projection.
    Reduce(
        sendbuf=(MPI.IN_PLACE if master else projection),
        recvbuf=(projection   if master else None),
        op=MPI.SUM,
    )
    if not master:
        return projection
    # The values in the projection correspond to physical densities.
    # Convert to mass.
    projection_arr *= (universals.a*cellsize)**3
    # Transpose the projection such that the first dimension (rows)
    # correspond to the upward/downward direction and the second
    # dimension (columns) correspond to the left/right direction.
    # Also flip the upward/downward axis by flipping the rows.
    # Together, this puts the projection into the proper state
    # for saving it as an image.
    # Transpose.
    for i in range(gridsize):
        for j in range(i):
            projection[i, j], projection[j, i] = projection[j, i], projection[i, j]
    # Vertical flip
    for i in range(gridsize//2):
        i2 = ℤ[gridsize - 1] - i
        for j in range(gridsize):
            projection[i, j], projection[i2, j] = projection[i2, j], projection[i, j]
    return projection

# Function for enhancing the contrast of the 2D renders
@cython.header(
    # Arguments
    declaration=object,  # Render2DDeclaration
    # Locals
    bin_edges='double[::1]',
    bins='Py_ssize_t[::1]',
    color_truncation_factor_lower='double',
    color_truncation_factor_upper='double',
    shifting_factor='double',
    exponent='double',
    exponent_lower='double',
    exponent_max='double',
    exponent_min='double',
    exponent_tol='double',
    exponent_upper='double',
    index='Py_ssize_t',
    index_center='Py_ssize_t',
    index_max='Py_ssize_t',
    index_min='Py_ssize_t',
    key=str,
    n_bins='Py_ssize_t',
    n_bins_fac='double',
    n_bins_min='Py_ssize_t',
    occupation='Py_ssize_t',
    projection='double[:, ::1]',
    projection_ptr='double*',
    size='Py_ssize_t',
    value='double',
    vmax='double',
    vmin='double',
    Σbins='Py_ssize_t',
    returns='void',
)
def enhance_render2D(declaration):
    """This function enhances the 2D renders by applying a non-linear
    transformation of the form
    projection → projection**exponent,
    plus enforced saturation of extreme values.
    The transformation happens in-place.

    The value for the exponent is chosen such that it leads to a nice
    distribution of the values in the projections. We take this to be
    the case when the histogram of these values is "centred" at the
    value specified by the shifting_factor parameter. A shifting_factor
    of 0.5 implies that the histogram of the pixel values is "centred"
    in the middle of the axis, with the same distance to the first and
    last bin. For Gaussian data, this requires a value of the exponent
    tending to 0. Thus, the shifting factor should be below 0.5.
    A shifting_factor between 0 and 0.5 shifts the centre of the
    histogram to be at the location of shifting_factor, measured
    relative to the histogram axis. Here, the centre is defined to be
    the point which partitions the histogram into two parts which
    integrate to the same value.

    This function contains several hard-coded numerical parameters,
    the values of which have been obtained through a process of trial
    and error and are judged purely on the artistic merit of the
    resulting images.
    """
    if not master:
        return
    if not declaration.enhance:
        return
    # Numerical parameters
    shifting_factor = 0.28
    exponent_min = 1e-2
    exponent_max = 1e+2
    exponent_tol = 1e-3
    n_bins_min = 25
    n_bins_fac = 1e-2
    color_truncation_factor_lower = 0.005
    color_truncation_factor_upper = 0.0001
    # Enforce all pixel values to be between 0 and 1
    rescale_render2D(declaration)
    # Perform independent enhancements
    # of the 'image' and 'terminalimage'.
    for key, projection in declaration.projections.items():
        if key == 'data':
            continue
        # The terminal image projection only contains data
        # in the upper half of the rows.
        if key == 'terminalimage':
            projection = projection[:projection.shape[0]//2, :]
        # Completely homogeneous projections cannot be enhanced
        vmin = np.min(projection)
        vmax = np.max(projection)
        if vmin == vmax:
            continue
        # Find a good value for the exponent using a binary search
        size = projection.size
        n_bins = int(n_bins_fac*size)
        n_bins = pairmax(n_bins, n_bins_min)
        exponent_lower = exponent_min
        exponent_upper = exponent_max
        exponent = 1
        index_min = -4
        index_max = -2
        while True:
            # Construct histogram over projection**exponent
            bins, bin_edges = np.histogram(asarray(projection)**exponent, n_bins)
            # Compute the sum of all bins. This is equal to the sum of
            # values in the projection. However, we skip bins[0] since
            # sometimes empty cells results in a large spike there.
            Σbins = size - bins[0]
            # Find the position of the centre of the histogram,
            # defined by the sums of bins being the same on both
            # sides of this centre. We again skip bins[0].
            occupation = 0
            for index in range(1, n_bins):
                occupation += bins[index]
                if occupation >= ℤ[Σbins//2]:
                    index_center = index
                    break
            else:
                # Something went wrong. Bail out.
                masterwarn('Something went wrong during 2D render enhancement')
                exponent = 1
                break
            if index_center < ℤ[n_bins*shifting_factor]:
                # The exponent should be decreased
                exponent_upper = exponent
                index_min = index_center
            elif index_center > ℤ[n_bins*shifting_factor]:
                # The exponent should be increased
                exponent_lower = exponent
                index_max = index_center
            else:
                # Good choice of exponent found
                break
            # The current value of the exponent does not place the
            # "centre" of the histogram at the desired location
            # specified by shifting_factor.
            # Check if the binary search has (almost) converged on
            # some other value.
            if index_max >= index_min and index_max - index_min <= 1:
                break
            # Check if the exponent is close
            # to one of the extreme values.
            if exponent/exponent_min < ℝ[1 + exponent_tol]:
                exponent = exponent_min
                break
            elif exponent_max/exponent < ℝ[1 + exponent_tol]:
                exponent = exponent_max
                break
            # Update the exponent. As the range of the exponent is
            # large, the binary step is done in logarithmic space.
            exponent = sqrt(exponent_lower*exponent_upper)
        # Apply the enhancement
        projection_ptr = cython.address(projection[:, :])
        for index in range(size):
            projection_ptr[index] **= exponent
        bins, bin_edges = np.histogram(projection, n_bins)
        Σbins = size - bins[0]
        # To further enhance the projected image, we set the colour
        # limits so as to truncate the colour space at both ends,
        # saturating pixels with very little or very high intensity.
        # The colour limits vmin and vmax are determined based on the
        # color_truncation_factor_* parameters. These specify the
        # accumulated fraction of Σbins at which the histogram should be
        # truncated, for the lower and upper intensity ends.
        # For projections with a lot of structure, the best results are
        # obtained by giving the lower colour truncation quite a large
        # value (this effectively removes the background), while giving
        # the higher colour truncation a small value,
        # so that small very overdense regions appear clearly.
        occupation = 0
        for index in range(1, n_bins):
            occupation += bins[index]
            if occupation >= ℤ[color_truncation_factor_lower*Σbins]:
                vmin = bin_edges[index - 1]
                break
        occupation = 0
        for index in range(n_bins - 1, 0, -1):
            occupation += bins[index]
            if occupation >= ℤ[color_truncation_factor_upper*Σbins]:
                vmax = bin_edges[index + 1]
                break
        # Apply colour limits
        for index in range(size):
            value = projection_ptr[index]
            value = pairmax(value, vmin)
            value = pairmin(value, vmax)
            projection_ptr[index] = value

# Function for rescaling the values in the projections
# so that they lie in [0, 1].
@cython.header(
    # Arguments
    declaration=object,  # Render2DDeclaration
    # Locals
    index='Py_ssize_t',
    key=str,
    projection='double[:, ::1]',
    projection_ptr='double*',
    vmin='double',
    vmax='double',
    returns='void',
)
def rescale_render2D(declaration):
    if not master:
        return
    for key, projection in declaration.projections.items():
        if key == 'data':
            continue
        # The terminal image projection only contains data
        # in the upper half of the rows.
        if key == 'terminalimage':
            projection = projection[:projection.shape[0]//2, :]
        projection_ptr = cython.address(projection[:, :])
        # Rescale values
        vmin = np.min(projection)
        vmax = np.max(projection)
        if vmin != 0 and vmax != 0 and isclose(vmin, vmax):
            # The projection is completely homogeneous and non-empty.
            # Set all values to ½.
            for index in range(projection.size):
                projection_ptr[index] = 0.5
        else:
            # The projection contains a proper distribution of values
            for index in range(projection.size):
                projection_ptr[index] = (projection_ptr[index] - vmin)*ℝ[1/(vmax - vmin)]

# Function for saving an already computed 2D render as an HDF5 file
@cython.header(
    # Arguments
    declaration=object,  # Render2DDeclaration
    filename=str,
    n_dumps='int',
    # Locals
    axis=str,
    component='Component',
    components=list,
    components_str=str,
    ext=str,
    extent=tuple,
    projection='double[:, ::1]',
    returns='void',
)
def save_render2D_data(declaration, filename, n_dumps):
    if not master:
        return
    if not declaration.do_data:
        return
    # Extract some variables from the 2D render declaration
    components = declaration.components
    axis       = declaration.axis
    extent     = declaration.extent
    projection = declaration.projections['data']
    # Set filename extension to hdf5
    for ext in ('hdf5', 'png'):
        filename = filename.removesuffix(f'.{ext}')
    filename += '.hdf5'
    # The filename should reflect the components
    # if multiple renders are to be dumped.
    if n_dumps > 1:
        filename = augment_filename(
            filename,
            '_'.join([component.name.replace(' ', '-') for component in components]),
            '.hdf5',
        )
    masterprint(f'Saving data to "{filename}" ...')
    components_str = ', '.join([component.name for component in components])
    if len(components) > 1:
        components_str = f'{{{components_str}}}'
    with open_hdf5(filename, mode='w') as hdf5_file:
        # Save used base unit
        hdf5_file.attrs['unit time'  ] = unit_time
        hdf5_file.attrs['unit length'] = unit_length
        hdf5_file.attrs['unit mass'  ] = unit_mass
        # Save attributes
        hdf5_file.attrs['boxsize'   ] = boxsize
        hdf5_file.attrs['components'] = components_str
        hdf5_file.attrs['axis'      ] = axis
        hdf5_file.attrs['extent'    ] = extent
        if enable_Hubble:
            hdf5_file.attrs['a'] = universals.a
        hdf5_file.attrs['t'    ] = universals.t
        # Store the 2D projection
        dset = hdf5_file.create_dataset('data', asarray(projection).shape, dtype=C2np['double'])
        dset[...] = projection
    masterprint('done')

# Function for saving an already computed 2D render as a PNG file
@cython.header(
    # Arguments
    declaration=object,  # Render2DDeclaration
    filename=str,
    n_dumps='int',
    # Locals
    component='Component',
    components=list,
    components_str=str,
    colormap=str,
    ext=str,
    projection='double[:, ::1]',
    returns='void',
)
def save_render2D_image(declaration, filename, n_dumps):
    if not declaration.do_image or not master:
        return
    # Fetch Matplotlib
    plt = get_matplotlib().pyplot
    # Extract some variables from the 2D render declaration
    components = declaration.components
    colormap   = declaration.colormap
    projection = declaration.projections['image']
    # Set filename extension to png
    for ext in ('hdf5', 'png'):
        filename = filename.removesuffix(f'.{ext}')
    filename += '.png'
    # The filename should reflect the components
    # if multiple renders are to be dumped.
    if n_dumps > 1:
        filename = augment_filename(
            filename,
            '_'.join([component.name.replace(' ', '-') for component in components]),
            '.png',
        )
    # Save colourised image to disk
    masterprint(f'Saving image to "{filename}" ...')
    plt.imsave(filename, asarray(projection), cmap=colormap, vmin=0, vmax=1)
    masterprint('done')

# Function for augmenting a filename with a given text
def augment_filename(filename, text, ext=''):
    """Example of use:
    augment_filename('/path/to/powerspec_a=1.0.png', 'matter', 'png')
      → '/path/to/powerspec_matter_a=1.0.png'
    """
    text = text.lstrip('_')
    ext = '.' + ext.lstrip('.')
    dirname, basename = os.path.split(filename)
    basename, baseext = os.path.splitext(basename)
    if baseext != ext:
        basename += baseext
    time_param_indices = collections.defaultdict(int)
    for time_param in ('t', 'a'):
        try:
            time_param_indices[time_param] = basename.index(f'_{time_param}=')
        except ValueError:
            continue
    if time_param_indices['t'] == time_param_indices['a']:
        basename += f'_{text}'
    else:
        time_param = sorted(time_param_indices.items(), key=lambda tup: tup[::-1])[-1][0]
        basename = (f'_{text}_{time_param}='
            .join(basename.rsplit(f'_{time_param}=', 1))
        )
    if ext != '.':
        basename += ext
    return os.path.join(dirname, basename)

# Function for displaying colourised 2D render directly in the terminal
@cython.header(
    # Arguments
    declaration=object,  # Render2DDeclaration
    # Locals
    colormap=str,
    colornumber='int',
    esc_space=str,
    i='Py_ssize_t',
    j='Py_ssize_t',
    projection='double[:, ::1]',
    terminal_ansi=list,
    returns='void',
)
def display_terminal_render(declaration):
    if not master:
        return
    if not declaration.do_terminalimage:
        return
    # Extract some variables from the 2D render declaration
    colormap = declaration.colormap
    projection = declaration.projections['terminalimage']
    # The terminal image projection only contains data
    # in the upper half of the rows.
    projection = projection[:projection.shape[0]//2, :]
    # Apply the terminal colormap
    set_terminal_colormap(colormap)
    # Construct list of strings, each string being a space prepended
    # with an ANSI/VT100 control sequences which sets the background
    # colour. When printed together, these strings produce an ANSI image
    # of the terminal projection.
    # We need to map the values between 0 and 1 to the 238 higher
    # integer colour numbers 18–255 (the lowest 18 colour numbers are
    # already occupied).
    esc_space = f'{esc_background} '
    terminal_ansi = []
    for     i in range(ℤ[projection.shape[0]]):
        for j in range(ℤ[projection.shape[1]]):
            colornumber = 18 + cast(round(projection[i, j]*237), 'int')
            # Insert a space with coloured background
            terminal_ansi.append(esc_space.format(colornumber))
        # Insert newline with no background colour
        terminal_ansi.append(f'{esc_normal}\n')
    # Print the ANSI image to the terminal
    masterprint(''.join(terminal_ansi), end='', indent=-1, wrap=False)

# Function for chancing the colormap of the terminal
def set_terminal_colormap(colormap):
    """This function constructs and apply a terminal colormap with
    256 - (16 + 2) = 238 ANSI/VT100 control sequences, remapping the
    238 higher colour numbers. The 16 + 2 = 18 lowest are left alone in
    order not to mess with standard terminal colouring and the colours
    used for the CO𝘕CEPT logo at startup.
    We apply the colormap even if the specified colormap is already
    in use, as the resulting log file is easier to parse with every
    colormap application present.
    """
    if not master:
        return
    matplotlib = get_matplotlib()
    colormap_ansi = getattr(matplotlib.cm, colormap)(linspace(0, 1, 238))[:, :3]
    for i, rgb in enumerate(colormap_ansi):
        colorhex = matplotlib.colors.rgb2hex(rgb)
        statechange = esc_set_color.format(18 + i, *[colorhex[c:c+2] for c in range(1, 7, 2)])
        # As this does not actually print anything on the screen,
        # we use the normal print function as to not mess with the
        # bookkeeping inside fancyprint.
        print(statechange, end='')

# Top-level function for rendering and saving 3D renders
@cython.pheader(
    # Arguments
    components=list,
    filename=str,
    # Locals
    component='Component',
    components_str=str,
    declaration=object,  # Render3DDeclaration
    declarations=list,
    img='float[:, :, ::1]',
    n_dumps='int',
    returns='void',
)
def render3D(components, filename):
    # Get render3D declarations
    declarations = get_render3D_declarations(components)
    # Count up number of 3D renders to be dumped to disk
    n_dumps = 0
    for declaration in declarations:
        n_dumps += declaration.do_image
    # Construct and save 3D render for each declaration
    for declaration in declarations:
        if not declaration.do_image:
            continue
        components_str = ', '.join([component.name for component in declaration.components])
        if len(declaration.components) > 1:
            components_str = f'{{{components_str}}}'
        masterprint(f'Rendering {components_str} in 3D ...')
        # Compute the 3D render
        img = compute_render3D(declarations, declaration)
        # Add text and background
        img = finalize_render3D(declaration, img)
        # Save 3D render image to a PNG file on disk
        save_render3D(declaration, img, filename, n_dumps)
        masterprint('done')

# Function for getting declarations for all needed 3D renders,
# given a list of components.
def get_render3D_declarations(components):
    matplotlib = get_matplotlib()
    # Look up declarations in cache
    cache_key = tuple(components)
    declarations = render3D_declarations_cache.get(cache_key)
    if declarations:
        return declarations
    # If no colour is assigned to a component by the user, a colour will
    # be automatically assigned. We define a few colormaps. If the
    # number of components exceeds the number of defined colormaps,
    # single-valued colours will be used for the rest.
    def implement_colormaps_default():
        def colormap_lims(a):
            return (0.1, 0.75 + 0.25*a)
        yield 'inferno', colormap_lims
        def colormap_lims(a):
            return (0.15, 0.85 + 0.15*a)
        yield 'viridis', colormap_lims
    colors_default = itertools.chain(
        implement_colormaps_default(),
        itertools.cycle([f'C{i}' for i in range(10)]),
    )
    # Dictionary for keeping track of assigned colours
    components_colors = {}
    def get_declarations(components, selections):
        # Get declarations with basic fields populated
        declarations = get_output_declarations(
            'render3D',
            components,
            selections,
            render3D_options,
            Render3DDeclaration,
        )
        # Helper functions
        def construct_func(f):
            if not callable(f):
                def f(*, f=f):
                    return f
            def f(*, f=f):
                return float(eval_func_of_t_and_a(f))
            return f
        def construct_pairfunc(f):
            p = any2list(f)
            if len(p) == 1:
                def f(*, f=f):
                    val = tuple([float(x) for x in any2list(eval_func_of_t_and_a(f))])
                    if len(val) == 1:
                        val *= 2
                    return val
            else:
                def f(*, f0=construct_func(p[0]), f1=construct_func(p[1])):
                    return (float(eval_func_of_t_and_a(f0)), float(eval_func_of_t_and_a(f1)))
            return f
        # Add missing declaration fields
        for index, declaration in enumerate(declarations):
            # Camera settings
            elevation = construct_func(declaration.elevation)
            azimuth   = construct_func(declaration.azimuth)
            roll      = construct_func(declaration.roll)
            zoom      = construct_func(declaration.zoom)
            projection, focal_length = declaration.projection
            focal_length = (
                (lambda: None) if focal_length is None
                else construct_func(focal_length)
            )
            def projection(
                *,
                projection=projection,
                focal_length=focal_length,
            ):
                return projection, focal_length()
            # The color field should be a list with one element for each
            # component, each element being a function of scale factor a,
            # realising a colormap. No α information will be stored.
            colormaps = declaration.color
            if not isinstance(colormaps, list):
                colormaps = [colormaps]
            colormaps += [None]*(len(declaration.components) - len(colormaps))
            for i, (c, component) in enumerate(zip(colormaps.copy(), declaration.components)):
                colormap = None
                if c is None:
                    c = is_selected(component, render3D_options['color'])
                if c is None:
                    color = components_colors.get(component.name)
                    if color is None:
                        # Assign next color(map)
                        color = next(colors_default)
                    if isinstance(color, tuple) and len(color) == 2:
                        colormap, colormap_lims = color
                    else:
                        rgb = to_rgb(color)
                    components_colors[component.name] = color
                elif isinstance(c, str) and hasattr(matplotlib.cm, c):
                    # Colormap
                    colormap, colormap_lims = c, (0, 1)
                elif (
                    isinstance(c, tuple) and isinstance(c[0], str) and hasattr(matplotlib.cm, c[0])
                    and len(c) in (2, 3)
                ):
                    # Colormap and limits
                    if len(c) == 2:
                        colormap, colormap_lims = c
                    else:
                        colormap, colormap_lims = c[0], c[1:]
                else:
                    # Colour
                    rgb = to_rgb(c)
                # Create function for realising colormap
                if colormap is None:
                    # Single colour specified
                    def colormap(*, rgb=rgb):
                        return asarray(
                            np.expand_dims(to_rgb(rgb), 0),
                            dtype=C2np['float'],
                        )
                else:
                    n_colors = 256
                    colormap_lims = construct_pairfunc(colormap_lims)
                    def colormap(
                        *, colormap=colormap, colormap_lims=colormap_lims, n_colors=n_colors,
                    ):
                        return np.ascontiguousarray(
                            getattr(matplotlib.cm, colormap)(
                                linspace(
                                    *colormap_lims(),
                                    n_colors,
                                )
                            )[:, :3],
                            dtype=C2np['float'],
                        )
                colormaps[i] = colormap
            # Process the 'enhancement' dict into a named tuple
            # of functions of the scale factor.
            enhancement = Render3DEnhancement(
                construct_func    (declaration.enhancement['contrast']),
                construct_pairfunc(declaration.enhancement['clip']),
                construct_func    (declaration.enhancement['α']),
                construct_func    (declaration.enhancement['brightness']),
            )
            # Evaluate the fontsize
            resolution = declaration.resolution
            fontsize = declaration.fontsize
            if isinstance(fontsize, str):
                fontsize = fontsize.lower()
                for resolution_str in ('resolution', 'res'):
                    fontsize = fontsize.replace(resolution_str, str(resolution))
                fontsize = eval_unit(fontsize)
            fontsize = float(fontsize)
            # Replace old declaration with a new, fully populated one
            declaration = declaration._replace(
                elevation=elevation,
                azimuth=azimuth,
                roll=roll,
                zoom=zoom,
                projection=projection,
                color=colormaps,
                enhancement=enhancement,
                fontsize=fontsize,
            )
            declarations[index] = declaration
        return declarations
    # Get main declarations
    declarations = get_declarations(components, render3D_select)
    # Some of the declarations may contain multiple components.
    # Such multi-component renders are constructed from several
    # single-component renders, which are reused across all declarations
    # containing a given component. If some components are present in
    # multi-component declarations but do not occur alone, we add in
    # extra, single-component declarations for these. This is needed as
    # various render options (interpolation options, color, etc.) are
    # inherent to the particular component, and thus will be used within
    # all renders containing the particular component. Other attributes
    # (resolution, camera angle, etc.) are determined solely by the
    # declaration.
    components_torender = {
        component
        for declaration in declarations
        for component in declaration.components
    }
    declarations_extra = []
    select = {'default': {'image': True}}
    for component in components_torender:
        for declaration in declarations:
            if declaration.components == [component]:
                break
        else:
            declaration_extra = get_declarations([component], select)[0]
            # This declaration should not produce output of its own
            declaration_extra = declaration_extra._replace(do_image=False)
            declarations_extra.append(declaration_extra)
    declarations += declarations_extra
    # Store declarations in cache and return
    render3D_declarations_cache[cache_key] = declarations
    return declarations
# Cache used by the get_render3D_declarations() function
cython.declare(render3D_declarations_cache=dict)
render3D_declarations_cache = {}
# Create the Render3DDeclaration type
fields = (
    'components', 'do_image', 'gridsize',
    'interpolation', 'deconvolve', 'interlace',
    'azimuth', 'elevation', 'roll', 'zoom', 'projection',
    'color', 'depthshade', 'enhancement', 'background',
    'fontsize', 'resolution',
)
Render3DDeclaration = collections.namedtuple(
    'Render3DDeclaration', fields, defaults=[None]*len(fields),
)
# Create the Render3DEnhancement type
Render3DEnhancement = collections.namedtuple(
    'Render3DEnhancement', ('contrast', 'clip', 'α', 'brightness'),
)

# Function for computing a 3D render
@cython.header(
    # Arguments
    declarations=list,
    declaration=object,  # Render3DDeclaration
    # Locals
    colormap_func=object,
    component='Component',
    img='float[:, :, ::1]',
    img_full='float[:, :, ::1]',
    singlecomponent_img_key=object,  # Render3DSingleComponentImgKey
)
def compute_render3D(declarations, declaration):
    """The full render will only be returned on the master process"""
    # Clean up outdated images in the global store
    for singlecomponent_img_key in render3D_singlecomponent_imgs.copy():
        if singlecomponent_img_key.time != universals.t:
            render3D_singlecomponent_imgs.pop(singlecomponent_img_key)
    # Fetch figure, axis and scatter artist
    fig, ax, arts = fetch_render3D_fig(declaration, autoscale=True)
    art = arts.scatter
    # Handle each component in turn
    img_full = None
    for component in declaration.components:
        img = compute_render3D_single(declarations, declaration, component, fig, art)
        if not master:
            continue
        # Blend component renders together
        if img_full is None:
            img_full = img
        else:
            blend_render3D(img_full, img, mode='overunder')
    if not master:
        return img_full
    # The master process now stores the full image, possibly made up of
    # several single-component images. Though each single-component
    # image has had its brightness enhanced, we still perform brightness
    # enhancement on the combined image.
    if len(declaration.components) > 1:
        enhance_brightness_render3D(declaration, img_full)
    return img_full

# Function for computing a 3D render of a single component
@cython.header(
    # Arguments
    declarations=list,
    declaration=object,  # Render3DDeclaration
    component='Component',
    fig=object,  # matplotlib.figure.Figure
    art=object,
    # Locals
    singlecomponent_img_key=object,  # Render3DSingleComponentImgKey
    colormap_func=object,
    declaration_single=object,  # Render3DDeclaration
    img='float[:, :, ::1]',
    already_rendered='bint',
    x='double[:]',
    y='double[:]',
    z='double[:]',
    slab='double[:, :, ::1]',
    size='Py_ssize_t',
    size_local='Py_ssize_t',
    marker_size='double',
    rgbα='float[:, ::1]',
    blend_mode=str,
    img_buff='float[:, :, ::1]',
    n_merge_steps='int',
    step='int',
    source='int',
    returns='float[:, :, ::1]',
)
def compute_render3D_single(declarations, declaration, component, fig, art):
    """The passed declarations should contain all declarations.
    The additional declaration passed by its own is the declaration
    which governs the current render. This can be of either a single or
    of multiple components. The render of one of these components are
    computed by this function, specifically that which is passed.
    Only the master process returns the image.
    """
    # Find the unique single-component declaration for the given
    # component among all the declarations.
    for declaration_single in declarations:
        if declaration_single.components == [component]:
            break
    else:
        abort(
            f'compute_render3D_single(): No single-component declaration for component '
            f'"{component.name}" found amongst the passed declarations.'
        )
    # Look up single-component 3D render in global store
    img = None
    already_rendered = False
    if master:
        singlecomponent_img_key = Render3DSingleComponentImgKey(
            component.name,
            universals.t,
            # Use values from the multi-component declaration
            declaration.elevation(),
            declaration.azimuth(),
            declaration.roll(),
            declaration.zoom(),
            declaration.projection(),
            declaration.depthshade,
            declaration.resolution,
        )
        img = render3D_singlecomponent_imgs.get(singlecomponent_img_key)
        already_rendered = (img is not None)
        if already_rendered:
            # Take copy of single-component image,
            # ensuring that it will not be mutated.
            img = asarray(img).copy()
    already_rendered = bcast(already_rendered)
    if already_rendered:
        return img
    # Get scatter positions and density values
    x, y, z, slab, size, size_local = fetch_render3D_data(
        component,
        declaration_single.gridsize, declaration_single.interpolation,
        declaration_single.deconvolve, declaration_single.interlace,
    )
    # Get scatter size
    marker_size = get_render3D_marker(fig, size, declaration.zoom())
    # Get colour and α for each scatter marker
    rgbα = compute_render3D_rgbα(
        slab, size, size_local,
        declaration_single.color[0],  # same as declaration.color[i] with i the index for this component
        declaration.depthshade,       # depthshade defined by multi-component declaration
        declaration_single.enhancement,
    )
    # Update plot
    art._offsets3d = (x, y, z)  # _offsets also exists but should not be touched
    facecolors_attr = asarray(rgbα)
    for attr in ['_facecolors', '_facecolor3d']:
        if hasattr(art, attr):
            setattr(art, attr, facecolors_attr)
    sizes_attr = [marker_size]
    for attr in ['_sizes', '_sizes3d']:
        if hasattr(art, attr):
            setattr(art, attr, sizes_attr)
    # Save to in-memory image array
    img = render_render3D(fig)
    # Determine blending mode for partial images
    if slab is None:
        # As the partial renders are not of slabs, no simple ordering of
        # these exists. We use 'overunder' blending, which amounts to
        # averaging 'over' and 'under'.
        blend_mode = 'overunder'
    else:
        # The partial renders (of slabs) will be blended together with a
        # blend mode of either 'over' or 'under', in accordance with the
        # current azimuthal viewing angle.
        blend_mode = (
            'over' if π/2 < np.mod(declaration.azimuth(), 2*π) < 3*π/2
            else 'under'
        )
    # The scheme employed below for merging the partial images requires
    # a buffer image to be allocated on some processes and takes place
    # over n_merge_steps steps.
    img_buff = None
    n_merge_steps = ilog2(nprocs) + (not ispowerof2(nprocs))
    for step in range(n_merge_steps):
        if rank%(2**(1 + step)):
            # Send image, after which this process is done
            Send(img, dest=(rank - 2**step))
            break
        else:
            # Receive external image
            source = rank + 2**step
            if source >= nprocs:
                continue
            if img_buff is None:
                img_buff = empty(asarray(img).shape, dtype=C2np['float'])
            Recv(img_buff, source=source)
            # Blend received and local image together
            blend_render3D(img, img_buff, blend_mode)
    # The master process now stores the complete image
    if not master:
        img = None  # encourage garbage collection
        return img
    # Shrink possibly enlarged image down to the intended resolution
    img = resize_render3D(img, declaration.resolution)
    # Enhance the brightness in accordance with
    # the single-component declaration.
    enhance_brightness_render3D(declaration_single, img)
    # Store single-component render in the global store
    render3D_singlecomponent_imgs[singlecomponent_img_key] = img
    # Return copy of single-component image,
    # ensuring that it will not be mutated.
    img = asarray(img).copy()
    return img
# Global store of single-component renders
cython.declare(render3D_singlecomponent_imgs=dict)
render3D_singlecomponent_imgs = {}
# Create the Render3DSingleComponentImgKey type
Render3DSingleComponentImgKey = collections.namedtuple(
    'Render3DSingleComponentImgKey',
    (
        'name', 'time',
        'elevation', 'azimuth', 'rolll', 'zoom',
        'projection', 'depthshade', 'resolution',
    ),
)

# Function for setting up the reused figure, axes and artists
# for 3D renders.
def fetch_render3D_fig(declaration, autoscale=False):
    plt = get_matplotlib().pyplot
    if render3D_fig:
        # Extract from global store
        fig, ax, arts = render3D_fig
        # Update axis and figure in accordance with declaration options
        # common to all components.
        ax.view_init(
            elev=declaration.elevation()*180/π,
            azim=declaration.azimuth()  *180/π,
            roll=declaration.roll()     *180/π,
        )
        zoom = declaration.zoom()
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
        ax.set_proj_type(*declaration.projection())
        # Set figure size to match requested resolution
        resolution = declaration.resolution
        figsize = resolution/fig.get_dpi()
        fig.set_size_inches(figsize, figsize)
        # The figure size affects the scatter marker sizes.
        # Too small a scatter marker size makes the render less bright,
        # or even completely invisible. Scale up the figure size (and
        # hence the resolution) to ensure a minimum allowed marker size.
        if autoscale:
            size = np.max([
                get_render3D_size(component, declaration.gridsize, declaration.interpolation)
                for component in declaration.components
            ])
            marker_size = get_render3D_marker(fig, size, zoom)
            if marker_size < marker_size_threshold:
                figsize *= sqrt(marker_size_threshold/marker_size)
                resolution = int(ceil(figsize*fig.get_dpi()))
                resolution += resolution%2
                figsize = resolution/fig.get_dpi()
                fig.set_size_inches(figsize, figsize)
        # Update scatter artist in accordance with declaration options
        # common to all components.
        arts.scatter._depthshade = declaration.depthshade
        # Update text artists in accordance with declaration options
        # common to all components. The text colour is either black
        # or white, depending on the brightness of the background.
        if declaration.background is None:
            # Special case of transparent background
            text_color = 'black'
        else:
            # Solid background
            text_color = 'white'
            if get_perceived_brightness(declaration.background) > 0.5:
                text_color = 'black'
        for art in arts.text:
            art.set_color(text_color)
            art.set_fontsize(declaration.fontsize*resolution/declaration.resolution)
        # Clear out positions of scatter artist
        arts.scatter._offsets3d = tuple([
            asarray(offset3d[:0]).copy()
            for offset3d in arts.scatter._offsets3d
        ])
        # Clear out rgbα of scatter artist
        for attr in ['_facecolors', '_facecolor3d']:
            facecolors_attr = getattr(arts.scatter, attr, None)
            if facecolors_attr is not None:
                setattr(arts.scatter, attr, asarray(facecolors_attr)[:0, :].copy())
        # Clear out text data of text artists
        for art in arts.text:
            art.set_text('')
        return render3D_fig
    # Create 3D figure and axis
    dpi = 100
    fig = plt.figure(dpi=dpi)  # this figure will never be closed
    ax = fig.add_subplot(projection='3d')
    # Create scatter artist.
    # We place a particle in each corner, for use with tight layout.
    x, y, z = [], [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x.append(i*boxsize)
                y.append(j*boxsize)
                z.append(k*boxsize)
    art_scatter = ax.scatter(
        x, y, z,
        marker='.',
        linewidth=0,
    )
    # Explicitly set an rgbα array as a scatter artist attribute
    rgbα = zeros((1, 4), dtype=C2np['float'])
    for attr in ['_facecolors', '_facecolor3d']:
        if hasattr(art_scatter, attr):
            setattr(art_scatter, attr, rgbα)
    # Create text artists
    spacing = 0.08
    arts_text = [
        ax.text2D(
            {'left': spacing, 'right': 1 - spacing}[alignment],
            spacing,
            '',
            horizontalalignment=alignment,
            transform=ax.transAxes,
        )
        for alignment in ['left', 'right']
    ]
    # Collect all artists
    arts = Render3DArtists(art_scatter, arts_text)
    # Basic axis setup, common for all 3D renders
    ax.set_axis_off()
    ax.set_xlim(0, boxsize)
    ax.set_ylim(0, boxsize)
    ax.set_zlim(0, boxsize)
    # Set tight layout, using the 8 particles at the simulation box
    # corners to form the bounding box.
    fig.tight_layout(pad=0)
    # Store figure, axis and artist globally
    render3D_fig.append(fig)
    render3D_fig.append(ax)
    render3D_fig.append(arts)
    # With the figure, axis and artists created, call this function once
    # more to set their attributes according to the passed declaration.
    return fetch_render3D_fig(declaration, autoscale)
# Global store used by the fetch_render3D_fig() function
cython.declare(render3D_fig=list)
render3D_fig = []
# Create the Render3DArtists type
Render3DArtists = collections.namedtuple('Render3DArtists', ('scatter', 'text'))

# Function for converting RGB to perceived brightness value
def get_perceived_brightness(rgb):
    """Converting RGB (really sRGB; https://en.wikipedia.org/wiki/SRGB)
    to brightness level as perceived by humans is non-trivial.
    Here we make use of the following particular StackOverflow answer:
      https://stackoverflow.com/a/56678483/4056181
    Note that what we refer to as 'perceived brightness' is then really
    the 'perceived lightness'.
    """
    rgb_linear = [
        (v/12.92 if v <= 0.04045 else ((v + 0.055)/1.055)**2.4)
        for v in rgb
    ]
    luminance = np.dot((0.2126, 0.7152, 0.0722), rgb_linear)
    lightness_perceived = (
        luminance*903.3
        if luminance <= 0.008856
        else luminance**(1./3.)*116 - 16
    )/100
    lightness_perceived = np.max((0, lightness_perceived))
    lightness_perceived = np.min((1, lightness_perceived))
    return lightness_perceived

# Function for retrieving the reusable scatter coordinates for
# 3D renders of interpolated components.
@cython.header(
    # Arguments
    component='Component',
    gridsize='Py_ssize_t',
    interpolation='int',
    deconvolve='bint',
    interlace=str,
    # Locals
    dim='int',
    i='Py_ssize_t',
    index='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    posx='double[:]',
    posy='double[:]',
    posz='double[:]',
    size='Py_ssize_t',
    size_i='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    size_local='Py_ssize_t',
    slab='double[:, :, ::1]',
    slab_start_i='Py_ssize_t',
    x='double[::1]',
    xi='double',
    y='double[::1]',
    yj='double',
    z='double[::1]',
    zk='double',
    returns=tuple,
)
def fetch_render3D_data(component, gridsize, interpolation, deconvolve, interlace):
    """This functions returns arrays x, y, z, holding coordinate values
    for the scatter markers. For particle components, the exact particle
    positions will be used if interpolation == 0. Otherwise, a density
    grid will be constructed through interpolation, with x, y and z then
    being the grid coordinates.
    The density grid is also returned,
    in slab format. In order for the depthshading to be applied
    similarly by all processes (with their slabs located at different
    positions), we add fake scatter markers in the box corners.
    """
    size = get_render3D_size(component, gridsize, interpolation)
    if component.representation == 'particles' and interpolation == 0:
        # Use raw particle distribution
        slab = None
        # Use the (non-contiguous) particle data arrays directly.
        # Add the fake corner markers to the end.
        size_local = component.N_local + 8
        if component.N_allocated < size_local:
            component.resize(size_local)
        posx = component.posx[:size_local]
        posy = component.posy[:size_local]
        posz = component.posz[:size_local]
        index = component.N_local
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    posx[index] = i*boxsize
                    posy[index] = j*boxsize
                    posz[index] = k*boxsize
                    index += 1
        return posx, posy, posz, slab, size, size_local
    # Interpolate component onto slab-decomposed grid.
    # Note that though the returned slab is in real space, it has gone
    # through Fourier space where it had its Nyquist values nullified.
    slab = interpolate_upstream(
        [component], [component.render3D_upstream_gridsize], gridsize, 'ρ', interpolation,
        deconvolve=deconvolve, interlace=interlace,
        output_space='real', output_as_slabs=True,
    )
    size_i = slab.shape[0]
    size_j = slab.shape[1]
    size_k = gridsize  # ignore padding in last dimension
    size_local = size_i*size_j*size_k + 8
    if render3D_xyz_prevkey == [gridsize]:
        # Reuse global x, y and z from previous call
        x = render3D_xyz[0][:size_local]
        y = render3D_xyz[1][:size_local]
        z = render3D_xyz[2][:size_local]
        return x, y, z, slab, size, size_local
    render3D_xyz_prevkey[:] = [gridsize]
    # Resize global arrays
    if render3D_xyz[0].size < size_local:
        for dim in range(3):
            render3D_xyz[dim].resize(size_local, refcheck=False)
    x = render3D_xyz[0][:size_local]
    y = render3D_xyz[1][:size_local]
    z = render3D_xyz[2][:size_local]
    # Populate arrays with actual positions
    index = 0
    slab_start_i = size_i*rank
    for i in range(size_i):
        xi = (ℝ[slab_start_i + 0.5*cell_centered] + i)*ℝ[boxsize/gridsize]
        for j in range(size_j):
            yj = (ℝ[0.5*cell_centered] + j)*ℝ[boxsize/gridsize]
            for k in range(size_k):
                zk = (ℝ[0.5*cell_centered] + k)*ℝ[boxsize/gridsize]
                x[index] = xi
                y[index] = yj
                z[index] = zk
                index += 1
    # Add the fake corner markers to the end
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x[index] = i*boxsize
                y[index] = j*boxsize
                z[index] = k*boxsize
                index += 1
    return x, y, z, slab, size, size_local
# Global arrays and shape used by the fetch_render3D_data() function
cython.declare(
    render3D_xyz=tuple,
    render3D_xyz_prevkey=list,
)
render3D_xyz = (
    empty(1, dtype=C2np['double']),
    empty(1, dtype=C2np['double']),
    empty(1, dtype=C2np['double']),
)
render3D_xyz_prevkey = [-1]

# Function returning the number of scatter markers
# to use for 3D rendering of a component.
@cython.header(
    # Arguments
    component='Component',
    gridsize='Py_ssize_t',
    interpolation='int',
    # Locals
    size='Py_ssize_t',
    returns='Py_ssize_t',
)
def get_render3D_size(component, gridsize, interpolation):
    if component.representation == 'particles' and interpolation == 0:
        size = component.N
    else:
        size = gridsize**3
    return size

# Function for determining appropriate scatter marker size
def get_render3D_marker(fig, size, zoom):
    if size == 0:
        # Nothing to plot
        return marker_size_threshold
    # Set the scatter marker size of the spherical markers such that
    # the spheres kiss when they are packed in a cubic lattice
    # (viewed with orthographic projection).
    figsize = np.mean(fig.get_size_inches())
    marker_size = (83*zoom*figsize/cbrt(size))**2
    # Grow the scatter size such that planes of markers have
    # no holes (viewed with orthographic projection).
    marker_size *= 2
    # Shrink the marker size for large sizes (scatter marker counts)
    marker_size *= 10.5/log(1 + size)
    return marker_size
# Using a scatter marker size too low will result in artificial
# darkening of the render, or even leading to completely invisible
# markers. This limit depend on the α in some complicated manner.
# The below threshold is chosen rather conservative.
cython.declare(marker_size_threshold='double')
marker_size_threshold = 5

# Function computing the colours and α values
# of the scatter markers in 3D renders.
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    size='Py_ssize_t',
    size_local='Py_ssize_t',
    colormap_func=object,
    depthshade='bint',
    enhancement=object,  # Render3DEnhancement
    # Locals
    bin_edges='double[::1]',
    bins='Py_ssize_t[::1]',
    c='Py_ssize_t',
    clip_lower='double',
    clip_upper='double',
    colormap='float[:, ::1]',
    exponent='double',
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    index='Py_ssize_t',
    index_color='Py_ssize_t',
    index_rgbα='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    logμ='double',
    logμ_err='double',
    n_bins='Py_ssize_t',
    n_counts='Py_ssize_t',
    rgbα='float[:, ::1]',
    rgbα_constant='float[::1]',
    rgbα_constant_ptr='float*',
    size_local_actual='Py_ssize_t',
    slab_max='double',
    slab_min='double',
    slab_ptr='double*',
    slab_size='Py_ssize_t',
    value='double',
    vmax='double',
    vmax_index='Py_ssize_t',
    vmin='double',
    vmin_index='Py_ssize_t',
    α='float',
    α_fac='float',
    α_fac_min='float',
    α_value='float',
    μ_goal='double',
    σ='double',
    σ_err='double',
    returns='float[:, ::1]',
)
def compute_render3D_rgbα(slab, size, size_local, colormap_func, depthshade, enhancement):
    # Set α. Not that for grid rendering, this does not represent the
    # final α value applied to the markers.
    if slab is None:
        α = 0.02
    else:
        α = 150
    # Apply further α tuning unless disabled
    if enhancement.α() != -1:
        α *= enhancement.α()
        # Using depthshade lowers the α of scatter markers according to
        # their depth/distance, which thus also has the effect of
        # lowering the overall α. Here we counteract this.
        if depthshade:
            α *= 1.4
    # The passed size_local includes an additional 8 corner markers
    size_local_actual = size_local - 8
    # Realise colormap at the current time
    colormap = colormap_func()
    # Fetch global rgbα array
    if render3D_rgbα.size < 4*size_local:
        render3D_rgbα.resize(4*size_local)
    rgbα = render3D_rgbα[:4*size_local].reshape((size_local, 4))
    # Handle the case of a raw particle distribution
    # not interpolated onto a grid.
    if slab is None:
        α = pairmax(α, α_threshold)
        α = pairmin(α, 1)
        # Use midpoint of colormap as constant colour
        rgbα_constant = asarray(
            list(colormap[colormap.shape[0]//2, :]) + [α],
            dtype=C2np['float'],
        )
        rgbα_constant_ptr = cython.address(rgbα_constant[:])
        # Populate rgbα with data
        for index in range(size_local_actual):
            for c in range(4):
                rgbα[index, c] = rgbα_constant_ptr[c]
        # Make the fake corner markers completely transparent
        for index in range(size_local_actual, size_local):
            for c in range(4):
                rgbα[index, c] = 0
        return rgbα
    # We are dealing with a slab of the density field.
    # Extract slab shape and pointer.
    gridsize = slab.shape[1]
    slab_size = slab.shape[0]*slab.shape[1]*slab.shape[2]
    slab_ptr = cython.address(slab[:, :, :])
    # Find global minimum and maximum value
    slab_min = slab_max = slab_ptr[0]
    for index, i, j, k in slab_loop(gridsize, skip_padding=True):
        value = slab_ptr[index]
        if value < slab_min:
            slab_min = value
        elif value > slab_max:
            slab_max = value
    slab_min = allreduce(slab_min, op=MPI.MIN)
    slab_max = allreduce(slab_max, op=MPI.MAX)
    # Perform contrast, α and clipping enhancements
    vmin, vmax = 0, 1
    if slab_min == slab_max:
        # Completely homogeneous (possibly empty) grid
        if slab_max == 0:
            for i in range(slab_size):
                slab_ptr[i] = 0
        else:
            for i in range(slab_size):
                slab_ptr[i] = 0.5
    else:
        # Shift and scale slab values so that they are between 0 and 1
        for i in range(slab_size):
            slab_ptr[i] = (slab_ptr[i] - slab_min)*ℝ[1/(slab_max - slab_min)]
        # Contrast enhancement
        μ_goal = enhancement.contrast()
        if μ_goal != -1:
            # The log of the slab data is typically close to Gaussian.
            # Compute histogram of logged slab values.
            bin_edges, bins = compute_slab_histogram(slab, size, apply_log=True)
            # Fit histogram to Gaussian with mean μ
            (_, logμ, _), (_, logμ_err, _) = bcast(
                fit_histogram_gaussian(bin_edges, bins)
                if master else None
            )
            if logμ_err != -1:
                # Find exponent required to shift Gaussian
                # to have mean μ_goal.
                μ_goal = pairmin(1 - machine_ϵ, μ_goal)
                μ_goal = pairmax(machine_ϵ, μ_goal)
                exponent = log(μ_goal)/logμ
                exponent = pairmax(machine_ϵ, exponent)
                # Apply exponent to data. Here it is important that the
                # slab does not contain any negative values, including
                # in the padding region. Note that this is already
                # ensured due to the call to compute_slab_histogram().
                for i in range(slab_size):
                    slab_ptr[i] **= exponent
        # Recompute histogram for the shifted distribution
        bin_edges, bins = compute_slab_histogram(slab, size)
        # Master determines α and clipping
        α_fac = 1
        α_fac_min = 1e-3
        if master:
            # Set α enhancement from relative error on the fitted σ
            if enhancement.α() != -1:
                (_, _, σ), (_, _, σ_err) = fit_histogram_gaussian(bin_edges, bins)
                if σ_err != -1:
                    α_fac = pairmax(α_fac_min, σ_err/σ)
            # Determine vmin and vmax for clipping
            n_bins = bins.shape[0]
            n_counts = np.sum(bins)
            clip_lower, clip_upper = enhancement.clip()
            if clip_lower > clip_upper:
                clip_lower, clip_upper = clip_upper, clip_lower
            clip_lower = pairmax(0, clip_lower)
            clip_upper = pairmin(1, clip_upper)
            vmin_index, vmax_index = trim_histogram(
                bins,
                int(round(clip_lower*n_counts)),
                int(round((1 - clip_upper)*n_counts)),
            )
            vmin = bin_edges[vmin_index]
            vmax = bin_edges[vmax_index]
        α_fac = bcast(α_fac)
        vmin, vmax = bcast((vmin, vmax))
        # Apply α factor
        α *= α_fac
        α = pairmax(α, 0)
        α = pairmin(α, 1)
        # Clip (saturate) lower and upper values
        if vmin == vmax:
            # Completely homogeneous (possibly empty) grid
            for i in range(slab_size):
                slab_ptr[i] = vmin
        else:
            for i in range(slab_size):
                value = (slab_ptr[i] - vmin)*ℝ[1/(vmax - vmin)]
                if value < 0:
                    value = 0
                elif value > 1:
                    value = 1
                slab_ptr[i] = value
    # Populate rgbα with data
    index_rgbα = -1
    for index, i, j, k in slab_loop(gridsize, skip_padding=True):
        index_rgbα += 1
        value = slab_ptr[index]
        index_color = int(value*ℝ[colormap.shape[0]*(1 - machine_ϵ)])
        for c in range(3):
           rgbα[index_rgbα, c] = colormap[index_color, c]
        α_value = pairmin(α*value, 1)
        if α_value > 0:
            rgbα[index_rgbα, 3] = pairmax(α_value, α_threshold)
        rgbα[index_rgbα, 3] = α_value
    # Make the fake corner markers completely transparent
    for index_rgbα in range(size_local_actual, size_local):
        for c in range(4):
            rgbα[index_rgbα, c] = 0
    return rgbα
# Global array used by the compute_render3D_rgbα() function
cython.declare(
    render3D_rgbα=object,  # np.ndarray
)
render3D_rgbα = empty(1, dtype=C2np['float'])
# There exist a lower limit for the α value of individual markers,
# below which they are rendered completely invisible. Markers with α
# below this limit but strictly greater than 0 are assigned this
# limiting value. The α limit is discussed e.g. here, where they
# find it to be 1/256:
#   https://github.com/matplotlib/matplotlib/issues/2287
# At least for the present case of a 3D scatter plot, we find that
# the actual limit is really 1/170. We choose a conservative 1/160.
cython.declare(α_threshold='float')
α_threshold = 1./160.

# Function for computing histogram of slab values
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    size='Py_ssize_t',
    apply_log='bint',
    # Locals
    bin_edges='double[::1]',
    bins='Py_ssize_t[::1]',
    index='Py_ssize_t',
    n_bins='Py_ssize_t',
    n_bins_fac='double',
    n_bins_max='Py_ssize_t',
    n_bins_min='Py_ssize_t',
    size_i='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    slab_2ndmin='double',
    slab_max='double',
    slab_ptr='double*',
    slab_size='Py_ssize_t',
    value='double',
    returns=tuple,
)
def compute_slab_histogram(slab, size, apply_log=False):
    """It is assumed that the slab values are normalized to be between
    0 and 1. The values of the padding region will be ignored.
    Only the master process will hold the total histogram.
    If apply_log is True, a transformation slab → log(slab) will be
    applied prior to computing the histogram. To save memory, this is
    done in-place. The slab will be transformed back, log(slab) → slab,
    before the function returns.
    """
    # Extract slab shape
    size_i = slab.shape[0]
    size_j = slab.shape[1]
    size_k = size_j  # padded dimension
    # We first populate the slab padding with values of 0.
    # The entire contiguous slab may then be passed to np.histogram().
    fill_slab_padding(slab, 0)
    # We want to know the smallest and largest data values. These are
    # probably 0 and 1, though we do not care about the 0. We thus find
    # the next smallest value, along with the largest. If the data is to
    # be log transformed, we do so at the same time.
    slab_ptr = cython.address(slab[:, :, :])
    slab_size = slab.shape[0]*slab.shape[1]*slab.shape[2]
    if apply_log:
        slab_2ndmin = +ထ
        slab_max    = -ထ
        for index in range(slab_size):
            value = slab_ptr[index]
            if value == 0:
                slab_ptr[index] = -ထ
                continue
            value = log(value)
            slab_ptr[index] = value
            if value < slab_2ndmin:
                slab_2ndmin = value
            if value > slab_max:
                slab_max = value
    else:
        slab_2ndmin = slab_max = slab_ptr[0]
        for index in range(slab_size):
            value = slab_ptr[index]
            if value == 0:
                continue
            if value < slab_2ndmin:
                slab_2ndmin = value
            elif value > slab_max:
                slab_max = value
    slab_2ndmin = allreduce(slab_2ndmin, op=MPI.MIN)
    slab_max    = allreduce(slab_max,    op=MPI.MAX)
    # Compute histogram
    n_bins_min, n_bins_max = 2**6, 2**20
    n_bins_fac = 4
    n_bins = int(n_bins_fac*sqrt(size))
    n_bins = pairmax(n_bins, n_bins_min)
    n_bins = pairmin(n_bins, n_bins_max)
    bins, bin_edges = np.histogram(slab, n_bins, range=(slab_2ndmin, slab_max))
    Reduce(
        sendbuf=(MPI.IN_PLACE if master else bins),
        recvbuf=(bins         if master else None),
        op=MPI.SUM,
    )
    # Undo log transformation
    if apply_log:
        for index in range(slab_size):
            slab_ptr[index] = exp(slab_ptr[index])
    return bin_edges, bins

# Function for fitting histogram data to Gaussian
@cython.header(
    # Arguments
    bin_edges='double[::1]',
    bins='Py_ssize_t[::1]',
    # Locals
    a='double',
    a_err='double',
    count='Py_ssize_t',
    n_bins='Py_ssize_t',
    n_counts='Py_ssize_t',
    oneσ_left='double',
    oneσ_left_index='Py_ssize_t',
    oneσ_right='double',
    oneσ_right_index='Py_ssize_t',
    μ='double',
    μ_err='double',
    σ='double',
    σ_err='double',
    returns=tuple,
)
def fit_histogram_gaussian(bin_edges, bins):
    import scipy.optimize
    # Boundaries for fitting parameters a, μ and σ,
    # based purely on the size of the input.
    n_bins = bins.shape[0]
    n_counts = np.sum(bins)
    bounds = (
        [       1, bin_edges[     0], bin_edges[     1] - bin_edges[0]],
        [n_counts, bin_edges[n_bins], bin_edges[n_bins] - bin_edges[0]],
    )
    # Determine initial guess on fitting parameters by iterating over
    # the histogram from both sides, stopping when the central area
    # corresponds to 2×1σ.
    count = int(n_counts*(1 - erf(1/sqrt(2)))/2)
    oneσ_left_index, oneσ_right_index = trim_histogram(bins, count, count)
    oneσ_left = bin_edges[oneσ_left_index]
    oneσ_right = bin_edges[oneσ_right_index]
    a = np.max(bins[oneσ_left_index:oneσ_right_index+1])
    μ = 0.5*(oneσ_left + oneσ_right)
    σ = 0.5*(oneσ_right - oneσ_left)
    a_err = -1
    μ_err = -1
    σ_err = -1
    # Perform the fitting
    popt = None
    try:
        popt, pcov = scipy.optimize.curve_fit(
            gaussian,
            bin_edges[:n_bins],
            bins,
            (a, μ, σ),
            check_finite=False,
            bounds=bounds,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            maxfev=1_000,
        )
    except Exception:
        pass
    if popt is not None:
        a, μ, σ = popt
        a_err, μ_err, σ_err = np.sqrt(np.diag(pcov))
    return (a, μ, σ), (a_err, μ_err, σ_err)
# Helper function for fit_histogram_gaussian()
def gaussian(x, a, μ, σ):
    return a*np.exp(-0.5*((x - μ)/σ)**2)

# Function for finding left and right indices into histogram,
# given left and right counts.
@cython.header(
    # Arguments
    bins='Py_ssize_t[::1]',
    count_left='Py_ssize_t',
    count_right='Py_ssize_t',
    # Locals
    bins_ptr='Py_ssize_t*',
    count='Py_ssize_t',
    i='Py_ssize_t',
    index_left='Py_ssize_t',
    index_right='Py_ssize_t',
    n_bins='Py_ssize_t',
    returns=tuple,
)
def trim_histogram(bins, count_left, count_right):
    n_bins = bins.shape[0]
    bins_ptr = cython.address(bins[:])
    # Left side
    index_left = 0
    count = 0
    for i in range(n_bins):
        count += bins_ptr[i]
        if count >= count_left:
            index_left = i
            break
    # Right side
    index_right = n_bins
    count = 0
    for i in range(n_bins - 1, -1, -1):
        count += bins_ptr[i]
        if count >= count_right:
            index_right = i + 1
            break
    # Expand the central region slightly,
    # ensuring index_left != index_right.
    if index_left > index_right:
        index_left, index_right = index_right, index_left
    index_left = pairmax(0, index_left - 1)
    index_right = pairmin(n_bins, index_right + 1)
    return index_left, index_right

# Function for generating an image array
# from the passed figure.
def render_render3D(fig):
    plt = get_matplotlib().pyplot
    # Save figure to transparent, in-memory image array
    with io.BytesIO() as f:
        fig.savefig(f, transparent=True)  # do not set dpi
        f.seek(0)
        img = plt.imread(f)
    # Completely transparent pixels (α = 0) will be assigned colour
    # values of 1 (white) by Matplotlib, but for image manipulations to
    # come we want such pixels to have colour values of 0 (black).
    img[img[:, :, 3] == 0, :3] = 0
    return img

# Function for adding text and background colour
# to an already computed 3D render.
@cython.header(
    # Arguments
    declaration=object,  # Render3DDeclaration
    img='float[:, :, ::1]',
    # Locals
    background=object,  # np.ndarray
    img_text='float[:, :, ::1]',
    returns='float[:, :, ::1]',
)
def finalize_render3D(declaration, img):
    if not master:
        return img
    # Do not alter the passed image in-place, so that it might be used
    # in combination renders as well.
    img = asarray(img).copy()
    # Normalize to max(α) = 1
    normalize_α_render3D(img)
    # Ensure pixel values to be in the interval [0, 1]
    truncate_saturated_render3D(img)
    # Fetch figure, axis and text artists
    fig, ax, arts = fetch_render3D_fig(declaration)
    arts = arts.text
    # Set text
    arts[0].set_text(
        r'$t = {}\, \mathrm{{{}}}$'
        .format(significant_figures(universals.t, 4, 'TeX'), unit_time)
    )
    if enable_Hubble:
        arts[1].set_text(
            r'$a = {}$'
            .format(significant_figures(universals.a, 4, 'TeX'))
        )
    # Compute 3D render for the text only
    img_text = render_render3D(fig)
    # Blend text render into scatter render
    blend_render3D(img, img_text, mode='under')
    # Blend solid background into scatter render,
    # unless a transparent background is specified.
    if declaration.background is not None:
        background = ones([1, 1, 4], dtype=C2np['float'])
        background[0, 0, :3] = declaration.background
        blend_render3D(img, background, mode='over')
    # Ensure pixel values to be in the interval [0, 1]
    truncate_saturated_render3D(img)
    return img

# Function for enhancing the brightness of an image
# in accordance with enhancement.brightness().
@cython.header(
    # Arguments
    declaration=object,  # Render3DDeclaration
    img='float[:, :, ::1]',
    # Locals
    brightness='float',
    brightness_target='float',
    fac_brightness='float',
    fac_brightness_max='float',
    fac_brightness_min='float',
    img_copy='float[:, :, ::1]',
    large='Py_ssize_t',
    n='Py_ssize_t',
    returns='void',
)
def enhance_brightness_render3D(declaration, img):
    # Brighten (or darken) the image
    brightness_target = declaration.enhancement.brightness()
    if brightness_target == -1:
        return
    large = 20
    brightness_target = pairmax(0, brightness_target)
    brightness_target = pairmin(2**large, brightness_target)
    fac_brightness = 1
    fac_brightness_min = fac_brightness_max = fac_brightness
    img_copy = asarray(img).copy()
    brightness = get_perceived_brightness(
        brighten_render3D(img_copy, fac_brightness, measure_rms=True)
    )
    if brightness < brightness_target:
        fac_brightness_min = fac_brightness
        for n in range(1, large):
            brightness = get_perceived_brightness(
                brighten_render3D(img_copy, 2, measure_rms=True)
            )
            if brightness >= brightness_target:
                break
        fac_brightness_max = fac_brightness*2**n
    elif brightness > brightness_target:
        fac_brightness_max = fac_brightness
        for n in range(1, large):
            brightness = get_perceived_brightness(
                brighten_render3D(img_copy, 0.5, measure_rms=True)
            )
            if brightness <= brightness_target:
                break
        fac_brightness_min = fac_brightness/2**n
    brightness = -1
    while (
            abs(brightness - brightness_target) > 1e-3
        and fac_brightness_min != fac_brightness_max
    ):
        img_copy[...] = img
        fac_brightness = 0.5*(fac_brightness_min + fac_brightness_max)
        brightness = get_perceived_brightness(
            brighten_render3D(img_copy, fac_brightness, measure_rms=True)
        )
        if brightness < brightness_target:
            if fac_brightness_min == fac_brightness:
                break
            fac_brightness_min = fac_brightness
        elif brightness > brightness_target:
            if fac_brightness_max == fac_brightness:
                break
            fac_brightness_max = fac_brightness
    brighten_render3D(img, fac_brightness)

# Function for brightening an image by a certain factor
@cython.pheader(
    # Arguments
    img='float[:, :, ::1]',
    fac='float',
    measure_rms='bint',
    # Locals
    b_rms='float',
    c='Py_ssize_t',
    g_rms='float',
    i='Py_ssize_t',
    j='Py_ssize_t',
    n_opaque='Py_ssize_t',
    r_rms='float',
    α='float',
    returns=tuple,
)
def brighten_render3D(img, fac, measure_rms=False):
    """Brightens the passed image in-place by multiplying r, g and b
    (not α) by the passed factor. When measure_rms is True,
    the root-mean-square of each colour after the brightening
    will be computed and returned.
    """
    r_rms = 0
    g_rms = 0
    b_rms = 0
    n_opaque = 0
    for     i in range(ℤ[img.shape[0]]):
        for j in range(ℤ[img.shape[1]]):
            for c in range(3):
                img[i, j, c] *= fac
            with unswitch:
                if measure_rms:
                    α = img[i, j, 3]
                    if α > 0:
                        n_opaque += 1
                        r_rms += img[i, j, 0]**2
                        g_rms += img[i, j, 1]**2
                        b_rms += img[i, j, 2]**2
    if measure_rms:
        r_rms = sqrt(r_rms/n_opaque)
        g_rms = sqrt(g_rms/n_opaque)
        b_rms = sqrt(b_rms/n_opaque)
    return r_rms, g_rms, b_rms

# Function for normalizing α of an image such that max(α) = 1
@cython.header(
    # Arguments
    img='float[:, :, ::1]',
    # Locals
    i='Py_ssize_t',
    j='Py_ssize_t',
    α_max='float',
    α_max_inv='float',
    returns='void',
)
def normalize_α_render3D(img):
    α_max = np.max(img[:, :, 3])
    if α_max in (0, 1):
        return
    α_max_inv = 1/α_max
    for     i in range(ℤ[img.shape[0]]):
        for j in range(ℤ[img.shape[1]]):
            img[i, j, 3] *= α_max_inv

# Function for blending together two images
@cython.header(
    # Arguments
    img0='float[:, :, ::1]',
    img1='float[:, :, ::1]',
    mode=str,
    # Locals
    c='Py_ssize_t',
    i0='Py_ssize_t',
    i1='Py_ssize_t',
    j0='Py_ssize_t',
    j1='Py_ssize_t',
    α='float',
    α_inv='float',
    α0='float',
    α0_blend='float',
    α1='float',
    α1_blend='float',
    returns='void',
)
def blend_render3D(img0, img1, mode):
    """This function will combine two images using alpha blending.
    The first of the two passed images will be updated in-place.
    Note that rgbα overflow (> 1) is ignored.
    The blending modes implemented are 'screen', 'over', 'under',
    and 'overunder', where 'under' is just 'over' with the images
    switched and 'overunder' gives the average of 'over' and 'under'.
    While 'screen' and 'overunder' are symmetric with respect to the
    two images, this is not so for 'over'/'under'.
    For the second image you may pass just a single rgbα value, which is
    then equivalent to passing an image with the same shape as the
    first image, with the same rgbα value present throughout.
    """
    if mode not in ('screen', 'over', 'under', 'overunder'):
        abort(f'blend_render3D() got mode = "{mode}" ∉ {{"screen", "over", "under", "overunder"}}')
    # Indices into img1 in case it is just a single rgbα value
    i1 = j1 = 0
    # Blend img1 into img0
    for     i0 in range(ℤ[img0.shape[0]]):
        for j0 in range(ℤ[img0.shape[1]]):
            # Set proper indices into img1
            with unswitch:
                if img1.shape[0] > 1:
                    i1 = i0
                    j1 = j0
            # Compute combined α value
            α0 = img0[i0, j0, 3]
            α1 = img1[i1, j1, 3]
            α = α0 + α1 - α0*α1
            # The individual α values of the two images are used as
            # colour weights. As set below, the blending
            # corresponds to 'screen'.
            α0_blend = α0
            α1_blend = α1
            # Alter blending if not 'screen'
            with unswitch:
                if mode == 'over':
                    α1_blend *= 1 - α0
                elif mode == 'under':
                    α0_blend *= 1 - α1
                elif mode == 'overunder':
                    α0_blend *= 1 - 0.5*α1
                    α1_blend *= 1 - 0.5*α0
            # Blend this pixel
            α_inv = 1/(α + machine_ϵ_32)
            for c in range(3):
                img0[i0, j0, c] = α_inv*(
                    + img0[i0, j0, c]*α0_blend
                    + img1[i1, j1, c]*α1_blend
                )
            img0[i0, j0, 3] = α

# Function ensuring that rgbα values of an image
# stay within the legal range.
@cython.header(
    # Arguments
    img='float[:, :, ::1]',
    # Locals
    img_ptr='float*',
    index='Py_ssize_t',
    value='float',
    returns='void',
)
def truncate_saturated_render3D(img):
    img_ptr = cython.address(img[:, :, :])
    for index in range(img.shape[0]*img.shape[1]*img.shape[2]):
        value = img_ptr[index]
        if value > 1:
            img_ptr[index] = 1
        elif value < 0:
            img_ptr[index] = 0

# Function for rescaling an image
@cython.header(
    # Arguments
    img='float[:, :, ::1]',
    resolution='Py_ssize_t',
    # Locals
    img_ptr='float*',
    index='Py_ssize_t',
    returns='float[:, :, ::1]',
)
def resize_render3D(img, resolution):
    if img.shape[0] == resolution:
        return img
    # Fetch the PIL.Image module from the pillow library.
    # We can get this directly off of Matplotlib.
    Image = get_matplotlib().colors.Image
    # Transform value interval from [0, 1] to [0, 256)
    img_ptr = cython.address(img[:, :, :])
    for index in range(img.shape[0]*img.shape[1]*img.shape[2]):
        img_ptr[index] *= 256*(1 - machine_ϵ_32)
    # Use the pillow image library to carry out the rescaling.
    # Specifically, use the Lanczos method for the resampling,
    # minimizing Moiré patterns when downsampling.
    img = asarray(
        Image.fromarray(
            asarray(img, dtype=np.uint8)
        ).resize((resolution, resolution), resample=Image.Resampling.LANCZOS),
        dtype=C2np['float'],
    )
    # Transform value interval back from [0, 255] to [0, 1]
    img_ptr = cython.address(img[:, :, :])
    for index in range(img.shape[0]*img.shape[1]*img.shape[2]):
        img_ptr[index] *= 1.0/255.0
    return img

# Function for saving an already computed 3D render
@cython.header(
    # Arguments
    declaration=object,  # Render3DDeclaration,
    img='float[:, :, ::1]',
    filename=str,
    n_dumps='int',
    # Locals
    returns='void',
)
def save_render3D(declaration, img, filename, n_dumps):
    if not master:
        return
    plt = get_matplotlib().pyplot
    # Set filename extension to png
    if not filename.endswith('.png'):
        filename += '.png'
    # The filename should reflect the components
    # if multiple renders are to be dumped.
    if n_dumps > 1:
        filename = augment_filename(
            filename,
            '_'.join([component.name.replace(' ', '-') for component in declaration.components]),
            '.png',
        )
    # Make sure that the rgbα values stay within the legal range
    # Save image to disk
    masterprint(f'Saving image to "{filename}" ...')
    plt.imsave(filename, asarray(img))
    masterprint('done')

# Get local domain information
domain_info = get_domain_info()
cython.declare(
    domain_layout_local_indices='int[::1]',
    domain_bgn_x='double',
    domain_bgn_y='double',
    domain_bgn_z='double',
)
domain_layout_local_indices = domain_info.layout_local_indices
domain_bgn_x                = domain_info.bgn_x
domain_bgn_y                = domain_info.bgn_y
domain_bgn_z                = domain_info.bgn_z
