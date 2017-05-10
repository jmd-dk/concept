# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2017 Jeppe Mosgaard Dakin.
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
        )

# Pure Python imports
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import juggle_axes



# Function for plotting the power spectrum
# and saving the figure to filename.
@cython.header(# Arguments
               data_list='list',
               filename='str',
               power_dict='object',  # OrderedDict
               # Locals
               a_string='str',
               filename_component='str',
               i='Py_ssize_t',
               k='double[::1]',
               kmax='double',
               kmin='double',
               maxpowermax='double',
               power='double[::1]',
               power_index='Py_ssize_t',
               power_indices='list',
               power_σ='double[::1]',
               powermin='double',
               t_string='str',
               tmp='str',
               name='str',
               names='list',
               )
def plot_powerspec(data_list, filename, power_dict):
    """This function will do separate power spectrum
    plots for each component.
    The power spectra are given in dat_list, which are a list with
    the following content: [k, power, power_σ, power, power_σ, ...]
    where a pair of power and power_σ is for one component.
    The power_dict is an ordered dict and hold the component names
    for the power spectra in data_list, in the correct order.
    """
    # Only the master process takes part in the power spectra plotting
    if not master:
        return
    # Do not plot any power spectra if
    # powerspec_plot_select does not contain any True values.
    if not any(powerspec_plot_select.values()):
        return
    # Attach missing extension to filename
    if not filename.endswith('.png'):
        filename += '.png'
    # Switch to the powerspec figure
    plt.figure('powerspec')
    # Extract k values, common to all power spectra
    k = data_list[0]
    kmin = k[0]
    kmax = k[k.shape[0] - 1]
    # Get relevant indices and component name from power_dict
    power_indices = []
    names = []
    for i, name in enumerate(power_dict.keys()):
        # The power spectrum of the i'th component should only be
        # plotted if {name: True} or {'all': True} exist
        # in powerspec_plot_select. Also, if name exists,
        # the value for 'all' is ignored.
        if name.lower() in powerspec_plot_select:
            if not powerspec_plot_select[name.lower()]:
                continue
        elif not powerspec_plot_select.get('all', False):
            continue
        # The i'th power spectrum should be plotted
        power_indices.append(1 + 2*i)
        names.append(name)
    # Plot the power spectrum for each component separately
    for power_index, name in zip(power_indices, names):
        # The filename should reflect the individual component names,
        # when several components are being plotted.
        filename_component = filename
        if len(names) > 1:
            if '_t=' in filename:
                filename_component = filename.replace('_t=',
                                                      '_{}_t='.format(name.replace(' ', '-')))
            elif '_a=' in filename:
                filename_component = filename.replace('_a=',
                                                      '_{}_a='.format(name.replace(' ', '-')))
            else:
                filename_component = filename.replace('.png',
                                                      '_{}.png'.format(name.replace(' ', '-')))
        # The filename should reflect the individual component names
        masterprint('Plotting power spectrum of {} and saving to "{}" ...'
                    .format(name, filename_component))
        # Extract the power and its standard
        # deviation for the i'th component.
        power   = data_list[power_index]
        power_σ = data_list[power_index + 1]
        powermin = min(power)
        maxpowermax = np.max(asarray(power) + asarray(power_σ))
        # Plot power spectrum
        plt.figure()
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log', nonposy='clip')
        plt.errorbar(k, power, yerr=power_σ,
                     fmt='.',
                     markersize=3,
                     ecolor='C1',
                     capsize=2,
                     )
        tmp = '{:.1e}'.format(kmin)
        plt.xlim(xmin=float(tmp[0] + tmp[3:]))
        tmp = '{:.1e}'.format(kmax)
        plt.xlim(xmax=float(str(int(tmp[0]) + 1) + tmp[3:]))
        tmp = '{:.1e}'.format(powermin)
        plt.ylim(ymin=float(tmp[0] + tmp[3:]))
        tmp = '{:.1e}'.format(maxpowermax)
        plt.ylim(ymax=float(str(int(tmp[0]) + 1) + tmp[3:]))
        plt.xlabel('$k$ $\mathrm{{[{}^{{-1}}]}}$'.format(unit_length), fontsize=14)
        plt.ylabel('power $\mathrm{{[{}^3]}}$'.format(unit_length),    fontsize=14)
        t_string = '$t = {}\, \mathrm{{{}}}$'.format(significant_figures(universals.t,
                                                                         4,
                                                                         fmt='TeX',
                                                                         ),
                                                     unit_time)
        a_string = ', $a = {}$'.format(significant_figures(universals.a, 
                                                           4,
                                                           fmt='TeX',
                                                           )
                                       ) if enable_Hubble else ''
        plt.title('{} at {}{}'.format(name, t_string, a_string), fontsize=16)
        plt.gca().tick_params(labelsize=13)
        plt.tight_layout()
        plt.savefig(filename_component)
        # Close the figure, leaving no trace in memory of the plot
        plt.close()
        # Finish progress message
        masterprint('done')

# Function for 3D renderings of the components
@cython.header(# Arguments
               components='list',
               filename='str',
               cleanup='bint',
               tmp_dirname='str',
               # Locals
               N='Py_ssize_t',
               N_local='Py_ssize_t',
               a_str='str',
               alpha='double',
               alpha_min='double',
               artists_text='dict',
               color='double[::1]',
               combined='double[:, :, ::1]',
               component='Component',
               figname='str',
               filename_component='str',
               filename_component_alpha='str',
               filename_component_alpha_part='str',
               filenames_component_alpha='list',
               filenames_component_alpha_part='list',
               i='Py_ssize_t',
               index='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               label_props='list',
               label_spacing='double',
               name='str',
               names='tuple',
               part='int',
               posx_mv='double[::1]',
               posy_mv='double[::1]',
               posz_mv='double[::1]',
               render_dir='str',
               rgba='double[:, ::1]',
               scatter_size='double',
               size_nopseudo_noghost='Py_ssize_t',
               size='Py_ssize_t',
               size_i='Py_ssize_t',
               size_j='Py_ssize_t',
               size_k='Py_ssize_t',
               t_str='str',
               x='double*',
               x_mv='double[::1]',
               xi='double',
               y='double*',
               y_mv='double[::1]',
               yj='double',
               z='double*',
               z_mv='double[::1]',
               zk='double',
               Σmass='double',
               ϱ_noghosts='double[:, :, :]',
               ϱbar_component='double',
               )
def render(components, filename, cleanup=True, tmp_dirname='.renders'):
    global render_image
    # Do not render anything if
    # render_select does not contain any True values.
    if not any(render_select.values()):
        return
    # Attach missing extension to filename
    if not filename.endswith('.png'):
        filename += '.png'
    # The directory for storing the temporary renders
    render_dir = '{}/{}'.format(os.path.dirname(filename), tmp_dirname)
    # Initialize figures by building up render_dict, if this is the
    # first time this function is called.
    if not render_dict:
        # Make cyclic default colors as when doing multiple plots in
        # one figure. Make sure that none of the colors are identical
        # to the background color.
        default_colors = itertools.cycle([to_rgb(prop['color'])
                                          for prop in matplotlib.rcParams['axes.prop_cycle']
                                          if not all(to_rgb(prop['color']) == bgcolor)])
        for component in components:
            # The i'th component should only be rendered if
            # {name: True} or {'all': True} exist in render_select.
            # Also, if name exists, the value for 'all' is ignored.
            if component.name.lower() in render_select:
                if not render_select[component.name.lower()]:
                    continue
            elif not render_select.get('all', False):
                continue
            # This component should be rendered.
            # Prepare a figure for the render of the i'th component.
            figname = 'render_{}'.format(component.name)
            dpi = 100  # The value of dpi is irrelevant
            fig = plt.figure(figname,
                             figsize=[resolution/dpi]*2,
                             dpi=dpi)
            ax = fig.gca(projection='3d', facecolor=bgcolor)
            # The color of this component
            if component.name.lower() in render_colors:
                # This component is given a specific color by the user
                color = render_colors[component.name.lower()]
            elif 'all' in render_colors:
                # All components are given the same color by the user
                color = render_colors['all']
            else:
                # No color specified for this particular component.
                # Assign the next color from the default cyclic colors.
                color = next(default_colors)
            # The artist for the component
            if component.representation == 'particles':
                artist_component = ax.scatter(0, 0, 0, c=color, depthshade=False, lw=0)
            elif component.representation == 'fluid':
                N = np.prod(-2 + asarray(component.shape) - 1 - 2)
                rgba = np.empty((N, 4), dtype=C2np['double'])
                for i in range(N):
                    for dim in range(3):
                        rgba[i, dim] = color[dim]
                    rgba[i, 3] = 0
                artist_component = ax.scatter([0]*N, [0]*N, [0]*N, c=rgba, depthshade=False, lw=0)
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
            plt.tight_layout(pad=-1)  # Extra tight layout, to prevent white frame
            proj3d.persp_transformation = orthographic_proj  # Use orthographic 3D projection
            # Store the figure, axes and the component
            # and text artists in the render_dict.
            render_dict[component.name] = {'fig': fig,
                                           'ax': ax,
                                           'artist_component': artist_component,
                                           'artists_text': artists_text,
                                           }
            # Explicit arrays of positions are needed
            # also for fluid components.
            if component.representation == 'fluid':
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
                # Place the grids in the render_dict
                render_dict[component.name]['posx_mv'] = posx_mv
                render_dict[component.name]['posy_mv'] = posy_mv
                render_dict[component.name]['posz_mv'] = posz_mv
        # Return if no component is to be rendered
        if not render_dict:
            return
        # Create the temporary render directory if necessary
        if not (nprocs == 1 == len(render_dict)):
            if master:
                os.makedirs(render_dir, exist_ok=True)
            Barrier()
    # Print out progress message
    names = tuple(render_dict.keys())
    if len(names) == 1:
        masterprint('Rendering {} and saving to "{}" ...'.format(names[0], filename))
    else:
        masterprint('Rendering:')
        for name in names:
            filename_component = filename
            if '_t=' in filename:
                filename_component = filename.replace('_t=', '_{}_t='.format(name))
            elif '_a=' in filename:
                filename_component = filename.replace('_a=', '_{}_a='.format(name))
            else:
                filename_component = filename.replace('.png', '_{}.png'.format(name))
            masterprint('Rendering {} and saving to "{}"'.format(name, filename_component),
                        indent=4)
        masterprint('...', indent=4)
    # Render each component separately
    for component in components:
        if component.name not in render_dict:
            continue
        # Switch to the render figure
        figname = 'render_{}'.format(component.name)
        plt.figure(figname)
        # Extract figure elements
        fig = render_dict[component.name]['fig']
        ax = render_dict[component.name]['ax']
        artist_component = render_dict[component.name]['artist_component']
        artists_text = render_dict[component.name]['artists_text']
        if component.representation == 'particles':
            # Extract particle meta data
            N = component.N
            N_local = component.N_local
            # Update particle positions on the figure
            artist_component._offsets3d = juggle_axes(component.posx_mv[:N_local],
                                                      component.posy_mv[:N_local],
                                                      component.posz_mv[:N_local],
                                                      zdir='z')
            # The particle size on the figure.
            # The size is chosen such that the particles stand side
            # by side in a homogeneous universe (more or less).
            scatter_size = 1000*np.prod(fig.get_size_inches())/N**ℝ[2/3]
            # The particle alpha on the figure.
            # The alpha is chosen such that in a homogeneous universe,
            # a column of particles have a collective alpha
            # of 1 (more or less).
            alpha = 1/cbrt(N)
            # Alpha values lower than alpha_min appear completely
            # invisible. Allow no alpha values lower than alpha_min. 
            # Shrink the size to make up for the larger alpha.
            alpha_min = 0.0059
            if alpha < alpha_min:
                scatter_size *= alpha/alpha_min
                alpha = alpha_min
            # Apply size and alpha
            artist_component.set_sizes([scatter_size])
            artist_component.set_alpha(alpha)
        elif component.representation == 'fluid':
            # Extract the ϱ grid
            ϱ_noghosts = component.ϱ.grid_noghosts
            # The particle (fluid element) size on the figure.
            # The size is chosen such that the particles stand side
            # by side in a homogeneous universe (more or less).
            N = component.gridsize**3  # Number of fluid elements
            scatter_size = 1000*np.prod(fig.get_size_inches())/N**ℝ[2/3]
            # Grab the color and alpha array from the last artist
            rgba = artist_component.get_facecolor()
            # Multiplication factor for alpha values. An alpha value of
            # 1 is assigned if the relative overdensity
            # ϱ(component)/ϱbar(component) >= 1/alpha_fac.
            # The chosen alpha_fac is such that in a homogeneous
            # universe, a column of fluid elements have a collective
            # alpha of 1 (more or less).            
            alpha_fac = 1/cbrt(N)
            # Alpha values lower than alpha_min appear completely
            # invisible. Do not allow alpha_fac (the mean alpha) to be
            # lower than alpha_min.
            # Shrink the size to make up for the larger alpha.
            alpha_min = 0.0059
            if alpha_fac < alpha_min:
                scatter_size *= alpha_fac/alpha_min
                alpha_fac = alpha_min
            # Update the alpha values in rgba array. The rgb-values
            # remain the same for all renders of this component.
            Σmass = universals.a**(-3*component.w())*component.Σmass_present
            ϱbar_component = Σmass/boxsize**3
            index = 0
            for         i in range(ℤ[ϱ_noghosts.shape[0] - 1]):
                for     j in range(ℤ[ϱ_noghosts.shape[1] - 1]):
                    for k in range(ℤ[ϱ_noghosts.shape[2] - 1]):
                        alpha = ℝ[alpha_fac/ϱbar_component]*ϱ_noghosts[i, j, k]
                        if alpha > 1:
                            alpha = 1
                        rgba[index, 3] = alpha
                        index += 1
            # The previous scatter artist cannot be re-used due to a bug
            # in matplotlib (the colors/alphas cannot be updated).
            # Get rid of the old scatter plot.
            artist_component._offsets3d = juggle_axes([-boxsize], [-boxsize], [-boxsize], zdir='z')
            # Create new scatter plot and stick it into the render_dict
            artist_component = ax.scatter(render_dict[component.name]['posx_mv'],
                                          render_dict[component.name]['posy_mv'],
                                          render_dict[component.name]['posz_mv'],
                                          c=rgba,
                                          s=scatter_size,
                                          depthshade=False,
                                          lw=0,
                                          )
            render_dict[component.name]['artist_component'] = artist_component
        # Print the current cosmic time and scale factor on the figure
        if master:
            t_str = a_str = ''
            t_str = '$t = {}\, \mathrm{{{}}}$'.format(significant_figures(universals.t, 4, 'TeX'),
                                                      unit_time)
            artists_text['t'].set_text(t_str)
            if enable_Hubble:
                a_str = '$a = {}$'.format(significant_figures(universals.a, 4, 'TeX'))
                artists_text['a'].set_text(a_str)
            # Make the text color black or white,
            # dependent on the bgcolor
            for artist_text in artists_text.values():
                if sum(bgcolor) < 1:
                    artist_text.set_color('white')
                else:
                    artist_text.set_color('black')
        # Save the render
        if nprocs == 1:
            filename_component_alpha_part = ('{}/{}_alpha.png'
                                              .format(render_dir,
                                                      component.name.replace(' ', '-')))
        else:
            filename_component_alpha_part = ('{}/{}_alpha_{}.png'
                                             .format(render_dir,
                                                     component.name.replace(' ', '-'),
                                                     rank))
        if nprocs == 1 == len(render_dict):
            # As this is the only render which should be done, it can
            # be saved directly in its final, non-transparent state.
            plt.savefig(filename, transparent=False)
            masterprint('done')
        else:
            # Save transparent render
            plt.savefig(filename_component_alpha_part, transparent=True)
    # All rendering done
    Barrier()
    # The partial renders will now be combined into full renders,
    # stored in the 'render_image', variable. Partial renders of the
    # j'th component will be handled by the process with rank j.
    if not (nprocs == 1 == len(render_dict)):
        # Loop over components designated to each process
        for i in range(1 + len(render_dict)//nprocs):
            # Break out when there is no more work for this process
            j = rank + nprocs*i
            if j >= len(names):
                break
            name = names[j].replace(' ', '-')
            if nprocs == 1:
                # Simply load the already fully constructed image
                filename_component_alpha = '{}/{}_alpha.png'.format(render_dir, name)
                render_image = plt.imread(filename_component_alpha)
            else:
                # Create list of filenames for the partial renders
                filenames_component_alpha_part = ['{}/{}_alpha_{}.png'.format(render_dir,
                                                                              name,
                                                                              part)
                                             for part in range(nprocs)]
                # Read in the partial renders and blend
                # them together into the render_image variable.
                blend(filenames_component_alpha_part)
                # Save combined render of the j'th component
                # with transparency. Theese are then later combined into
                # a render containing all components.
                if len(names) > 1:
                    filename_component_alpha = '{}/{}_alpha.png'.format(render_dir, name)
                    plt.imsave(filename_component_alpha, render_image)
            # Add opaque background to render_image
            add_background()
            # Save combined render of the j'th component
            # without transparency.
            filename_component = filename
            if len(names) > 1:
                if '_t=' in filename:
                    filename_component = filename.replace('_t=', '_{}_t='.format(name))
                elif '_a=' in filename:
                    filename_component = filename.replace('_a=', '_{}_a='.format(name))
                else:
                    filename_component = filename.replace('.png', '_{}.png'.format(name))
            plt.imsave(filename_component, render_image)
        Barrier()
        masterprint('done')
        # Finally, combine the full renders of individual components
        # into a total render containing all components.
        if master and len(names) > 1:
            masterprint('Combining component renders and saving to "{}" ...'.format(filename))
            filenames_component_alpha = ['{}/{}_alpha.png'.format(render_dir,
                                                                  name.replace(' ', '-'))
                                         for name in names]
            blend(filenames_component_alpha)
            # Add opaque background to render_image and save it
            add_background()
            plt.imsave(filename, render_image)
            masterprint('done')
    # Remove the temporary directory, if cleanup is requested
    if master and cleanup and not (nprocs == 1 == len(render_dict)):
        shutil.rmtree(render_dir)
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
# together into the global render_image array.
@cython.header(# Arguments
               filenames='list',
               # Locals
               alpha_A='float',
               alpha_B='float',
               alpha_tot='float',
               i='int',
               j='int',
               rgb='int',
               rgba='int',
               tmp_image='float[:, :, ::1]',
               transparency='float',
               )
def blend(filenames):
    # Make render_image black and transparent
    render_image[...] = 0
    for filename in filenames:
        tmp_image = plt.imread(filename)
        for i in range(resolution):
            for j in range(resolution):
                # Pixels with 0 alpha has (r, g, b) = (1, 1, 1)
                # (this is a defect of plt.savefig).
                # These should be disregarded completely.
                alpha_A = tmp_image[i, j, 3]
                if alpha_A != 0:
                    # Combine render_image with tmp_image by
                    # adding them together, using their alpha values
                    # as weights.
                    alpha_B = render_image[i, j, 3]
                    alpha_tot = alpha_A + alpha_B - alpha_A*alpha_B
                    for rgb in range(3):
                        render_image[i, j, rgb] = ((alpha_A*tmp_image[i, j, rgb]
                                                    + alpha_B*render_image[i, j, rgb])/alpha_tot)
                    render_image[i, j, 3] = alpha_tot
    # Some pixel values in the combined render may have overflown.
    # Clip at saturation value.
    for i in range(resolution):
        for j in range(resolution):
            for rgba in range(4):
                if render_image[i, j, rgba] > 1:
                    render_image[i, j, rgba] = 1

# Add background color to render_image
@cython.header(# Arguments
               filenames='list',
               # Locals
               alpha='float',
               i='int',
               j='int',
               rgb='int',
               )
def add_background():
    for i in range(resolution):
        for j in range(resolution):
            alpha = render_image[i, j, 3]
            # Add background using "A over B" alpha blending
            for rgb in range(3):
                render_image[i, j, rgb] = (alpha*render_image[i, j, rgb]
                                           + (1 - alpha)*bgcolor[rgb])
                render_image[i, j, 3] = 1

# This function projects the particle/fluid element positions onto the
# xy-plane and renders this projection directly in the terminal,
# using ANSI/VT100 control sequences.
@cython.header(# Arguments
               components='list',
               # Locals
               N='Py_ssize_t',
               N_local='Py_ssize_t',
               Vcell='double',
               colornumber='int',
               colornumber_offset='int',
               gridsize='Py_ssize_t',
               i='Py_ssize_t',
               index_x='Py_ssize_t',
               index_y='Py_ssize_t',
               j='Py_ssize_t',
               mass='double',
               maxoverdensity='double',
               maxval='double',
               component='Component',
               posx='double*',
               posy='double*',
               projection_ANSI='list',
               scalec='double',
               size_x='Py_ssize_t',
               size_y='Py_ssize_t',
               size_z='Py_ssize_t',
               total_mass='double',
               Σmass='double',
               ϱ_noghosts='double[:, :, :]',
               )
def terminal_render(components):
    # Project all particle positions onto the 2D projection array,
    # counting the number of particles in each pixel.
    # The projection is done onto the xy-plane.
    projection[...] = 0
    total_mass = 0
    for component in components:
        if component.representation == 'particles':
            # Extract relevant particle data
            N = component.N
            N_local = component.N_local
            mass = component.mass
            posx = component.posx
            posy = component.posy
            # Update the total mass
            total_mass += N*mass
            # Do the projection. Each particle is weighted by its mass.
            for i in range(N_local):
                index_x = int(posx[i]*projection_scalex)
                index_y = int(posy[i]*projection_scaley)
                projection[index_y, index_x] += mass
        elif component.representation == 'fluid':
            # Extract relevant fluid data
            gridsize = component.gridsize
            Σmass = universals.a**(-3*component.w())*component.Σmass_present
            mass = Σmass/component.gridsize**3
            Vcell = (boxsize/gridsize)**3
            ϱ_noghosts = component.ϱ.grid_noghosts
            size_x = ϱ_noghosts.shape[0] - 1
            size_y = ϱ_noghosts.shape[1] - 1
            size_z = ϱ_noghosts.shape[2] - 1
            # Update the total mass
            total_mass += gridsize**3*mass
            # Do the projection.
            # Each fluid element is weighted by its mass.
            for i in range(size_x):
                index_x = int((domain_start_x + i*domain_size_x/size_x)*projection_scalex)
                for j in range(size_y):
                    index_y = int((domain_start_y + j*domain_size_y/size_y)*projection_scaley)
                    projection[index_y, index_x] += Vcell*np.sum(ϱ_noghosts[i, j, :size_z])
    # Sum up local projections into the master process
    Reduce(sendbuf=(MPI.IN_PLACE if master else projection),
           recvbuf=(projection   if master else None),
           op=MPI.SUM)
    if not master:
        return
    # Values in the projection array equal to or larger than maxval
    # will be mapped to color nr. 255. The value of the maxoverdensity
    # is arbitrarily chosen.
    maxoverdensity = 10
    maxval = maxoverdensity*total_mass/(projection.shape[0]*projection.shape[1])
    if maxval < 5:
        maxval = 5
    # Construct list of strings, each string being a space prepended
    # with an ANSI/VT100 control sequences which sets the background
    # color. When printed together, these strings produce an ANSI image
    # of the projection.
    projection_ANSI = []
    scalec = terminal_render_colormap_rgb.shape[0]/maxval
    colornumber_offset = 256 - terminal_render_colormap_rgb.shape[0]
    for i in range(projection.shape[0]):
        for j in range(projection.shape[1]):
            colornumber = int(colornumber_offset + projection[i, j]*scalec)
            if colornumber > 255:
                colornumber = 255
            projection_ANSI.append('\x1b[48;5;{}m '.format(colornumber))
        projection_ANSI.append('\x1b[0m\n')
    # Print the ANSI image
    masterprint(''.join(projection_ANSI), end='', wrap=False)



# Declare global variables used in above functions
cython.declare(render_dict='object',  # OrderedDict
               render_image='float[:, :, ::1]',
               )
# Prepare a figure for the render
if any(render_times.values()) or special_params.get('special', '') == 'render':
    # (Ordered) dictionary containing the figure, axes, component
    # artist and text artist for each component.
    render_dict = collections.OrderedDict()
# The array storing the render
render_image = empty((resolution, resolution, 4), dtype=C2np['float'])

# Prepare a figure for the powerspec plot
if (any(powerspec_plot_select.values())
    and (powerspec_times or special_params.get('special', '') == 'powerspec')):
    fig_powerspec = plt.figure('powerspec')
# The array storing the terminal render and the color map
if any(terminal_render_times.values()):
    # Allocate the 2D projection array storing the terminal render
    cython.declare(projection='double[:, ::1]',
                   projection_scalex='double',
                   projection_scaley='double',
                   terminal_render_colormap_rgb='double[:, ::1]',
                   )
    projection = np.empty((terminal_render_resolution//2, terminal_render_resolution),
                          dtype=C2np['double'])
    projection_scalex = projection.shape[1]/boxsize
    projection_scaley = projection.shape[0]/boxsize
    # Construct terminal colormap with 256 - 16 - 2 = 238 colors
    # and apply it to the terminal, remapping the 238 higher color
    # numbers. The 16 + 2 = 18 lowest are left alone in order not to
    # mess with standard terminal coloring and the colors used for the
    # CO𝘕CEPT logo at startup.
    if master:
        terminal_render_colormap_rgb = np.ascontiguousarray(getattr(matplotlib.cm,
                                                                    terminal_render_colormap)
                                                            (linspace(0, 1, 238))[:, :3])
        for i, rgb in enumerate(asarray(terminal_render_colormap_rgb)):
            colorhex = matplotlib.colors.rgb2hex(rgb)
            masterprint('\x1b]4;{};rgb:{}/{}/{}\x1b\\'
                         .format(256 - terminal_render_colormap_rgb.shape[0] + i, colorhex[1:3],
                                                                                  colorhex[3:5],
                                                                                  colorhex[5:]),
                        end='')
