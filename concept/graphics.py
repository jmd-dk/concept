# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2016 Jeppe Mosgaard Dakin.
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
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Pure Python imports
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from mpl_toolkits.mplot3d.art3d import juggle_axes
import pexpect
import subprocess



# Function for plotting the power spectrum
# and saving the figure to filename.
@cython.header(# Arguments
               data_list='list',
               a='double',
               filename='str',
               power_dict='object',  # OrderedDict
               # Locals
               filename_type='str',
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
               tmp='str',
               typename='str',
               typenames='list',
               )
def plot_powerspec(data_list, a, filename, power_dict):
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
    # Get relevant indices and particle types from power_dict
    power_indices = []
    typenames = []
    for i, typename in enumerate(power_dict.keys()):
        # The power spectrum of the i'th particles should only be
        # plotted if {typename: True} or {'all': True} exist
        # in powerspec_plot_select. Also, if typename exists,
        # the value for 'all' is ignored.
        if typename.lower() in powerspec_plot_select:
            if not powerspec_plot_select[typename.lower()]:
                continue
        elif not powerspec_plot_select.get('all', False):
            continue
        # The i'th power spectrum should be plotted
        power_indices.append(1 + 2*i)
        typenames.append(typename.replace(' ', '-'))
    # Plot the power spectrum for each particle type separately
    for power_index, typename in zip(power_indices, typenames):
        # The filename should reflect the individual particle types,
        # when several particle types are being plotted.
        filename_type = filename
        if len(typenames) > 1:
            if '_a=' in filename:
                filename_type = filename.replace('_a=', '_{}_a='.format(typename))
            else:
                filename_type = filename.replace('.png', '_{}.png'.format(typename))
        # The filename should reflect the individual particle types
        masterprint('Plotting power spectrum of {} and saving to "{}" ...'.format(typename,
                                                                                  filename_type))
        # Extract the power and its standard
        # deviation for the i'th particle type.
        power   = data_list[power_index]
        power_σ = data_list[power_index + 1]
        powermin = min(power)
        maxpowermax = np.max(asarray(power) + asarray(power_σ))
        # Clear the figure
        plt.clf()
        # Plot powerspectrum
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log', nonposy='clip')
        plt.errorbar(k, power, yerr=power_σ, fmt='b.', ms=3, ecolor='r', lw=0)
        tmp = '{:.1e}'.format(kmin)
        plt.xlim(xmin=float(tmp[0] + tmp[3:]))
        tmp = '{:.1e}'.format(kmax)
        plt.xlim(xmax=float(str(int(tmp[0]) + 1) + tmp[3:]))
        tmp = '{:.1e}'.format(powermin)
        plt.ylim(ymin=float(tmp[0] + tmp[3:]))
        tmp = '{:.1e}'.format(maxpowermax)
        plt.ylim(ymax=float(str(int(tmp[0]) + 1) + tmp[3:]))
        plt.xlabel('$k$ $\mathrm{[' + base_length + '^{-1}]}$',
                   fontsize=14,
                   )
        plt.ylabel('power $\mathrm{[' + base_length + '^3]}$',
                   fontsize=14,
                   )
        plt.title('{} at $a = {}$'.format(typename,
                                          significant_figures(a, 4, fmt='TeX')),
                  fontsize=16,
                  )
        plt.gca().tick_params(labelsize=13)
        plt.tight_layout()
        plt.savefig(filename_type)
        # Finish progress message
        masterprint('done')

# Setting up figure and plot the particles
@cython.header(# Arguments
               particles_list='list',
               a='double',
               filename='str',
               cleanup='bint',
               # Locals
               N='Py_ssize_t',
               N_local='Py_ssize_t',
               a_str='str',
               alpha='double',
               alpha_min='double',
               color='double[::1]',
               combined='double[:, :, ::1]',
               dirname='str',
               figname='str',
               filename_type='str',
               filename_type_alpha='str',
               filename_type_alpha_part='str',
               filenames_type_alpha='list',
               filenames_type_alpha_part='list',
               i='int',
               j='int',
               part='int',
               particle_type='str',
               particle_types='tuple',
               particles='Particles',
               render_dirname='str',
               rgb='int',
               size='double',
               tmp_image='float[:, :, ::1]',
               )
def render(particles_list, a, filename, cleanup=True):
    global render_dict, render_image
    # Do not render anything if
    # render_select does not contain any True values.
    if not any(render_select.values()):
        return
    # Attach missing extension to filename
    if not filename.endswith('.png'):
        filename += '.png'
    # The directory for storing the temporary renders
    dirname = os.path.dirname(filename)
    render_dirname = '{}/.renders'.format(dirname)
    # Initialize figures by building up render_dict, if this is the
    # first time this function is called.
    if not render_dict:
        # Make cyclic default colors as when doing multiple plots in
        # one figure. Make sure that none of the colors are identical
        # to the background color.
        try:
            # Matplotlib >= 1.5
            default_colors = itertools.cycle([to_rgb(prop['color'])
                                              for prop in matplotlib.rcParams['axes.prop_cycle']
                                              if not all(to_rgb(prop['color']) == bgcolor)])
        except:
            # Matplotlib <= 1.4
            default_colors = itertools.cycle([to_rgb(c)
                                              for c in matplotlib.rcParams['axes.color_cycle']
                                              if not all(to_rgb(c) == bgcolor)])
        for particles in particles_list:
            # The i'th particles should only be rendered if
            # {'type i': True} or {'all': True} exist in render_select.
            # Also, if 'type i' exists, the value for 'all' is ignored.
            if particles.type.lower() in render_select:
                if not render_select[particles.type.lower()]:
                    continue
            elif not render_select.get('all', False):
                continue
            # These particles should be rendered!
            # Prepare a figure for the render of the i'th particle type.
            # The 77.50 scaling is needed to
            # map the resolution to pixel units.
            figname = 'render_{}'.format(particles.type)
            fig = plt.figure(figname,
                             figsize=[resolution/77.50]*2)
            try:
                # Matplotlib 2.x
                ax = fig.gca(projection='3d', facecolor=bgcolor)
            except:
                # Matplotlib 1.x
                ax = fig.gca(projection='3d', axisbg=bgcolor)
            # The color of the particles
            if particles.type.lower() in render_colors:
                color = render_colors[particles.type.lower()]
            else:
                # No color specified for these particular particles.
                # Use default cyclic colors.
                color = next(default_colors)
            # The artist for the particles
            artist_particles = ax.scatter(0, 0, 0, c=color, lw=0)
            # The artist for the scalefactor text
            artist_text = ax.text(+0.25*boxsize,
                                  -0.30*boxsize,
                                  +0.00*boxsize,
                                  '',
                                  fontsize=16,
                                  )
            # Configure axis options
            ax.set_aspect('equal')
            ax.dist = 8.55  # Zoom level
            ax.set_xlim(0, boxsize)
            ax.set_ylim(0, boxsize)
            ax.set_zlim(0, boxsize)
            ax.w_xaxis.set_pane_color(zeros(4))
            ax.w_yaxis.set_pane_color(zeros(4))
            ax.w_zaxis.set_pane_color(zeros(4))
            ax.w_xaxis.gridlines.set_lw(0)
            ax.w_yaxis.gridlines.set_lw(0)
            ax.w_zaxis.gridlines.set_lw(0)
            ax.grid(False)
            ax.w_xaxis.line.set_visible(False)
            ax.w_yaxis.line.set_visible(False)
            ax.w_zaxis.line.set_visible(False)
            ax.w_xaxis.pane.set_visible(False)
            ax.w_yaxis.pane.set_visible(False)
            ax.w_zaxis.pane.set_visible(False)
            for tl in ax.w_xaxis.get_ticklines():
                tl.set_visible(False)
            for tl in ax.w_yaxis.get_ticklines():
                tl.set_visible(False)
            for tl in ax.w_zaxis.get_ticklines():
                tl.set_visible(False)
            for tl in ax.w_xaxis.get_ticklabels():
                tl.set_visible(False)
            for tl in ax.w_yaxis.get_ticklabels():
                tl.set_visible(False)
            for tl in ax.w_zaxis.get_ticklabels():
                tl.set_visible(False)
            # Store the figure, axes and the particles
            # and text artists in the render_dict.
            render_dict[particles.type] = {'fig': fig,
                                           'ax': ax,
                                           'artist_particles': artist_particles,
                                           'artist_text': artist_text,
                                           }
        # Return if no particles are to be rendered
        if not render_dict:
            return
        # Create the temporary render directory if necessary
        if not (nprocs == 1 == len(render_dict)):
            if master:
                os.makedirs(render_dirname, exist_ok=True)
            Barrier()
    # Render each particle type separately
    for particles in particles_list:
        if particles.type not in render_dict:
            continue
        masterprint('Rendering {} ...'.format(particles.type))
        # Switch to the render figure
        figname = 'render_{}'.format(particles.type)
        plt.figure(figname)
        # Extract figure elements
        fig = render_dict[particles.type]['fig']
        ax = render_dict[particles.type]['ax']
        artist_particles = render_dict[particles.type]['artist_particles']
        artist_text = render_dict[particles.type]['artist_text']
        # Extract particle data
        N = particles.N
        N_local = particles.N_local
        # Update particle positions on the figure
        artist_particles._offsets3d = juggle_axes(particles.posx_mv[:N_local],
                                                  particles.posy_mv[:N_local],
                                                  particles.posz_mv[:N_local],
                                                  zdir='z')
        # The particle size on the figure.
        # The size is chosen such that the particles stand side
        # by side in a homogeneous universe (more or less).
        size = 1000*np.prod(fig.get_size_inches())/N**ℝ[2/3]
        # The particle alpha on the figure.
        # The alpha is chosen such that in a homogeneous unvierse,
        # a column of particles have a collective alpha
        # of 1 (more or less).
        alpha = N**ℝ[-1/3]
        # Alpha values lower than alpha_min appear completely invisible.
        # Allow no alpha values lower than alpha_min. Shrink the size to
        # make up for the large alpha.
        alpha_min = 0.0059
        if alpha < alpha_min:
            size *= alpha/alpha_min
            alpha = alpha_min
        # Apply size and alpha
        artist_particles.set_sizes([size])
        artist_particles.set_alpha(alpha)
        # Print the current scale factor on the figure
        if master:
            a_str = '$a = {}$'.format(significant_figures(a, 4, 'TeX'))
            artist_text.set_text(a_str)
            # Make the text color black or white,
            # dependent on the bgcolor
            if sum(bgcolor) < 1:
                artist_text.set_color('white')
            else:
                artist_text.set_color('black')
        # Save the render
        if nprocs == 1:
            filename_type_alpha_part = '{}/{}_alpha.png'.format(render_dirname,
                                                                particles.type.replace(' ', '-'))
        else:
            filename_type_alpha_part = ('{}/{}_alpha_{}.png'
                                        .format(render_dirname,
                                                particles.type.replace(' ', '-'),
                                                rank))
        if nprocs == 1 == len(render_dict):
            # As this is the only render which should be done, it can
            # be saved directly in its final, non-transparent state.
            plt.savefig(filename,
                        bbox_inches='tight',
                        pad_inches=0,
                        transparent=False)
        else:
            # Save transparent render
            plt.savefig(filename_type_alpha_part,
                        bbox_inches='tight',
                        pad_inches=0,
                        transparent=True)
        masterprint('done')
    # All rendering done
    Barrier()
    # The partial renders will now be combined into full renders,
    # stored in the 'render_image', variable. Partial renders of the
    # j'th particle type will be handled by the process with rank j.
    if not (nprocs == 1 == len(render_dict)):
        masterprint('Compositing partial renders ...')
        particle_types = tuple(render_dict.keys())
        # Loop over particle types designated to each process
        for i in range(1 + len(render_dict)//nprocs):
            # Break out when there is no more work for this process
            j = rank + nprocs*i
            if j >= len(particle_types):
                break
            particle_type = particle_types[j].replace(' ', '-')
            if nprocs == 1:
                # Simply load the already fully constructed image
                filename_type_alpha = '{}/{}_alpha.png'.format(render_dirname, particle_type)
                render_image = plt.imread(filename_type_alpha)
            else:
                # Create list of filenames for the partial renders
                filenames_type_alpha_part = ['{}/{}_alpha_{}.png'.format(render_dirname,
                                                                         particle_type,
                                                                         part)
                                             for part in range(nprocs)]
                # Read in the partial renders and blend
                # them together into the render_image variable.
                blend(filenames_type_alpha_part)
                # Save combined render of particle type j
                # with transparency. Theese are then later combined into
                # a render containing all particle types.
                if len(particle_types) > 1:
                    filename_type_alpha = '{}/{}_alpha.png'.format(render_dirname, particle_type)
                    plt.imsave(filename_type_alpha, render_image)
            # Add opaque background to render_image
            add_background()
            # Save combined render of particle type j
            # without transparency.
            filename_type = filename
            if len(particle_types) > 1:
                if '_a=' in filename:
                    filename_type = filename.replace('_a=', '_{}_a='.format(particle_type))
                else:
                    filename_type = filename.replace('.png', '_{}.png'.format(particle_type))
            plt.imsave(filename_type, render_image)
        Barrier()
        # Finally, combine the full renders of individual particle types
        # into a total render containing all particles.
        if master and len(particle_types) > 1:
            filenames_type_alpha = ['{}/{}_alpha.png'.format(render_dirname,
                                                             particle_type.replace(' ', '-'))
                                    for particle_type in particle_types]
            blend(filenames_type_alpha)
            # Add opaque background to render_image and save it
            add_background()
            plt.imsave(filename, render_image)
        masterprint('done')
    # Remove the temporary directory, if cleanup is requested
    if master and cleanup and not (nprocs == 1 == len(render_dict)):
        shutil.rmtree(render_dirname)
    # Update the live render (local and remote)
    #update_liverender(filename_type)



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
    global render_image
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
    global render_image
    for i in range(resolution):
        for j in range(resolution):
            alpha = render_image[i, j, 3]
            # Add background using "A over B" alpha blending
            for rgb in range(3):
                render_image[i, j, rgb] = (alpha*render_image[i, j, rgb]
                                           + (1 - alpha)*bgcolor[rgb])
                render_image[i, j, 3] = 1

# Update local and remote live renders
@cython.header(# Arguments
               filename='str',
               )
def update_liverender(filename):
    # Updating the live render cannot be done in parallel
    if not master:
        return
    # Update the live render with the newly produced render 
    if liverender:
        masterprint('Updating live render "{}" ...'.format(liverender), indent=4)
        shutil.copy(filename, liverender)
        masterprint('done')
    # Updating the remote live render with the newly produced render
    if not remote_liverender or not scp_password:
        return
    cmd = 'scp "{}" "{}"'.format(filename, remote_liverender)
    scp_host = re.search('@(.*):', remote_liverender).group(1)
    scp_dist = re.search(':(.*)',  remote_liverender).group(1)
    masterprint('Updating remote live render "{}:{}" ...'.format(scp_host, scp_dist),
                indent=4)
    expects = ['password.',
               'passphrase.',
               'continue connecting',
               pexpect.EOF,
               pexpect.TIMEOUT,
               ]
    child = pexpect.spawn(cmd, timeout=15, env={'SSH_ASKPASS': '', 'DISPLAY': ''})
    for i in range(2):
        n = child.expect(expects)
        if n < 2:
            # scp asks for password or passphrase. Supply it
            child.sendline(scp_password)
        elif n == 2:
            # scp cannot authenticate host. Connect anyway
            child.sendline('yes')
        elif n == 3:
            break
        else:
            child.kill(9)
            break
    child.close(force=True)
    if child.status:
        msg = "Remote live render could not be scp'ed to" + scp_host
        masterwarn(msg)
    else:
        masterprint('done')

# This function projects the particle positions onto the xy-plane
# and renders this projection directly in the terminal, using
# ANSI/VT100 control sequences.
@cython.header(# Arguments
               particles_list='list',
               # Locals
               N='Py_ssize_t',
               N_local='Py_ssize_t',
               N_total='Py_ssize_t',
               colornumber='Py_ssize_t',
               i='Py_ssize_t',
               index_x='Py_ssize_t',
               index_y='Py_ssize_t',
               j='Py_ssize_t',
               maxoverdensity='Py_ssize_t',
               maxval='Py_ssize_t',
               particles='Particles',
               posx='double*',
               posy='double*',
               projection_ANSI='list',
               scalec='double',
               )
def terminal_render(particles_list):
    global projection
    # Project all particle positions onto the 2D projection array,
    # counting the number of particles in each pixel.
    projection[...] = 0
    N_total = 0
    for particles in particles_list:
        if particles.representation == 'particles':
            # Extract relevant particle data
            N = particles.N
            N_local = particles.N_local
            N_total += N
            posx = particles.posx
            posy = particles.posy
            # Do the projection
            for i in range(N_local):
                index_x = int(posx[i]*projection_scalex)
                index_y = int(posy[i]*projection_scaley)
                projection[index_y, index_x] += 1
        elif particles.representation == 'fluid':
            pass
    # Sum up local projections into the master process
    Reduce(sendbuf=(MPI.IN_PLACE if master else projection),
           recvbuf=(projection   if master else None),
           op=MPI.SUM)
    if not master:
        return
    # Values in the projection array equal to or larger than maxval
    # will be mapped to color nr. 255. The value of the maxoverdensity
    # is arbitrarily chosen.
    maxoverdensity = 12
    maxval = maxoverdensity*N_total//(projection.shape[0]*projection.shape[1])
    if maxval < 5:
        maxval = 5
    # Construct list of strings, each string being a space prepended
    # with an ANSI/VT100 control sequences which sets the background
    # color. When printed together, these strings produce an ANSI image
    # of the projection.
    projection_ANSI = []
    scalec = 240/maxval
    for i in range(projection.shape[0]):
        for j in range(projection.shape[1]):
            colornumber = int(16 + projection[i, j]*scalec)
            if colornumber > 255:
                colornumber = 255
            projection_ANSI.append('\x1b[48;5;{}m '.format(colornumber))
        projection_ANSI.append('\x1b[0m\n')
    # Print the ANSI image
    masterprint(''.join(projection_ANSI), end='')



# Declare global variables used in above functions
cython.declare(render_dict='object',  # OrderedDict
               render_image='float[:, :, ::1]',
               )
# Prepare a figure for the render
if render_times or special_params.get('special', '') == 'render':
    # (Ordered) dictionary containing the figure, axes, particles
    # artist and text artist for each particle type.
    render_dict = collections.OrderedDict()
# The array storing the render
render_image = empty((resolution, resolution, 4), dtype=C2np['float'])

# Prepare a figure for the powerspec plot
if (any(powerspec_plot_select.values())
    and (powerspec_times or special_params.get('special', '') == 'powerspec')):
    fig_powerspec = plt.figure('powerspec')
# The array storing the terminal render and the color map
if terminal_render_times:
    # Allocate the 2D projection array storing the terminal render
    cython.declare(projection='Py_ssize_t[:, ::1]',
                   projection_scalex='double',
                   projection_scaley='double',
                   )
    projection = np.empty((terminal_resolution//2, terminal_resolution),
                          dtype=C2np['Py_ssize_t'])
    projection_scalex = projection.shape[1]/boxsize
    projection_scaley = projection.shape[0]/boxsize
    # Construct instance of the colormap with 256 - 16 = 240 colors
    colormap_240 = getattr(matplotlib.cm, terminal_colormap)(arange(240))[:, :3]
    # Apply the colormap to the terminal, remapping the 240 higher color
    # numbers. The 16 lowest are left alone in order not to mess with
    # standard terminal coloring.
    if terminal_render_times:
        for i in range(240):
            colorhex = matplotlib.colors.rgb2hex(colormap_240[i])
            masterprint('\x1b]4;{};rgb:{}/{}/{}\x1b\\'
                         .format(16 + i, colorhex[1:3],
                                         colorhex[3:5],
                                         colorhex[5:]), end='')
