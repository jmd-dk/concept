# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: you can redistribute it and/or modify
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
# The auther of CO𝘕CEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
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
               filename='str',
               k='double[::1]',
               power='double[::1]',
               power_σ='double[::1]',
               # Locals
               tmp='str',
               )
def plot_powerspec(filename, k, power, power_σ):
    # Only the master process takes part in the plotting
    if not master:
        return
    # Attach missing extension to filename
    if not filename.endswith('.png'):
        filename += '.png'
    masterprint('Plotting power spectrum and saving to "{}" ...'.format(filename))
    # Switch to the powerspec figure and clean the figure
    plt.figure(fig_powerspec_nr)
    plt.clf()
    # Plot powerspectrum
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log', nonposy='clip')
    plt.errorbar(k, power, yerr=power_σ, fmt='b.', ms=3, ecolor='r', lw=0)
    plt.xlabel('$k\,\mathrm{[' + base_length + '^{-1}]}$')
    plt.ylabel('$\mathrm{power}\,\mathrm{[' + base_length + '^3]}$')
    tmp = '{:.1e}'.format(min(k))
    plt.xlim(xmin=float(tmp[0] + tmp[3:]))
    tmp = '{:.1e}'.format(max(k))
    plt.xlim(xmax=float(str(int(tmp[0]) + 1) + tmp[3:]))
    tmp = '{:.1e}'.format(min(power))
    plt.ylim(ymin=float(tmp[0] + tmp[3:]))
    tmp = '{:.1e}'.format(np.max(asarray(power) + asarray(power_σ)))
    plt.ylim(ymax=float(str(int(tmp[0]) + 1) + tmp[3:]))
    plt.savefig(filename)
    # Finish progress message
    masterprint('done')


# Setting up figure and plot the particles
@cython.header(# Arguments
               particles='Particles',
               a='double',
               filename='str',
               cleanup='bint',
               # Locals
               a_str='str',
               alpha='double',
               alpha_min='double',
               basename='str',
               combined='double[:, :, ::1]',
               dirname='str',
               N='Py_ssize_t',
               N_local='Py_ssize_t',
               renderpart_filename='str',
               renderpart_filenames='list',
               renderparts_dirname='str',
               i='int',
               r='int',
               size='double',
               size_max='double',
               size_min='double',
               )
def render(particles, a, filename, cleanup=True):
    global artist_particles, artist_text, ax_render
    # Print out progress message
    masterprint('Rendering and saving image "{}" ...'.format(filename))
    # Switch to the render figure
    plt.figure(fig_render_nr)
    # Attach missing extension to filename
    if not filename.endswith('.png'):
        filename += '.png'
    # Extract particle data
    N = particles.N
    N_local = particles.N_local
    # Update particle positions on the figure
    artist_particles._offsets3d = juggle_axes(particles.posx_mv[:N_local],
                                              particles.posy_mv[:N_local],
                                              particles.posz_mv[:N_local],
                                              zdir='z')
    # The particle size on the figure.
    # The size is chosen such that the particles stand side by side in a
    # homogeneous unvierse (more or less).
    size = 1000*np.prod(fig_render.get_size_inches())/N**ℝ[2/3]
    # The particle alpha on the figure.
    # The alpha is chosen such that in a homogeneous unvierse, a column
    # of particles have a collective alpha of 1 (more or less).
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
        artist_text.set_text('')
        a_str = significant_figures(a, 4, 'LaTeX')
        artist_text = ax_render.text(+0.25*boxsize,
                                     -0.3*boxsize,
                                     0,
                                     '$a = {}$'.format(a_str),
                                     fontsize=16,
                                     )
        # Make the text color black or white, dependent on the bgcolor
        if sum(bgcolor) < 1:
            artist_text.set_color('white')
        else:
            artist_text.set_color('black')
    # Update axis limits if a boxsize were explicitly passed
    if boxsize:
        ax_render.set_xlim(0, boxsize)
        ax_render.set_ylim(0, boxsize)
        ax_render.set_zlim(0, boxsize)
    # If running with a single process, save the render, make a call to
    # update the liverender and then return.
    if nprocs == 1:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        masterprint('done')
        update_liverender(filename)
        return
    # Running with multiple processes.
    # Each process save its rendered part to disk.
    # First, make a temporary directory to hold the render parts.
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    renderparts_dirname = '{}/.renderparts'.format(dirname)
    renderpart_filename = '{}/{}.png'.format(renderparts_dirname, rank)
    if master:
        os.makedirs(renderparts_dirname, exist_ok=True)
    # Now save the render parts, including transparency
    Barrier()
    plt.savefig(renderpart_filename,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True,
                )
    Barrier()
    # The master process combines the parts using ImageMagick
    if master:
        # List of all newly created renderparts
        renderpart_filenames = ['{}/{}.png'.format(renderparts_dirname, r) for r in range(nprocs)]
        # Combine all render parts into one,
        # with the correct background color and no transparency.
        subprocess.call([paths['convert']] + renderpart_filenames
                         + ['-background', 'rgb({}%, {}%, {}%)'.format(100*bgcolor[0],
                                                                       100*bgcolor[1],
                                                                       100*bgcolor[2]),
                            '-layers', 'flatten', '-alpha', 'remove', filename])
        # Remove the temporary directory, if cleanup is requested
        if cleanup:
            shutil.rmtree(renderparts_dirname)
    masterprint('done')
    # Update the live render (local and remote)
    update_liverender(filename)

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
        masterprint('Updating live render "{}" ...'.format(liverender),
                    indent=4)
        shutil.copy(filename, liverender)
        masterprint('done')
    # Updating the remote live render with the newly produced render
    if not remote_liverender or not scp_password:
        return
    cmd = 'scp "{}" "{}"'.format(filename, remote_liverender)
    scp_host = re.search('@(.*):', remote_liverender).group(1)
    scp_dist = re.search(':(.*)',  remote_liverender).group(1)
    masterprint('Updating remote live render "{}:{}" ...'.format(scp_host,
                                                                 scp_dist),
                indent=4)
    expects = ['password.',
               'passphrase.',
               'continue connecting',
               pexpect.EOF,
               pexpect.TIMEOUT,
               ]
    child = pexpect.spawn(cmd, timeout=15, env={'SSH_ASKPASS': '',
                                                'DISPLAY'    : ''})
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
               particles='Particles',
               # Locals
               N='Py_ssize_t',
               N_local='Py_ssize_t',
               colornumber='unsigned long long int',
               i='Py_ssize_t',
               j='Py_ssize_t',
               maxval='unsigned long long int',
               posx='double*',
               posy='double*',
               projection='unsigned long long int[:, ::1]',
               projection_ANSI='list',
               scalec='double',
               scalex='double',
               scaley='double',
               )
def terminal_render(particles):
    # Extract particle data
    N = particles.N
    N_local = particles.N_local
    posx = particles.posx
    posy = particles.posy
    # Project particle positions onto a 2D array,
    # counting the number of particles in each pixel.
    projection = np.zeros((terminal_resolution//2, terminal_resolution),
                          dtype=C2np['unsigned long long int'])
    scalex = projection.shape[1]/boxsize
    scaley = projection.shape[0]/boxsize
    for i in range(N_local):
        projection[cast(particles.posy[i]*scaley, 'Py_ssize_t'),
                   cast(particles.posx[i]*scalex, 'Py_ssize_t')] += 1
    Reduce(sendbuf=(MPI.IN_PLACE if master else projection),
           recvbuf=(projection   if master else None),
           op=MPI.SUM)
    if not master:
        return
    # Values in the projection array equal to or larger than maxval
    # will be mapped to color nr. 255. The numerical coefficient is
    # more or less arbitrarily chosen.
    maxval = 12*N//(projection.shape[0]*projection.shape[1])
    if maxval < 5:
        maxval = 5
    # Construct list of strings, each string being a space prepended
    # with an ANSI/VT100 control sequences which sets the background
    # color. When printed together, these strings produce an ANSI image
    # of the projection.
    projection_ANSI = []
    scalec = 240.0/maxval
    for i in range(projection.shape[0]):
        for j in range(projection.shape[1]):
            colornumber = cast(16 + projection[i, j]*scalec, 'unsigned long long int')
            if colornumber > 255:
                colornumber = 255
            if colornumber < 16 or colornumber > 255:
                masterprint('wrong color:', colornumber, projection[i, j], scalec, projection[i, j]*scalec, maxval)
                sleep(1000)
            projection_ANSI.append('\x1b[48;5;{}m '.format(colornumber))
        projection_ANSI .append('\x1b[0m\n')
    # Print the ANSI image
    masterprint(''.join(projection_ANSI), end='')

# Set up figures
cython.declare(fig_render_nr='int',
               fig_powerspec_nr='int')
fig_render_nr    = 1
fig_powerspec_nr = 2
# Prepare a figure for the render
if render_times or special_params.get('special', '') == 'render':
    # The 77.50 scaling is needed to map the resolution to pixel units
    fig_render = plt.figure(fig_render_nr, figsize=[resolution/77.50]*2)
    ax_render = fig_render.gca(projection='3d', axisbg=bgcolor)
    ax_render.set_aspect('equal')
    ax_render.dist = 8.55  # Zoom level
    # The artist for the particles
    artist_particles = ax_render.scatter(0, 0, 0, color=color, lw=0)
    # The artist for the scalefactor text
    artist_text = ax_render.text(0, 0, 0, '')
    # Configure axis options
    ax_render.set_xlim(0, boxsize)
    ax_render.set_ylim(0, boxsize)
    ax_render.set_zlim(0, boxsize)
    ax_render.w_xaxis.set_pane_color(zeros(4))
    ax_render.w_yaxis.set_pane_color(zeros(4))
    ax_render.w_zaxis.set_pane_color(zeros(4))
    ax_render.w_xaxis.gridlines.set_lw(0)
    ax_render.w_yaxis.gridlines.set_lw(0)
    ax_render.w_zaxis.gridlines.set_lw(0)
    ax_render.grid(False)
    ax_render.w_xaxis.line.set_visible(False)
    ax_render.w_yaxis.line.set_visible(False)
    ax_render.w_zaxis.line.set_visible(False)
    ax_render.w_xaxis.pane.set_visible(False)
    ax_render.w_yaxis.pane.set_visible(False)
    ax_render.w_zaxis.pane.set_visible(False)
    for tl in ax_render.w_xaxis.get_ticklines():
        tl.set_visible(False)
    for tl in ax_render.w_yaxis.get_ticklines():
        tl.set_visible(False)
    for tl in ax_render.w_zaxis.get_ticklines():
        tl.set_visible(False)
    for tl in ax_render.w_xaxis.get_ticklabels():
        tl.set_visible(False)
    for tl in ax_render.w_yaxis.get_ticklabels():
        tl.set_visible(False)
    for tl in ax_render.w_zaxis.get_ticklabels():
        tl.set_visible(False)
# Prepare a figure for the powerspec plot
if powerspec_plot and (powerspec_times or special_params.get('special', '') == 'powerspec'):
    fig_powerspec = plt.figure(fig_powerspec_nr)

# The color map for the terminal render
if terminal_render_times:
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
