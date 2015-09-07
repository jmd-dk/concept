# This file is part of CONCEPT, the cosmological N-body code in Python.
# Copyright (C) 2015 Jeppe Mosgard Dakin.
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CONCEPT. If not, see http://www.gnu.org/licenses/
#
# The auther of CONCEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CONCEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Use a matplotlib backend that does not require a running X-server
import matplotlib
matplotlib.use('Agg')
# Imports for plotting
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
from matplotlib.pyplot import figure, imsave, savefig

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    pass
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    """

# Imports and definitions common to pure Python and Cython
import pexpect
import subprocess

# Setting up figure and plot the particles
@cython.header(# Arguments
               particles='Particles',
               a='double',
               filename='str',
               passed_boxsize='double',
               # Locals
               N='size_t',
               N_local='size_t',
               c='int',
               combined='double[:, :, ::1]',
               renderpart='double[:, :, ::1]',
               i='int',
               r='int',
               size_max='double',
               size_min='double',
               )
def render(particles, a, filename, passed_boxsize=boxsize):
    global artist_particles, artist_text, upload_liverender, ax
    # Print out progress message
    masterprint('Rendering and saving image "' + filename + '" ...')
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
    size = 1000*np.prod(fig.get_size_inches())/N**two_thirds
    # The particle alpha on the figure.
    # The alpha is chosen such that in a homogeneous unvierse, a column
    # of particles have a collective alpha of 1 (more or less).
    alpha = N**(-one_third)
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
        a_str = significant_figures(a, 4, just=0, scientific=True)
        artist_text = ax.text(+0.25*passed_boxsize,
                              -0.3*passed_boxsize,
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
    if passed_boxsize:
        ax.set_xlim(0, passed_boxsize)
        ax.set_ylim(0, passed_boxsize)
        ax.set_zlim(0, passed_boxsize)
    # If running with a single process, save the render and return
    if nprocs == 1:
        savefig(filename, bbox_inches='tight', pad_inches=0)
        masterprint('done')
        return
    # Running with multiple processes.
    # Each process save its rendered part to disk.
    # First, make a temporary directory to hold the render parts.
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    renderparts_dirname = '{}/.renderparts'.format(dirname)
    renderpart_filename = '{}/rank{}.{}'.format(renderparts_dirname,
                                                rank,
                                                basename)
    if master:
        os.makedirs(renderparts_dirname, exist_ok=True)
    # Now save the render parts, including transparency
    Barrier()
    savefig(renderpart_filename,
            bbox_inches='tight',
            pad_inches=0,
            transparent=True,
            )
    Barrier()
    # The master process combines the parts using ImageMagick
    if master:
        # List of all newly created renderparts
        renderpart_filenames = [(renderparts_dirname + '/rank{}.' + basename)
                                 .format(r) for r in range(nprocs)]
        # Combine all render parts into one,
        # with the correct background color and no transparency.
        subprocess.call([paths['convert']] + renderpart_filenames
                         + ['-background', 'rgb({}%, {}%, {}%)'
                                            .format(100*bgcolor[0],
                                                    100*bgcolor[1],
                                                    100*bgcolor[2]),
                            '-layers', 'flatten', '-alpha', 'remove',
                            filename])
        # Remove the temporary directory
        shutil.rmtree(renderparts_dirname)
    masterprint('done')
    return
    if save_liverender:
        # Print out message
        masterprint('    Updating live render "' + liverender_full + '" ...')
        # Save the live render
        if nprocs > 1:
            imsave(liverender_full, combined)
        else:
            savefig(liverender_full,
                    bbox_inches='tight', pad_inches=0)
        masterprint('done')
        if upload_liverender:
            # Print out message
            masterprint('    Uploading live render "' + remote_liverender
                        + '" ...')
            # Upload the live render
            child = pexpect.spawn(cmd1, timeout=10)
            try:
                msg = child.expect(['password',
                                    'passphrase',
                                    pexpect.EOF,
                                    'continue connecting',
                                    ])
                if msg < 2:
                    # The protocol asks for password/passphrase.
                    # Send it.
                    child.sendline(password)
                    msg = child.expect(['password',
                                        'passphrase',
                                        pexpect.EOF,
                                        'sftp>',
                                        ])
                    if msg == 3:
                        # Logged in to remote host via sftp.
                        # Upload file.
                        child.sendline(cmd2)
                        msg = child.expect(['sftp>', pexpect.EOF])
                        if msg == 0:
                            child.sendline('bye')
                        elif master:
                            raise Exception
                    elif msg < 2:
                        # Incorrect password. Kill protocol
                        child.terminate(force=True)
                        masterwarn('Permission to ' + user_at_host
                                   + ' denied\n' + 'Renders will not be '
                                   + protocol + "'ed")
                        upload_liverender = False
                elif msg == 3:
                    # The protocol cannot authenticate host.
                    # Connect anyway.
                    child.sendline('yes')
                    msg = child.expect(['password:',
                                        'passphrase',
                                        pexpect.EOF,
                                        ])
                    if msg < 2:
                        # The protocol asks for password/passphrase.
                        # Send it.
                        child.sendline(password)
                        msg = child.expect(['password:',
                                            'passphrase',
                                            pexpect.EOF,
                                            ])
                        if msg < 2:
                            # Incorrect password/passphrase.
                            # Kill the protocol.
                            child.terminate(force=True)
                            masterwarn('Permission to ' + user_at_host
                                       + ' denied\nRenders will not be '
                                       + protocol + "'ed")
                            upload_liverender = False
                child.close()
            except KeyboardInterrupt:
                # User tried to kill the program. Let her
                if master:
                    raise KeyboardInterrupt
            except:
                # An error occurred during uploading. Print warning
                child.terminate(force=False)
                masterwarn('An error occurred during ' + protocol
                           + ' to ' + user_at_host)
            masterprint('done')



# Set up figure at import time.
# The 77.50 scaling is needed to map the resolution to pixel units
fig = figure(figsize=[resolution/77.50]*2)
ax = fig.gca(projection='3d', axisbg=bgcolor)
ax.set_aspect('equal')
ax.dist = 8.55  # Zoom level
# The artist for the particles
artist_particles = ax.scatter(0, 0, 0, color=color, lw=0)
# The artist for the scalefactor text
artist_text = ax.text(0, 0,0, '')
# Configure axis options
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

# This function formats a floating point
# number f to only have n significant figures.
@cython.header(# Arguments
               f='double',
               n='int',
               just='int',
               scientific='bint',
               # Locals
               e_index='int',
               f_str='str',
               power='int',
               power10='double',
               sign='int',
               returns='str',
               )
def significant_figures(f, n, just=0, scientific=False):
    sign = 1
    if f == 0:
        # Nothing fancy happens to zero
        return '0'.ljust(n + 1)
    elif f < 0:
        # Remove the minus sign, for now
        sign = -1
        f *= sign
    # Round to significant digits
    power = n - 1 - int(log10(f))
    power10 = 10.0**power
    f = round(f*power10)/power10
    f_str = str(f)
    # Convert to e notation if f is very large or very small
    if (len(f_str) - 1 - (f_str[(len(f_str) - 2):] == '.0') > n
        and not (len(f_str) > 2
                 and f_str[:2] == '0.'
                 and f_str[2] != '0')):
        f_str = ('{:.' + str(n) + 'e}').format(f)
    if 'e' in f_str:
        # In scientific (e) notation
        e_index = f_str.find('e')
        f_str = f_str[:np.min(((n + 1), e_index))] + f_str[e_index:]
        if scientific:
            e_index = f_str.find('e')
            f_str = (f_str.replace('e', r'\times 10^{'
                     + f_str[(e_index + 1):].replace('+', '') + '}'))
            f_str = f_str[:(f_str.find('}') + 1)]
        # Put sign back in
        if sign == -1:
            f_str = '-' + f_str
        return f_str.ljust(just)
    else:
        # Numbers which do not need *10^? to be nicely expressed
        if len(f_str) == n + 2 and (f_str[(len(f_str) - 2):] == '.0'):
            # Unwanted .0
            f_str = f_str[:n]
        elif (len(f_str) - 1 - (f_str[:2] == '0.')) < n:
            # Pad with zeros to get correct amount of figures
            f_str += '0'*(n - (len(f_str) - 1) + (f_str[:2] == '0.'))
        # Put sign back in
        if sign == -1:
            f_str = '-' + f_str
        return f_str.ljust(just)


# Preparation for saving renders, done at import time
cython.declare(renderparts_folder='str',
               liverender_full='str',
               save_liverender='bint',
               save_renders='bint',
               cmd1='str',
               cmd2='str',
               password='str',
               pixelval_max='double',
               suffix='str',
               upload_liverender='bint',
               user_at_host='str',
               visualize='bint',
               )

# Check whether to save a live render
liverender_full = ''
save_liverender = False
cmd1 = ''
cmd2 = ''
upload_liverender = False
password = ''
if liverender != '':
    visualize = True
    save_liverender = True
    liverender_full = liverender + suffix
    if renderparts_folder == '':
        renderparts_folder = os.pathdirname(liverender) + '/'
    # Check whether to upload the live render to a remote host
    if sys.argv[1] != '':
        upload_liverender = True
        password = sys.argv[1]
        user_at_host = remote_liverender[:remote_liverender.find(':')]
        if protocol == 'scp':
            cmd1 = 'scp ' + liverender_full + ' ' + remote_liverender
        elif protocol == 'sftp':
            if remote_liverender.endswith('.'):
                # Full filename given in remote_liverender
                cmd1 = 'sftp ' + remote_liverender[:remote_liverender.rfind('/')]
                cmd2 = ('put ' + liverender_full + ' '
                        + os.pathbasename(remote_liverender))
            else:
                # Folder given in remote_liverender
                cmd1 = 'sftp ' + remote_liverender
                cmd2 = 'put ' + liverender_full
