# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Use a matplotlib backend that does not require a running X-server
import matplotlib
matplotlib.use('Agg')
# Imports for plotting
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
from matplotlib.pyplot import figure, imread, imsave, savefig

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    pass
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    """

# Imports and definitions common to pure Python and Cython
import pexpect
from os.path import basename, dirname


# Setting up figure and plot the particles
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               timestep='size_t',
               a='double',
               a_snapshot='double',
               filename='str',
               # Locals
               N='size_t',
               N_local='size_t',
               alpha='double',
               c='int',
               combined='double[:, :, ::1]',
               figsize='double[::1]',
               framepart='double[:, :, ::1]',
               i='int',
               inch2pts='double',
               r='int',
               size='double',
               size_max='double',
               size_min='double',
               )
def animate(particles, timestep, a, a_snapshot, filename=''):
    global artist_particles, artist_text, upload_liveframe, ax, size_fac
    if not visualize or (timestep % framespace and a != a_snapshot):
        return
    # Frame should be animated. Print out message
    if master:
        print('Rendering image')
    # Extract particle data
    N = particles.N
    N_local = particles.N_local
    # Set up figure. This is only done in the first call.
    if artist_particles is None:
        # Set up figure
        fig = figure(figsize=[resolution/77.50]*2)
        ax = fig.gca(projection='3d', axisbg='black')
        ax.set_aspect('equal')
        ax.dist = 8.5
        # The size of a particle is plotted so that the particles
        # stand side by side in a homogeneous unvierse (more or less).
        inch2pts = 72.27  # Number of points in an inch
        figsize = fig.get_size_inches()
        size_fac = 0.05*prod(figsize)*inch2pts**2/N**two_thirds
        size = size_fac/sqrt(a)
        # The alpha value is chosen so that a column of particles in a
        # homogeneous universe will appear to have alpha = 0.75 (more or
        # less).
        alpha = 0.75*N**(-one_third)
        # For some reason, an alpha below 0.0059
        # is rendered as completely transparent.
        if alpha < 0.0059:
            alpha = 0.0059
        # Create the plot of the particles
        artist_particles = ax.scatter(particles.posx_mv[:N_local],
                                      particles.posy_mv[:N_local],
                                      particles.posz_mv[:N_local],
                                      alpha=alpha,
                                      lw=0,
                                      c=color,
                                      s=size,
                                      )
        ax.set_xlim(0, boxsize)
        ax.set_ylim(0, boxsize)
        ax.set_zlim(0, boxsize)
        ax.w_xaxis.set_pane_color(zeros(4))
        ax.w_yaxis.set_pane_color(zeros(4))
        ax.w_zaxis.set_pane_color(zeros(4))
        ax.w_xaxis.gridlines.set_lw(0)
        ax.w_yaxis.gridlines.set_lw(0)
        ax.w_zaxis.gridlines.set_lw(0)
        # Prepare the text
        artist_text = ax.text(+0.25*boxsize,
                              -0.3*boxsize,
                              0,
                              '',
                              color='white',
                              fontsize=16,
                              )
    else:
        # Update particle positions on figure
        artist_particles._offsets3d = juggle_axes(particles.posx_mv[:N_local],
                                                  particles.posy_mv[:N_local],
                                                  particles.posz_mv[:N_local],
                                                  zdir='z')
    # Size
    artist_particles._sizes[0] = size_fac*np.log(np.e/a)
    #artist_particles.set_alpha(alpha)
    # The master process prints the current scale factor on the figure
    if master:
        t0 = time()
        artist_text.set_text('$a = ' + significant_figures(a,
                                                           4,
                                                           just=0,
                                                           scientific=True,
                                                           )
                                     + '$')
    # When run on multiple processes, each process saves its part of the total
    # frame to disk, which is then read in by the master process and combined
    # to produce the total frame.
    if nprocs > 1:
        savefig(frameparts_folder + '.rank' + str(rank) + suffix,
                bbox_inches='tight', pad_inches=0)
        # When done saving the image, all processes but the master is done
        Barrier()
        if not master:
            return
        # The master process combines the images into one
        combined = np.asarray(imread(frameparts_folder + '.rank0' + suffix),
                              dtype='float64')
        for i in range(1, nprocs):
            framepart = np.asarray(imread(frameparts_folder + '.rank' + str(i)
                                                            + suffix),
                                   dtype='float64')
            for r in range(combined.shape[0]):
                for c in range(combined.shape[1]):
                    for rgb in range(3):
                        combined[r, c, rgb] += framepart[r, c, rgb]
        # Normalize the image. Values should not be above pixelval_max
        if pixelval_max > 1:
            for r in range(combined.shape[0]):
                for c in range(combined.shape[1]):
                    for rgb in range(3):
                        if combined[r, c, rgb] > pixelval_max:
                            combined[r, c, rgb] = 1
                        else:
                            combined[r, c, rgb] /= pixelval_max
        else:
            for r in range(combined.shape[0]):
                for c in range(combined.shape[1]):
                    for rgb in range(3):
                        if combined[r, c, rgb] > 1:
                            combined[r, c, rgb] = 1
    # When at the last frame, delete the auxiliary
    # image files of partial frames.
    if nprocs > 1 and a == a_max:
        for i in range(nprocs):
            os.remove(frameparts_folder + '.rank' + str(i) + suffix)
    # If explicit filename is passed, save to this file and return. Also reset
    # the particles artist, forcing the plot to setup from scratch the next
    # time this function is called (important when called from external script).
    if filename != '':
        print('    Saving: ' + framefolder + filename)
        artist_particles = None
        if nprocs > 1:
            imsave(framefolder + filename, combined)
        else:
            savefig(framefolder + filename, bbox_inches='tight', pad_inches=0)
        return
    # Save the frame in framefolder
    if save_frames:
        print('    Saving: ' + framefolder + str(timestep) + suffix)
        if nprocs > 1:
            imsave(framefolder + str(timestep) + suffix, combined)
        else:
            savefig(framefolder + str(timestep) + suffix,
                    bbox_inches='tight', pad_inches=0)
    if save_liveframe:
        # Print out message
        print('    Updating live frame: ' + liveframe_full)
        # Save the live frame
        if nprocs > 1:
            imsave(liveframe_full, combined)
        else:
            savefig(liveframe_full,
                    bbox_inches='tight', pad_inches=0)
        if upload_liveframe:
            # Print out message
            print('    Uploading live frame: ' + remote_liveframe)
            # Upload the live frame
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
                        else:
                            raise Exception
                    elif msg < 2:
                        # Incorrect password. Kill protocol
                        child.terminate(force=True)
                        warn('Permission to ' + user_at_host + ' denied\n'
                             + 'Frames will not be ' + protocol + "'ed")
                        upload_liveframe = False
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
                            warn('Permission to ' + user_at_host +
                                 + ' denied\nFrames will not be '
                                 + protocol + "'ed")
                            upload_liveframe = False
                child.close()
            except KeyboardInterrupt:
                # User tried to kill the program. Let her.
                raise KeyboardInterrupt
            except:
                # An error occurred during uploading. Print warning.
                child.terminate(force=False)
                warn('An error occurred during ' + protocol
                     + ' to ' + user_at_host)


# This function formats a floating point number f to only
# have n significant figures.
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
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
               )
@cython.returns('str')
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
    power = n - 1 - int(floor(log10(f)))
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


# Preparation for saving frames done at import time
cython.declare(a_max='double',
               frameparts_folder='str',
               liveframe_full='str',
               save_liveframe='bint',
               save_frames='bint',
               size_fac='double',
               cmd1='str',
               cmd2='str',
               password='str',
               pixelval_max='double',
               suffix='str',
               upload_liveframe='bint',
               user_at_host='str',
               visualize='bint',
               )
size_fac = 0
# Set the artists as uninitialized at import time
artist_particles = None
artist_text = None
# The scale factor at the last snapshot/frame
a_max = np.max(outputtimes)
# The maximum pixel value depends on the image format
if image_format in ('png', ):
    pixelval_max = 1
elif image_format in ('jpg', 'jpeg', 'tif', 'tiff'):
    pixelval_max = 255
else:
    raise ValueError('Unrecognized image format "' + image_format + '".')
# Check whether frames should be stored and create the
# framefolder folder at import time
frameparts_folder = ''
suffix = '.' + image_format
visualize = False
save_frames = False
if framefolder != '':
    visualize = True
    save_frames = True
    if master and not os.path.exists(framefolder):
        os.makedirs(framefolder)
    if framefolder[-1] != '/':
        framefolder += '/'
    frameparts_folder = framefolder
# Check whether to save a live frame
liveframe_full = ''
save_liveframe = False
cmd1 = ''
cmd2 = ''
upload_liveframe = False
password = ''
if liveframe != '':
    visualize = True
    save_liveframe = True
    liveframe_full = liveframe + suffix
    if frameparts_folder == '':
        frameparts_folder = dirname(liveframe) + '/'
    # Check whether to upload the live frame to a remote host
    if sys.argv[1] != '':
        upload_liveframe = True
        password = sys.argv[1]
        user_at_host = remote_liveframe[:remote_liveframe.find(':')]
        if protocol == 'scp':
            cmd1 = 'scp ' + liveframe_full + ' ' + remote_liveframe
        elif protocol == 'sftp':
            if remote_liveframe.endswith('.' + image_format):
                # Full filename given in remote_liveframe
                cmd1 = 'sftp ' + remote_liveframe[:remote_liveframe.rfind('/')]
                cmd2 = ('put ' + liveframe_full + ' '
                        + basename(remote_liveframe))
            else:
                # Folder given in remote_liveframe
                cmd1 = 'sftp ' + remote_liveframe
                cmd2 = 'put ' + liveframe_full
