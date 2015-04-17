# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Use a matplotlib backend that does not require a running X-server
import matplotlib
matplotlib.use('Agg')
# Imports for plotting
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
from matplotlib.pyplot import figure, savefig

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    pass
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    """

# Imports and definitions common to pure Python and Cython
import pexpect
from os.path import basename


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
               # Locals
               N='size_t',
               N_local='size_t',
               N_locals='size_t[::1]',
               figsize='double[::1]',
               i='int',
               inch2pts='double',
               j='int',
               )
def animate(particles, timestep, a, a_snapshot):
    global all_posx, all_posy, all_posz, all_posx_mv, all_posy_mv, all_posz_mv
    global artist, upload_liveframe, ax
    if not visualize or (timestep % framespace and a != a_snapshot):
        return
    # Frame should be animated. Print out message
    if master:
        print('Rendering image')
    N = particles.N
    N_local = particles.N_local
    # The master process gathers N_local from all processes
    N_locals = empty(nprocs if master else 0, dtype='uintp')
    Gather(array(N_local, dtype='uintp'), N_locals)
    # Increase the buffer sizes
    if all_posx_mv.shape[0] < N:
        all_posx = realloc(all_posx, N*sizeof('double'))
        all_posy = realloc(all_posy, N*sizeof('double'))
        all_posz = realloc(all_posz, N*sizeof('double'))
        all_posx_mv = cast(all_posx, 'double[:N]')
        all_posy_mv = cast(all_posy, 'double[:N]')
        all_posz_mv = cast(all_posz, 'double[:N]')
    # The master process gathers all particle positions
    Gatherv(sendbuf=particles.posx_mv[:N_local],
            recvbuf=(all_posx_mv, N_locals))
    Gatherv(sendbuf=particles.posy_mv[:N_local],
            recvbuf=(all_posy_mv, N_locals))
    Gatherv(sendbuf=particles.posz_mv[:N_local],
            recvbuf=(all_posz_mv, N_locals))
    # Only the master process plots the particle data
    if not master:
        return
    # Set up figure. This is only done in the first call.
    if artist is None:
        # Set up figure
        inch2pts = 72.27  # Number of points in an inch
        fig = figure()
        ax = fig.add_subplot(111, projection='3d', axisbg='black')
        # The alpha value is chosen so that a column of particles in a
        # homogeneous universe will appear to have alpha = 1 (more or
        # less). The size of a particle is plotted so that the particles
        # stand side by side in a homogeneous unvierse (more or less).
        figsize = fig.get_size_inches()
        artist = ax.scatter(all_posx_mv, all_posy_mv, all_posz_mv,
                            lw=0,
                            alpha=N**(-one_third),
                            c=(180.0/256, 248.0/256, 95.0/256),
                            s=np.min((50, prod(figsize)*inch2pts**2/N)),
                            )
        ax.set_xlim3d(0, boxsize)
        ax.set_ylim3d(0, boxsize)
        ax.set_zlim3d(0, boxsize)
        ax.w_xaxis.set_pane_color(zeros(4))
        ax.w_yaxis.set_pane_color(zeros(4))
        ax.w_zaxis.set_pane_color(zeros(4))
        ax.w_xaxis.gridlines.set_lw(0)
        ax.w_yaxis.gridlines.set_lw(0)
        ax.w_zaxis.gridlines.set_lw(0)
        # Print the scale factor at the location of the xlabel
        ax.xaxis.set_rotate_label(False)
        ax.set_xlabel('$a = '
                      + significant_figures(a, 4, just=0, scientific=True)
                      + '$', rotation=0)
        ax.xaxis.label.set_color('white')
    # Update figure
    artist._offsets3d = juggle_axes(all_posx_mv, all_posy_mv, all_posz_mv,
                                    zdir='z')
    ax.set_xlabel('$a = ' + significant_figures(a, 4, just=0,
                                                scientific=True)
                          + '$', rotation=0)
    if save_frames:
        # Print out message
        print('    Saving: ' + framefolder + str(timestep) + suffix)
        # Save the frame in framefolder
        savefig(framefolder + str(timestep) + suffix,
                bbox_inches='tight', pad_inches=0, dpi=160)
    if save_liveframe:
        # Print out message
        print('    Updating live frame: ' + liveframe_full)
        # Save the live frame
        savefig(liveframe_full,
                bbox_inches='tight', pad_inches=0, dpi=160)
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

# Set the artist as uninitialized at import time
artist = None
# Preparation for saving frames done at import time
cython.declare(all_posx='double*',
               all_posx_mv='double[::1]',
               all_posy='double*',
               all_posy_mv='double[::1]',
               all_posz='double*',
               all_posz_mv='double[::1]',
               liveframe_full='str',
               save_liveframe='bint',
               save_frames='bint',
               cmd1='str',
               cmd2='str',
               password='str',
               upload_liveframe='bint',
               user_at_host='str',
               suffix='str',
               visualize='bint',
               )
# Buffers for particle positions
all_posx = malloc(1*sizeof('double'))
all_posy = malloc(1*sizeof('double'))
all_posz = malloc(1*sizeof('double'))
all_posx_mv = cast(all_posx, 'double[:1]')
all_posy_mv = cast(all_posy, 'double[:1]')
all_posz_mv = cast(all_posz, 'double[:1]')
# Check whether frames should be stored and create the
# framefolder folder at import time
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
