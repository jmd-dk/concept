# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Use a matplotlib backend that does not require a running X-server
import matplotlib
matplotlib.use('TkAgg')
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
@cython.cdivision(True)
@cython.boundscheck(False)
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
               X='double[::1]',
               Y='double[::1]',
               Z='double[::1]',
               i='int',
               j='int',
               )
def animate(particles, timestep, a, a_snapshot):
    global artist, upload_liveframe, ax
    if not visualize or ((timestep % framespace) and a != a_snapshot):
        return
    N = particles.N
    N_local = particles.N_local
    # The master process gathers N_local from all processes
    N_locals = empty(nprocs if master else 0, dtype='uintp')
    Gather(array(N_local, dtype='uintp'), N_locals)
    # The master process gathers all particle data
    X = empty(N if master else 0)
    Y = empty(N if master else 0)
    Z = empty(N if master else 0)
    sendbuf = particles.posx_mw[:N_local]
    Gatherv(sendbuf=sendbuf, recvbuf=(X, N_locals))
    sendbuf = particles.posy_mw[:N_local]
    Gatherv(sendbuf=sendbuf, recvbuf=(Y, N_locals))
    sendbuf = particles.posz_mw[:N_local]
    Gatherv(sendbuf=sendbuf, recvbuf=(Z, N_locals))
    # The master process plots the particle data
    if master:
        # Set up figure. This is only done in the first call.
        if artist is None:
            # Set up figure
            fig = figure()
            ax = fig.add_subplot(111, projection='3d', axisbg='black')
            artist = ax.scatter(X, Y, Z, lw=0,
                                alpha=0.05,
                                c=(64.0/256, 224.0/256, 208.0/256),
                                s=8,
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
        artist._offsets3d = juggle_axes(X, Y, Z, zdir='z')
        ax.set_xlabel('$a = ' + significant_figures(a, 4, just=0,
                                                    scientific=True)
                              + '$', rotation=0)
        if save_frames:
            # Save the frame in framefolder
            savefig(framefolder + str(timestep) + suffix,
                    bbox_inches='tight', pad_inches=0, dpi=160)
        if save_liveframe:
            # Save the live frame
            savefig(liveframe_full,
                    bbox_inches='tight', pad_inches=0, dpi=160)
            if upload_liveframe:
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
                            os.system('printf "\033[1m\033[91m'
                                      + 'Warning: Permission to '
                                      + user_at_host
                                      + " denied\nFrames will not be "
                                      + protocol + "'ed"
                                      + '\033[0m\n" >&2')
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
                                # Kill the protocol
                                child.terminate(force=True)
                                warn('Permission to ' + user_at_host
                                     + ' denied\nFrames will not be '
                                     + protocol + "'ed.")
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
@cython.cdivision(True)
@cython.boundscheck(False)
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
        f_str = f_str[:min((n + 1), e_index)] + f_str[e_index:]
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


# This function pretty prints information gathered through a time step
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               timestep='int',
               t_iter='double',
               a='double',
               t='double',
               )
def timestep_message(timestep, t_iter, a, t):
    if master:
        print('Time step ' + str(timestep) + ':',
              'Computation time: ' + significant_figures(time() - t_iter, 4,
                                                         just=7) + ' s',
              'Scale factor:     ' + significant_figures(a, 4, just=7),
              'Cosmic time:      ' + significant_figures(t/units.Gyr, 4,
                                                         just=7) + ' Gyr',
              sep='\n    ')


# Set the artist as uninitialized at import time
artist = None
# Preparation for saving frames done at import time
cython.declare(liveframe_full='str',
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
suffix = '.' + image_format
visualize = False
# Check whether frames should be stored and create the
# framefolder folder at import time
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
