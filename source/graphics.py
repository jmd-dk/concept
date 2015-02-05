# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
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
import os
import sys
import pexpect

# Setting up figure and plot the particles
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               timestep='size_t',
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
def animate(particles, timestep):
    global artist, scp_save_liveframe
    if not visualize or (timestep%framespace):
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
        if artist is None:
            # Set up figure
            fig = figure()
            ax = fig.add_subplot(111, projection='3d', axisbg='black')
            artist = ax.scatter(X, Y, Z,
                                lw=0,
                                alpha=0.2,
                                c='purple',
                                s=20,
                                )
            ax.set_xlim3d(0, boxsize)
            ax.set_ylim3d(0, boxsize)
            ax.set_zlim3d(0, boxsize)
            ax.set_axis_off()
        else:
            # Update figure
            artist._offsets3d = juggle_axes(X, Y, Z, zdir='z')
            if save_frames:
                # Save the frame in framefolder
                savefig(framefolder + str(timestep) + suffix,
                        bbox_inches='tight', pad_inches=0, dpi=160)
            if save_liveframe:
                # Save the live frame
                savefig(liveframe_full,
                        bbox_inches='tight', pad_inches=0, dpi=160)
                if scp_save_liveframe:
                    # scp the live frame
                    scpp = pexpect.spawn(scp_cmd, timeout=10)
                    try:
                        msg = scpp.expect(['password:', 'continue connecting (yes/no)?'])
                        if msg == 0:
                            # scp asks for password. Send it
                            scpp.sendline(scp_password)
                            msg = scpp.expect(['password:', pexpect.EOF])
                            if msg == 0:
                                # Incorrect password. Kill scp
                                scpp.terminate(force=False)
                                os.system(r'printf "\033[1m\033[91m'
                                          + 'Warning: Permission to '
                                          + scp_liveframe[:scp_liveframe.find(':')]
                                          + " denied\nFrames will not be scp'ed"
                                          + '\033[0m\n" >&2')
                                scp_save_liveframe = False
                        elif msg == 1:
                            # scp cannot authenticate host. Connect anyway
                            scpp.sendline('yes')
                            msg = scpp.expect(['password:', pexpect.EOF])
                            if msg == 0:
                                # scp asks for password. Send it
                                scpp.sendline(scp_password)
                                msg = scpp.expect(['password:', pexpect.EOF])
                                if msg == 0:
                                    # Incorrect password. Kill scp
                                    scpp.terminate(force=False)
                                    os.system(r'printf "\033[1m\033[91m'
                                              + 'Warning: Permission to '
                                              + scp_liveframe[:scp_liveframe.find(':')]
                                              + " denied\nFrames will not be scp'ed"
                                              + r'\033[0m\n" >&2')
                                    scp_save_liveframe = False
                        scpp.close()
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except:
                        # An error occurred during scp. 
                        scpp.terminate(force=False)
                        os.system(r'printf "\033[1m\033[91m'
                                  + 'Warning: An error occurred during scp to '
                                  + scp_liveframe[:scp_liveframe.find(':')]
                                  + '\033[0m\n" >&2')

# This function formats a floating point number f to have the length n
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               f='double',
               n='int',
               )
@cython.returns('str')
def format_number(f, n):
    return ('{:.' + str(n - len(str(int(f)))) + 'f}').format(f).ljust(n + 1)

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
              'Computation time: ' + format_number(time() - t_iter, 5) + ' s',
              'Scale factor:     ' + format_number(a, 5),
              'Cosmic time:      ' + format_number(t/units.Gyr, 5) + ' Gyr',
              sep='\n    ')

# Set the artist as uninitialized at import time
artist = None
# Preparation for saving frames done at import time
cython.declare(liveframe_full='str',
               save_liveframe='bint',
               save_frames='bint',
               scp_cmd='str',
               scp_password='str',
               scp_save_liveframe='bint',
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
    if not os.path.exists(framefolder):
        os.makedirs(framefolder)
    if framefolder[-1] != '/':
        framefolder += '/'
# Check whether to save a live frame
liveframe_full = ''
save_liveframe = False
scp_cmd=''
scp_save_liveframe = False
scp_password = ''
if liveframe != '':
    visualize = True
    save_liveframe = True
    liveframe_full = liveframe + suffix
    # Check whether to scp the live frame to a remote host
    if sys.argv[1] != '':
        scp_save_liveframe = True
        scp_password = sys.argv[1]
        scp_cmd = 'scp ' + liveframe_full + ' ' + scp_liveframe
