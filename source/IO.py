# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from species import construct
    from communication import exchange_all
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from species cimport construct
    from communication cimport exchange_all
    """

# Imports and definitions common to pure Python and Cython
import struct

# Function that saves particle data to an hdf5 file
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               particles='Particles',
               filename='str',
               # Locals
               N='size_t',
               N_local='size_t',
               N_locals='size_t[::1]',
               end_local='size_t',
               start_local='size_t',
               )
@cython.returns('Particles')
def save(particles, filename):
    # Print out message
    if master:
        print('Saving snapshot:', filename)
    with h5py.File(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
        # Create HDF5 group and datasets
        N = particles.N
        particles_h5 = hdf5_file.create_group('particles/' + particles.type)
        posx_h5 = particles_h5.create_dataset('posx', [N], dtype='float64')
        posy_h5 = particles_h5.create_dataset('posy', [N], dtype='float64')
        posz_h5 = particles_h5.create_dataset('posz', [N], dtype='float64')
        momx_h5 = particles_h5.create_dataset('momx', [N], dtype='float64')
        momy_h5 = particles_h5.create_dataset('momy', [N], dtype='float64')
        momz_h5 = particles_h5.create_dataset('momz', [N], dtype='float64')
        # Get local indices of the particle data
        N_local = particles.N_local
        N_locals = empty(nprocs, dtype='uintp')
        Allgather(array(N_local, dtype='uintp'), N_locals)
        start_local = sum(N_locals[:rank])
        end_local = start_local + N_local
        # In pure Python, the indices needs to be Python integers
        if not cython.compiled:
            start_local = int(start_local)
            end_local = int(end_local)
        # Save the local slices of the particle data and the attributes
        posx_h5[start_local:end_local] = particles.posx_mw[:N_local]
        posy_h5[start_local:end_local] = particles.posy_mw[:N_local]
        posz_h5[start_local:end_local] = particles.posz_mw[:N_local]
        momx_h5[start_local:end_local] = particles.momx_mw[:N_local]
        momy_h5[start_local:end_local] = particles.momy_mw[:N_local]
        momz_h5[start_local:end_local] = particles.momz_mw[:N_local]
        particles_h5.attrs['type'] = particles.type
        particles_h5.attrs['species'] = particles.species
        particles_h5.attrs['mass'] = particles.mass

# Function that loads particle data from an hdf5 file and instantiate a
# Particles instance on each process, storing the particles within its domain.
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               filename='str',
               # Locals
               N='size_t',
               N_local='size_t',
               N_locals='tuple',
               end_local='size_t',
               particle_type='str',
               particles='Particles',
               start_local='size_t',
               )
@cython.returns('Particles')
def load(filename):
    # Print out message
    if master:
        print('Loading snapshot:', filename)
    # Load all particles
    with h5py.File(filename, mode='r', driver='mpio', comm=comm) as hdf5_file:
        all_particles = hdf5_file['particles']
        for particle_type in all_particles:
            # Extract HDF5 group and datasets
            particles_h5 = all_particles[particle_type]
            posx_h5 = particles_h5['posx']
            posy_h5 = particles_h5['posy']
            posz_h5 = particles_h5['posz']
            momx_h5 = particles_h5['momx']
            momy_h5 = particles_h5['momy']
            momz_h5 = particles_h5['momz']
            # Compute a fair distribution of particle data to the processes
            N = posx_h5.size
            N_locals = ((N//nprocs, )*(nprocs - (N%nprocs))
                        + (N//nprocs + 1, )*(N%nprocs))
            N_local = N_locals[rank]
            start_local = sum(N_locals[:rank])
            end_local = start_local + N_local
            # In pure Python, the indices must be Python integers
            if not cython.compiled:
                start_local = int(start_local)
                end_local = int(end_local)
            # Construct a Particles instance
            particles = construct(particles_h5.attrs['type'],
                                  particles_h5.attrs['species'],
                                  mass=particles_h5.attrs['mass'],
                                  N=N,
                                  )
            # Populate the Particles instance with data from the file
            particles.populate(posx_h5[start_local:end_local], 'posx')
            particles.populate(posy_h5[start_local:end_local], 'posy')
            particles.populate(posz_h5[start_local:end_local], 'posz')
            particles.populate(momx_h5[start_local:end_local], 'momx')
            particles.populate(momy_h5[start_local:end_local], 'momy')
            particles.populate(momz_h5[start_local:end_local], 'momz')
    # Scatter particles to the correct domain-specific process
    exchange_all(particles)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # THE PARTICLES VARIABLE SHOULD BE A TUPLE OF PARTICLES OR SOMETHING
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return particles






@cython.cclass
class Gadget_snapshot:
    """
    """

    # Initialization method. Note that data attributes are declared in the .pxd file.
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self):
        self.HEAD = {}

    # Method for loading in a Gadget snapshot of type 2 from disk
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   filename='str',
                   # Locals
                   name='str',
                   size='int',
                   N='size_t',
                   N_local='size_t',
                   N_locals='tuple',
                   file_position='size_t',
                   gadget_H0='double',
                   gadget_boxsize='double',
                   gadget_Ωm='double',
                   gadget_ΩΛ='double',
                   start_local='size_t',
                   conversion_fac='double',
                   )
    def load(self, filename):
        """ It is assumed that the snapshot on the disk is a Gadget snapshot
        of type 2 and that it uses single precision. The Gadget_snapshot
        instance stores the data (positions and velocities) in double
        precision. Only Gadget type 1 (halo) particles, corresponding to
        dark matter particles, are supported.
        """
        self.offset = 0
        with open(filename, 'rb') as self.f:
            # Read in the HEAD block. No unit conversion will be done.
            self.new_block()
            name = self.read('4s').decode('utf8').rstrip()  # HEAD
            size = self.read('i')  # 264
            self.new_block()
            self.HEAD['Npart']         = self.read('6I')
            self.HEAD['Massarr']       = self.read('6d')
            self.HEAD['Time']          = self.read('d')
            self.HEAD['Redshift']      = self.read('d')
            self.HEAD['FlagSfr']       = self.read('i')
            self.HEAD['FlagFeedback']  = self.read('i')
            self.HEAD['Nall']          = self.read('6i')
            self.HEAD['FlagCooling']   = self.read('i')
            self.HEAD['NumFiles']      = self.read('i')
            self.HEAD['BoxSize']       = self.read('d')
            self.HEAD['Omega0']        = self.read('d')
            self.HEAD['OmegaLambda']   = self.read('d')
            self.HEAD['HubbleParam']   = self.read('d')
            self.HEAD['FlagAge']       = self.read('i')
            self.HEAD['FlagMetals']    = self.read('i')
            self.HEAD['NallHW']        = self.read('6i')
            self.HEAD['flag_entr_ics'] = self.read('i')
            # Check if the cosmology of the snapshot matches
            # that of the current simulation run. Display a
            # warning if it does not.
            tol = 1e-5
            gadget_boxsize = (self.HEAD['BoxSize']*units.kpc
                              /self.HEAD['HubbleParam'])
            gadget_H0 = (self.HEAD['HubbleParam']*100*units.km
                         /(units.s*units.Mpc))
            gadget_Ωm = self.HEAD['Omega0']
            gadget_ΩΛ = self.HEAD['OmegaLambda']
            if any([abs(gadget_param/param - 1) > tol for gadget_param, param
                    in zip((gadget_boxsize, gadget_H0, gadget_Ωm, gadget_ΩΛ),
                           (boxsize, H0, Ωm, ΩΛ))]):
                print('\033[91m\033[1m' + 'Warning: Mismatch between current '
                      + 'parameters and those in the Gadget snapshot "'
                      + filename + '":' + '\033[0m')
            if abs(gadget_boxsize/boxsize - 1) > tol:
                print('\033[91m\033[1m' + '    boxsize: ' + str(boxsize)
                       + ' vs ' + str(gadget_boxsize) + ' (kpc)' + '\033[0m')
            if abs(gadget_H0/H0 - 1) > tol:
                print('\033[91m\033[1m' + '    H0: '
                       + str(H0/(units.km/(units.s*units.Mpc)))
                       + ' vs ' + str(gadget_H0/(units.km/(units.s*units.Mpc)))
                       + ' (km/s/Mpc)' + '\033[0m')
            if abs(gadget_Ωm/Ωm - 1) > tol:
                print('\033[91m\033[1m' + '    \N{GREEK CAPITAL LETTER OMEGA}m: '
                       + str(Ωm) + ' vs ' + str(gadget_Ωm) + '\033[0m')
            if abs(gadget_ΩΛ/ΩΛ - 1) > tol:
                print('\033[91m\033[1m' + '    \N{GREEK CAPITAL LETTER OMEGA}'
                       + '\N{GREEK CAPITAL LETTER LAMDA}: '
                       + str(ΩΛ) + ' vs ' + str(gadget_ΩΛ) + '\033[0m')
            # Compute a fair distribution of particle data to the processes
            N = self.HEAD['Npart'][1]
            N_locals = ((N//nprocs, )*(nprocs - (N%nprocs))
                        + (N//nprocs + 1, )*(N%nprocs))
            N_local = N_locals[rank]
            start_local = sum(N_locals[:rank])
            # In pure Python, the index must be a Python integer
            if not cython.compiled:
                start_local = int(start_local)
            # Construct a Particles instance
            self.particles = construct('from Gadget snapshot',
                                       'dark matter',
                                       mass=(self.HEAD['Massarr'][1]
                                             /self.HEAD['HubbleParam']),
                                       N=N,
                                       )
            # Read in the POS block. The positions are given in kpc/h.
            conversion_fac = units.kpc/self.HEAD['HubbleParam']
            self.new_block()
            name = self.read('4s').decode('utf8').rstrip()  # POS
            size = (self.read('i') - 8)//struct.calcsize('f')
            self.new_block()
            self.f.seek(12*start_local, 1)  # 12 = sizeof(float32) x Ndims
            file_position = self.f.tell()
            self.particles.populate(asarray(np.fromfile(self.f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [0::3], dtype='float64')
                                    *conversion_fac,
                                    'posx')
            self.f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(self.f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [1::3], dtype='float64')
                                    *conversion_fac,
                                    'posy')
            self.f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(self.f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [2::3], dtype='float64')
                                    *conversion_fac,
                                    'posz')
            # Read in the VEL block. The velocities are peculiar
            # velocities u=a*dx/dt divided by sqrt(a), given in km/s.
            convertion_fac = units.km/units.s*self.particles.mass*self.HEAD['Time']**1.5
            self.new_block()
            name = self.read('4s').decode('utf8').rstrip()  # VEL
            size = (self.read('i') - 8)//struct.calcsize('f')
            self.new_block()
            self.f.seek(12*start_local, 1)  # 12 = sizeof(float32) x Ndims
            file_position = self.f.tell()
            self.particles.populate(asarray(np.fromfile(self.f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [0::3], dtype='float64')
                                    *conversion_fac,
                                    'momx')
            self.f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(self.f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [1::3], dtype='float64')
                                    *conversion_fac,
                                    'momy')
            self.f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(self.f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [2::3], dtype='float64')
                                    *conversion_fac,
                                    'momz')
            # Possible additional meta data
            while True:
                try:
                    # READ IN MASSES AND STUFF
                    self.new_block()
                    name = self.read('4s').decode('utf8').rstrip()
                    size = self.read('i')
                except:
                    break

    # Method used for reading series of bytes from the snapshot file
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   fmt='str',
                   # Locals
                   t='tuple',
                   )
    def read(self, fmt):
        # Convert bytes to python objects and store them in a tuple
        t = struct.unpack(fmt, self.f.read(struct.calcsize(fmt))) 
        # If the tuple contains just a single element, return this
        # element rather than the tuple.
        if len(t) == 1:
            return t[0]
        return t

    # Method that handles the file object's position in the snapshot file
    # during loading. Call it when the next block should be read.
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def new_block(self):
        # Set the current position in the file
        self.f.seek(self.offset)
        # Each block is bracketed with a 4-byte int
        # containing the size of the block
        self.offset += 8 + self.read('i')


cython.declare(snap='Gadget_snapshot')
snap = Gadget_snapshot()
snap.load('snapshot_005')


