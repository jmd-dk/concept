# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from species import construct
    from communication import exchange
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from species cimport construct
    from communication cimport exchange
    """

# Imports and definitions common to pure Python and Cython
import struct


# Function that saves particle data to an hdf5 file
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               particles='Particles',
               a='double',
               filename='str',
               )
def save(particles, a, filename):
    if output_type_fmt == 'standard':
        save_standard(particles, a, filename)
    elif output_type_fmt == 'gadget2':
        save_gadget(particles, a, filename)
    else:
        raise Exception('Error: Does not recognize output type "'
                        + output_type + '".')

# Function that saves particle data to an hdf5 file
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               particles='Particles',
               a='double',
               filename='str',
               # Locals
               N='size_t',
               N_local='size_t',
               N_locals='size_t[::1]',
               end_local='size_t',
               start_local='size_t',
               )
def save_standard(particles, a, filename):
    # Print out message
    if master:
        print('Saving snapshot:', filename)
    with h5py.File(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
        # Save global attributes
        hdf5_file.attrs['H0'] = H0
        hdf5_file.attrs['a'] = a
        hdf5_file.attrs['boxsize'] = boxsize
        hdf5_file.attrs['\N{GREEK CAPITAL LETTER OMEGA}m'] = Ωm
        hdf5_file.attrs[('\N{GREEK CAPITAL LETTER OMEGA}'
                       + '\N{GREEK CAPITAL LETTER LAMDA}')] = ΩΛ
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
        particles_h5.attrs['mass'] = particles.mass
        particles_h5.attrs['species'] = particles.species
        particles_h5.attrs['type'] = particles.type

# Function that loads particle data from an hdf5 file and instantiate a
# Particles instance on each process, storing the particles within its domain.
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               filename='str',
               # Locals
               N='size_t',
               N_local='size_t',
               N_locals='tuple',
               end_local='size_t',
               file_H0='double',
               #file_a='double',
               file_boxsize='double',
               file_Ωm='double',
               file_ΩΛ='double',
               msg='str',
               particle_type='str',
               particles='Particles',
               start_local='size_t',
               tol='double',
               )
@cython.returns('Particles')
def load_standard(filename):
    # Print out message
    if master:
        print('Loading snapshot:', filename)
    # Load all particles
    with h5py.File(filename, mode='r', driver='mpio', comm=comm) as hdf5_file:
        all_particles = hdf5_file['particles']
        for particle_type in all_particles:
            # Load global attributes
            file_H0 = hdf5_file.attrs['H0']
            #file_a = hdf5_file.attrs['a']
            file_boxsize = hdf5_file.attrs['boxsize']
            file_Ωm = hdf5_file.attrs['\N{GREEK CAPITAL LETTER OMEGA}m']
            file_ΩΛ = hdf5_file.attrs['\N{GREEK CAPITAL LETTER OMEGA}'
                                      + '\N{GREEK CAPITAL LETTER LAMDA}']
            # Check if the parameters of the snapshot matches those of the
            # current simulation run. Display a warning if they do not.
            tol = 1e-5
            if any([abs(file_param/param - 1) > tol for file_param, param
                    in zip((file_boxsize, file_H0, file_Ωm, file_ΩΛ),
                           (boxsize, H0, Ωm, ΩΛ))]):
                msg = ('Mismatch between current parameters and those in the '
                       + 'snapshot "' + filename + '":')
                #if abs(file_a/a_begin - 1) > tol:
                #    msg += ('\n    a_begin: ' + str(a_begin)
                #            + ' vs ' + str(file_a))
                if abs(file_boxsize/boxsize - 1) > tol:
                    msg += ('\n    boxsize: ' + str(boxsize) + ' vs '
                            + str(file_boxsize) + ' (kpc)')
                if abs(file_H0/H0 - 1) > tol:
                    msg += ('\n    H0: '
                            + str(H0/(units.km/(units.s*units.Mpc)))
                            + ' vs '
                            + str(file_H0/(units.km/(units.s*units.Mpc)))
                            + ' (km/s/Mpc)')
                if abs(file_Ωm/Ωm - 1) > tol:
                    msg += ('\n    \N{GREEK CAPITAL LETTER OMEGA}m: '
                            + str(Ωm) + ' vs ' + str(file_Ωm))
                if abs(file_ΩΛ/ΩΛ - 1) > tol:
                    msg += ('\n    \N{GREEK CAPITAL LETTER OMEGA}'
                            + '\N{GREEK CAPITAL LETTER LAMDA}: '
                            + str(ΩΛ) + ' vs ' + str(file_ΩΛ))
                warn(msg)
            # Extract HDF5 datasets
            particles_h5 = all_particles[particle_type]
            posx_h5 = particles_h5['posx']
            posy_h5 = particles_h5['posy']
            posz_h5 = particles_h5['posz']
            momx_h5 = particles_h5['momx']
            momy_h5 = particles_h5['momy']
            momz_h5 = particles_h5['momz']
            # Compute a fair distribution of particle data to the processes
            N = posx_h5.size
            N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                        + (N//nprocs + 1, )*(N % nprocs))
            N_local = N_locals[rank]
            start_local = np.sum(N_locals[:rank])
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
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # THE PARTICLES VARIABLE SHOULD BE A TUPLE OF PARTICLES OR SOMETHING
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return particles

# Function that loads particle data from an hdf5 file and instantiate a
# Particles instance on each process, storing the particles within its domain.
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               filename='str',
               # Locals
               input_type='str',
               particles='Particles',
               )
@cython.returns('Particles')
def load(filename):
    # Determine whether input snapshot is in standard or GADGET2 2 format
    # by searching for a HEAD identifier.
    input_type = 'standard'
    with open(filename, 'rb') as f:
        try:
            f.seek(4)
            if struct.unpack('4s',
                             f.read(struct.calcsize('4s')))[0] == b'HEAD':
                input_type = 'GADGET 2'
        except:
            pass
    # Dispatches the work to the appropriate function
    if input_type == 'standard':
        particles = load_standard(filename)
    elif input_type == 'GADGET 2':
        particles = load_gadget(filename)
    # Scatter particles to the correct domain-specific process.
    # Setting reset_indices_send == True ensures that buffers will be reset
    # afterwards, as this initial exchange is not representable for those
    # to come.
    exchange(particles, reset_buffers=True)
    return particles


@cython.cclass
class Gadget_snapshot:
    """
    """

    # Initialization method.
    # Note that data attributes are declared in the .pxd file.
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    def __init__(self):
        self.header = {}

    # This method populate the snapshot with particle data as well as ID's
    # (which are not used by this code) and additional header information.
    @cython.cfunc
    @cython.inline
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   particles='Particles',
                   a='double',
                   # Locals
                   N_locals='size_t[::1]',
                   h='double',
                   start_local='size_t',
                   unit='double',
                   )
    def populate(self, particles, a):
        """The following header fields depend on the particles:
            Npart, Massarr, Nall.
        The following header fields depend on the current time:
            Time, Redshift.
        The following header fields correspond to the parameters
        used in the current run:
            BoxSize, Omega0, OmegaLambda, HubbleParam.
        All other fields get generic values.
        """
        # The particle data
        self.particles = particles
        # The ID's of the local particles, generated such that the process
        # with the lowest rank has the lowest ID's.
        N_locals = empty(nprocs, dtype='uintp')
        Allgather(array(particles.N_local, dtype='uintp'), N_locals)
        start_local = sum(N_locals[:rank])
        # In pure Python, the index must be a Python integer
        if not cython.compiled:
            start_local = int(start_local)
        self.ID = arange(start_local, start_local + particles.N_local,
                         dtype='uint32')
        # The header data
        self.header['Npart'] = [0, particles.N, 0, 0, 0, 0]
        h = H0/(100*units.km/units.s/units.Mpc)
        unit = 1e+10*units.m_sun/h
        self.header['Massarr'] = [0.0, particles.mass/unit, 0.0, 0.0, 0.0, 0.0]
        self.header['Time'] = a  # Note that "Time" is really the scale factor
        self.header['Redshift'] = 1/a - 1
        self.header['FlagSfr'] = 0
        self.header['FlagFeedback'] = 0
        self.header['Nall'] = [0, particles.N, 0, 0, 0, 0]
        self.header['FlagCooling'] = 0
        self.header['Numfiles'] = 1
        unit = units.kpc/h
        self.header['BoxSize'] = boxsize/unit
        self.header['Omega0'] = Ωm
        self.header['OmegaLambda'] = ΩΛ
        self.header['HubbleParam'] = h
        self.header['FlagAge'] = 0
        self.header['FlagMetals'] = 0
        self.header['NallHW'] = [0]*6
        self.header['flag_entr_ics'] = 1

    # Method for saving a GADGET snapshot of type 2 to disk
    @cython.cfunc
    @cython.inline
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   filename='str',
                   # Locals
                   i='int',
                   unit='double',
                   )
    def save(self, filename):
        """The snapshot data (positions and velocities) are stored in single
        precision. Only GADGET type 1 (halo) particles, corresponding to
        dark matter particles, are supported.
        """
        N = self.header['Nall'][1]
        N_local = self.particles.N_local
        # The master process write the HEAD block
        if master:
            with open(filename, 'wb') as f:
                # 8 = 4*1 + 4 = 4*sizeof(s) + sizeof(i)
                f.write(struct.pack('i', 8))
                f.write(struct.pack('4s', b'HEAD'))
                # sizeof(i) + 256 + sizeof(i)
                f.write(struct.pack('i', 4 + 256 + 4))
                f.write(struct.pack('i', 8))
                f.write(struct.pack('i', 256))
                f.write(struct.pack('6I', *self.header['Npart']))
                f.write(struct.pack('6d', *self.header['Massarr']))
                f.write(struct.pack('d', self.header['Time']))
                f.write(struct.pack('d', self.header['Redshift']))
                f.write(struct.pack('i', self.header['FlagSfr']))
                f.write(struct.pack('i', self.header['FlagFeedback']))
                f.write(struct.pack('6i', *self.header['Nall']))
                f.write(struct.pack('i', self.header['FlagCooling']))
                f.write(struct.pack('i', self.header['Numfiles']))
                f.write(struct.pack('d', self.header['BoxSize']))
                f.write(struct.pack('d', self.header['Omega0']))
                f.write(struct.pack('d', self.header['OmegaLambda']))
                f.write(struct.pack('d', self.header['HubbleParam']))
                f.write(struct.pack('i', self.header['FlagAge']))
                f.write(struct.pack('i', self.header['FlagMetals']))
                f.write(struct.pack('6i', *self.header['NallHW']))
                f.write(struct.pack('i', self.header['flag_entr_ics']))
                # Padding to fill out the 256 bytes
                f.write(struct.pack('60s', b' '*60))
                f.write(struct.pack('i', 256))
        # Write the POS block in serial, one process at a time
        unit = units.kpc/self.header['HubbleParam']
        for i in range(nprocs):
            Barrier()
            if i == rank:
                with open(filename, 'ab') as f:
                    # The identifier
                    if i == 0:
                        # 8 = 4*1 + 4 = 4*sizeof(s) + sizeof(i)
                        f.write(struct.pack('i', 8))
                        f.write(struct.pack('4s', b'POS '))
                        # sizeof(i) + 3*N*sizeof(f) + sizeof(i)
                        f.write(struct.pack('i', 4 + 3*N*4 + 4))
                        f.write(struct.pack('i', 8))
                        f.write(struct.pack('i', 3*N*4))
                    # The data
                    (asarray(np.vstack((self.particles.posx_mw[:N_local],
                                        self.particles.posy_mw[:N_local],
                                        self.particles.posz_mw[:N_local])
                                       ).T.flatten(),
                             dtype='float32')/unit).tofile(f)
                    # The closing int
                    if i == (nprocs - 1):
                        f.write(struct.pack('i', 3*N*4))
        # Write the VEL block in serial, one process at a time
        unit = units.km/units.s*self.particles.mass*self.header['Time']**1.5
        for i in range(nprocs):
            Barrier()
            if i == rank:
                with open(filename, 'ab') as f:
                    # The identifier
                    if i == 0:
                        # 8 = 4*1 + 4 = 4*sizeof(s) + sizeof(i)
                        f.write(struct.pack('i', 8))
                        f.write(struct.pack('4s', b'VEL '))
                        # sizeof(i) + 3*N*sizeof(f) + sizeof(i)
                        f.write(struct.pack('i', 4 + 3*N*4 + 4))
                        f.write(struct.pack('i', 8))
                        f.write(struct.pack('i', 3*N*4))
                    # The data
                    (asarray(np.vstack((self.particles.momx_mw[:N_local],
                                        self.particles.momy_mw[:N_local],
                                        self.particles.momz_mw[:N_local])
                                       ).T.flatten(),
                             dtype='float32')/unit).tofile(f)
                    # The closing int
                    if i == (nprocs - 1):
                        f.write(struct.pack('i', 3*N*4))
        # Write the ID block in serial, one process at a time
        for i in range(nprocs):
            Barrier()
            if i == rank:
                with open(filename, 'ab') as f:
                    # The identifier
                    if i == 0:
                        # 8 = 4*1 + 4 = 4*sizeof(s) + sizeof(i)
                        f.write(struct.pack('i', 8))
                        f.write(struct.pack('4s', b'ID  '))
                        # sizeof(i) + N*sizeof(I) + sizeof(i)
                        f.write(struct.pack('i', 4 + N*4 + 4))
                        f.write(struct.pack('i', 8))
                        f.write(struct.pack('i', N*4))
                    # The data
                    asarray(self.ID, dtype='uint32').tofile(f)
                    # The closing int
                    if i == (nprocs - 1):
                        f.write(struct.pack('i', N*4))

    # Method for loading in a GADGET snapshot of type 2 from disk
    @cython.cfunc
    @cython.inline
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   filename='str',
                   # Locals
                   N='size_t',
                   N_local='size_t',
                   N_locals='tuple',
                   file_position='size_t',
                   gadget_H0='double',
                   #gadget_a='double',
                   gadget_boxsize='double',
                   gadget_Ωm='double',
                   gadget_ΩΛ='double',
                   msg='str',
                   name='str',
                   offset='size_t',
                   size='int',
                   start_local='size_t',
                   tol='double',
                   unit='double',
                   )
    def load(self, filename):
        """ It is assumed that the snapshot on the disk is a GADGET snapshot
        of type 2 and that it uses single precision. The Gadget_snapshot
        instance stores the data (positions and velocities) in double
        precision. Only GADGET type 1 (halo) particles, corresponding to
        dark matter particles, are supported.
        """
        offset = 0
        with open(filename, 'rb') as f:
            # Read in the HEAD block. No unit conversion will be done.
            offset = self.new_block(f, offset)
            name = self.read(f, '4s').decode('utf8')  # "HEAD"
            size = self.read(f, 'i')  # 264
            offset = self.new_block(f, offset)
            self.header['Npart']         = self.read(f, '6I')
            self.header['Massarr']       = self.read(f, '6d')
            self.header['Time']          = self.read(f, 'd')
            self.header['Redshift']      = self.read(f, 'd')
            self.header['FlagSfr']       = self.read(f, 'i')
            self.header['FlagFeedback']  = self.read(f, 'i')
            self.header['Nall']          = self.read(f, '6i')
            self.header['FlagCooling']   = self.read(f, 'i')
            self.header['NumFiles']      = self.read(f, 'i')
            self.header['BoxSize']       = self.read(f, 'd')
            self.header['Omega0']        = self.read(f, 'd')
            self.header['OmegaLambda']   = self.read(f, 'd')
            self.header['HubbleParam']   = self.read(f, 'd')
            self.header['FlagAge']       = self.read(f, 'i')
            self.header['FlagMetals']    = self.read(f, 'i')
            self.header['NallHW']        = self.read(f, '6i')
            self.header['flag_entr_ics'] = self.read(f, 'i')
            # Check if the parameters of the snapshot matches those of the
            # current simulation run. Display a warning if they do not.
            tol = 1e-5
            #gadget_a = self.header['Time']
            unit = units.kpc/self.header['HubbleParam']
            gadget_boxsize = self.header['BoxSize']*unit
            unit = 100*units.km/(units.s*units.Mpc)
            gadget_H0 = self.header['HubbleParam']*unit
            gadget_Ωm = self.header['Omega0']
            gadget_ΩΛ = self.header['OmegaLambda']
            if any([abs(gadget_param/param - 1) > tol for gadget_param, param
                    in zip((gadget_boxsize, gadget_H0, gadget_Ωm,
                            gadget_ΩΛ),
                           (boxsize, H0, Ωm, ΩΛ))]):
                msg = ('Mismatch between current parameters and those in the'
                       + ' GADGET snapshot "' + filename + '":')
                #if abs(gadget_a/a_begin - 1) > tol:
                #    msg += ('\n    a_begin: ' + str(a_begin)
                #            + ' vs ' + str(gadget_a))
                if abs(gadget_boxsize/boxsize - 1) > tol:
                    msg += ('\n    boxsize: ' + str(boxsize) + ' vs '
                            + str(gadget_boxsize) + ' (kpc)')
                if abs(gadget_H0/H0 - 1) > tol:
                    msg += ('\n    H0: '
                            + str(H0/(units.km/(units.s*units.Mpc)))
                            + ' vs '
                            + str(gadget_H0/(units.km/(units.s*units.Mpc)))
                            + ' (km/s/Mpc)')
                if abs(gadget_Ωm/Ωm - 1) > tol:
                    msg += ('\n    \N{GREEK CAPITAL LETTER OMEGA}m: '
                            + str(Ωm) + ' vs ' + str(gadget_Ωm))
                if abs(gadget_ΩΛ/ΩΛ - 1) > tol:
                    msg += ('\n    \N{GREEK CAPITAL LETTER OMEGA}'
                            + '\N{GREEK CAPITAL LETTER LAMDA}: '
                            + str(ΩΛ) + ' vs ' + str(gadget_ΩΛ))
                warn(msg)
            # Compute a fair distribution of particle data to the processes
            N = self.header['Npart'][1]
            N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                        + (N//nprocs + 1, )*(N % nprocs))
            N_local = N_locals[rank]
            start_local = np.sum(N_locals[:rank])
            # In pure Python, the index must be a Python integer
            if not cython.compiled:
                start_local = int(start_local)
            # Construct a Particles instance
            unit = 1e+10*units.m_sun/self.header['HubbleParam']
            self.particles = construct('from GADGET snapshot',
                                       'dark matter',
                                       mass=self.header['Massarr'][1]*unit,
                                       N=N,
                                       )
            # Read in the POS block. The positions are given in kpc/h.
            offset = self.new_block(f, offset)
            unit = units.kpc/self.header['HubbleParam']
            name = self.read(f, '4s').decode('utf8')  # "POS "
            size = self.read(f, 'i')
            offset = self.new_block(f, offset)
            f.seek(12*start_local, 1)  # 12 = sizeof(float32)*Ndims
            file_position = f.tell()
            self.particles.populate(asarray(np.fromfile(f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [0::3], dtype='float64')
                                    *unit,
                                    'posx')
            f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [1::3], dtype='float64')
                                    *unit,
                                    'posy')
            f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [2::3], dtype='float64')
                                    *unit,
                                    'posz')
            # Read in the VEL block. The velocities are peculiar
            # velocities u=a*dx/dt divided by sqrt(a), given in km/s.
            offset = self.new_block(f, offset)
            unit = units.km/units.s*self.particles.mass*self.header['Time']**1.5
            name = self.read(f, '4s').decode('utf8')  # "VEL "
            size = self.read(f, 'i')
            offset = self.new_block(f, offset)
            f.seek(12*start_local, 1)  # 12 = sizeof(float32)*Ndims
            file_position = f.tell()
            self.particles.populate(asarray(np.fromfile(f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [0::3], dtype='float64')
                                    *unit,
                                    'momx')
            f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [1::3], dtype='float64')
                                    *unit,
                                    'momy')
            f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(f,
                                                        dtype='float32',
                                                        count=3*N_local)
                                            [2::3], dtype='float64')
                                    *unit,
                                    'momz')
            # Read in the ID block.
            # The ID's will be distributed among all processes.
            offset = self.new_block(f, offset)
            name = self.read(f, '4s').decode('utf8')  # "ID  "
            size = self.read(f, 'i')
            offset = self.new_block(f, offset)
            f.seek(4*start_local, 1)  # 4 = sizeof(unsigned int)
            file_position = f.tell()
            self.ID = np.fromfile(f, dtype='uint32', count=N_local)
            # Possible additional meta data ignored

    # Method used for reading series of bytes from the snapshot file
    @cython.cfunc
    @cython.inline
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   f='object',  # io.TextIOWrapper instance
                   fmt='str',
                   # Locals
                   t='tuple',
                   )
    def read(self, f, fmt):
        # Convert bytes to python objects and store them in a tuple
        t = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        # If the tuple contains just a single element, return this
        # element rather than the tuple.
        if len(t) == 1:
            return t[0]
        # It is nicer to use mutable lists than tuples
        return list(t)

    # Method that handles the file object's position in the snapshot file
    # during loading. Call it when the next block should be read.
    @cython.cfunc
    @cython.inline
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.locals(# Argments
                   offset='size_t',
                   )
    @cython.returns('size_t')
    def new_block(self, f, offset):
        # Set the current position in the file
        f.seek(offset)
        # Each block is bracketed with a 4-byte int
        # containing the size of the block
        offset += 8 + self.read(f, 'i')
        return offset

# Function for loading a GADGET snapshot into a Particles instance
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               filename='str',
               # Locals
               snapshot='Gadget_snapshot',
               )
@cython.returns('Particles')
def load_gadget(filename):
    # Print out message
    if master:
        print('Loading GADGET snapshot:', filename)
    snapshot = Gadget_snapshot()
    snapshot.load(filename)
    return snapshot.particles

# Function for saving the current state as a GADGET snapshot
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               a='double',
               filename='str',
               # Locals
               N='size_t',
               i='int',
               snapshot='Gadget_snapshot',
               unit='double',
               )
def save_gadget(particles, a, filename):
    # Print out message
    if master:
        print('Saving GADGET snapshot:', filename)
    # Instantiate GADGET snapshot
    snapshot = Gadget_snapshot()
    snapshot.populate(particles, a)
    # Write GADGET snapshot to disk
    snapshot.save(filename)


# Create a formated version of output_type at import time
cython.declare(output_type_fmt='str')
output_type_fmt = output_type.lower().replace(' ', '')
# If output_dir does not exist, create it
if master and not os.path.exists(output_dir):
    os.makedirs(output_dir)
