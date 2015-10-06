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


# Function that saves particle data to an HDF5 file or a
# gadget snapshot file, based on the snapshot_type parameter.
@cython.header(# Argument
               particles='Particles',
               a='double',
               filename='str',
               )
def save(particles, a, filename):
    if snapshot_type == 'standard':
        save_standard(particles, a, filename)
    elif snapshot_type == 'gadget2':
        save_gadget(particles, a, filename)
    elif master:
        raise Exception('Does not recognize output type "{}"'
                         .format(snapshot_type))

# Function for determining the snapshot type of a file
@cython.header(# Arguments
               filename='str',
               # Locals
               head='tuple',
               returns='str',
               )
def get_snapshot_type(filename):
    # Raise an exception if the file does not exist
    if master and not os.path.exists(filename):
        raise Exception('The snapshot file "{}" does not exist'
                         .format(filename))
    # Test for standard HDF5 format by looking up ΩΛ and particles
    try:
        with h5py.File(filename, mode='r') as f:
            f.attrs[unicode('Ω') + unicode('Λ')]
            f['particles']
            return 'standard'
    except:
        pass
    # Test for GADGET2 2 format by searching for a HEAD identifier
    try:
        with open(filename, 'rb') as f:
            f.seek(4)
            head = struct.unpack('4s', f.read(struct.calcsize('4s')))
            if head[0] == b'HEAD':
                return 'gadget2'
    except:
        pass
    # Return None if the file is not a valid snapshot
    return None
    

# Function that loads particle data from a snapshot file and
# instantiate a Particles instance on each process,
# storing the particles within its domain.
@cython.header(# Argument
               filename='str',
               write_msg='bint',
               # Locals
               particles='Particles',
               snapshot='StandardSnapshot',
               returns='Particles',
               )
def load(filename, write_msg=True):
    # If no snapshot should be loaded, return immediately
    if not filename:
        return
    # Load in particles from snapshot
    snapshot = load_into_standard(filename, write_msg)
    particles = snapshot.particles
    return particles

# Class storing a standard snapshot. Besides holding methods for
# saving/loading, it stores particle data (positions, momenta, mass)
@cython.cclass
class StandardSnapshot:
    # Initialization method
    @cython.header
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the StandardSnapshot type. It will get picked up by the
        # pyxpp script and indluded in the .pxd file.
        """
        # Data attributes
        dict params
        Particles particles
        # Methods
        load(self, str filename, bint write_msg=*)
        populate(self, Particles particles, double a, dict params_dict=*)
        save(self, str filename)
        """
        self.params = {}

    # Method for loading in a standard snapshot from disk
    @cython.header(# Argument
                   filename='str',
                   write_msg='bint',
                   # Locals
                   N='Py_ssize_t',
                   N_local='Py_ssize_t',
                   N_locals='tuple',
                   end_local='Py_ssize_t',
                   msg='str',
                   particle_type='str',
                   start_local='Py_ssize_t',
                   tol='double',
                   )
    def load(self, filename, write_msg=True):
        # Load all particles
        with h5py.File(filename,
                       mode='r',
                       driver='mpio',
                       comm=comm,
                       ) as hdf5_file:
            # Load global attributes
            self.params['a']       = hdf5_file.attrs['a']
            self.params['boxsize'] = hdf5_file.attrs['boxsize']
            self.params['H0']      = hdf5_file.attrs['H0']
            self.params['Ωm']      = hdf5_file.attrs[unicode('Ω') + 'm']
            self.params['ΩΛ']      = hdf5_file.attrs[unicode('Ω')
                                                     + unicode('Λ')]
            # Check if the parameters of the snapshot
            # matches those of the current simulation run.
            # Display a warning if they do not.
            tol = 1e-4
            if write_msg:
                msg = ''
                if np.abs(self.params['a']/a_begin - 1) > tol:
                    msg += ('\n' + ' '*8 + 'a_begin: {} vs {}'
                            ).format(a_begin, self.params['a'])
                if np.abs(self.params['boxsize']/boxsize - 1) > tol:
                    unit = units.kpc
                    msg += ('\n' + ' '*8 + 'boxsize: {} vs {} (kpc)'
                            ).format(boxsize/unit, self.params['boxsize']/unit)
                if np.abs(self.params['H0']/H0 - 1) > tol:
                    unit = units.km/(units.s*units.Mpc)
                    msg += ('\n' + ' '*8 + 'H0: {} vs {} (km/s/Mpc)'
                            ).format(H0/unit, self.params['H0']/unit)
                if np.abs(self.params['Ωm']/Ωm - 1) > tol:
                    msg += ('\n' + ' '*8 + '\N{GREEK CAPITAL LETTER OMEGA}m: '
                            + '{} vs {}').format(Ωm, self.params['Ωm'])
                if np.abs(self.params['ΩΛ']/ΩΛ - 1) > tol:
                    msg += ('\n' + ' '*8 + '\N{GREEK CAPITAL LETTER OMEGA}'
                            + '\N{GREEK CAPITAL LETTER LAMDA}: '
                            + '{} vs {}').format(ΩΛ, self.params['ΩΛ'])
                if msg:
                    msg = ('Mismatch between current parameters and those in'
                           + 'the snapshot "{}":{}').format(filename, msg)
                    masterwarn(msg, indent=4)
            # Load particle data
            all_particles = hdf5_file['particles']
            for particle_type in all_particles:
                particles_h5 = all_particles[particle_type]
                # Write out progress message
                N = particles_h5['posx'].size
                masterprint('    Loading', N, particles_h5.attrs['species'],
                            'particles', '(' + particles_h5.attrs['type']
                            + ') ...')
                # Extract HDF5 datasets
                posx_h5 = particles_h5['posx']
                posy_h5 = particles_h5['posy']
                posz_h5 = particles_h5['posz']
                momx_h5 = particles_h5['momx']
                momy_h5 = particles_h5['momy']
                momz_h5 = particles_h5['momz']
                # Compute a fair distribution of 
                # particle data to the processes.
                N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                            + (N//nprocs + 1, )*(N % nprocs))
                N_local = N_locals[rank]
                start_local = np.sum(N_locals[:rank], dtype=C2np['Py_ssize_t'])
                end_local = start_local + N_local
                # In pure Python, the indices must be Python integers
                if not cython.compiled:
                    start_local = int(start_local)
                    end_local = int(end_local)
                # Construct a Particles instance
                self.particles = construct(particles_h5.attrs['type'],
                                           particles_h5.attrs['species'],
                                           mass=particles_h5.attrs['mass'],
                                           N=N)
                # Populate the Particles instance with data from the file
                self.particles.populate(posx_h5[start_local:end_local], 'posx')
                self.particles.populate(posy_h5[start_local:end_local], 'posy')
                self.particles.populate(posz_h5[start_local:end_local], 'posz')
                self.particles.populate(momx_h5[start_local:end_local], 'momx')
                self.particles.populate(momy_h5[start_local:end_local], 'momy')
                self.particles.populate(momz_h5[start_local:end_local], 'momz')
                # Finalize progress message
                masterprint('done')
        # Scatter particles to the correct domain-specific process.
        # Setting reset_indices_send == True ensures that buffers
        # will be reset afterwards, as this initial exchange is not
        # representable for those to come.
        exchange(self.particles, reset_buffers=True)

    # This method populate the snapshot with particle data
    # and additional parameters.
    @cython.header(# Arguments
                   particles='Particles',
                   a='double',
                   params_dict='dict',
                   )
    def populate(self, particles, a, params_dict=None):
        # Populate snapshot with the passed scalefactor and global parameters
        self.params['a']       = a
        self.params['boxsize'] = boxsize
        self.params['H0']      = H0
        self.params['Ωm']      = Ωm
        self.params['ΩΛ']      = ΩΛ
        self.particles = particles
        # Overwrite parameters with those from params_dict
        if params_dict:
            for key, val in params_dict.items():
                self.params[key] = val
           
    # Methd that the snapshot to an hdf5 file
    @cython.header(# Argument
                   filename='str',
                   # Locals
                   N='Py_ssize_t',
                   N_local='Py_ssize_t',
                   N_locals='Py_ssize_t[::1]',
                   end_local='Py_ssize_t',
                   start_local='Py_ssize_t',
                   )
    def save(self, filename):
        # Print out message
        masterprint('Saving snapshot "{}" ...'.format(filename))
        with h5py.File(filename,
                       mode='w',
                       driver='mpio',
                       comm=comm,
                       ) as hdf5_file:
            # Save global attributes
            hdf5_file.attrs['H0'] = self.params['H0']
            hdf5_file.attrs['a'] = self.params['a']
            hdf5_file.attrs['boxsize'] = self.params['boxsize']
            hdf5_file.attrs[unicode('Ω') + 'm'] = self.params['Ωm']
            hdf5_file.attrs[unicode('Ω') + unicode('Λ')] = self.params['ΩΛ']
            # Create HDF5 group and datasets
            N = self.particles.N
            particles_h5 = hdf5_file.create_group('particles/'
                                                  + self.particles.type)
            posx_h5 = particles_h5.create_dataset('posx', [N], dtype='float64')
            posy_h5 = particles_h5.create_dataset('posy', [N], dtype='float64')
            posz_h5 = particles_h5.create_dataset('posz', [N], dtype='float64')
            momx_h5 = particles_h5.create_dataset('momx', [N], dtype='float64')
            momy_h5 = particles_h5.create_dataset('momy', [N], dtype='float64')
            momz_h5 = particles_h5.create_dataset('momz', [N], dtype='float64')
            # Get local indices of the particle data
            N_local = self.particles.N_local
            N_locals = empty(nprocs, dtype=C2np['Py_ssize_t'])
            Allgather(array(N_local, dtype=C2np['Py_ssize_t']), N_locals)
            start_local = sum(N_locals[:rank])
            end_local = start_local + N_local
            # In pure Python, the indices needs to be Python integers
            if not cython.compiled:
                start_local = int(start_local)
                end_local = int(end_local)
            # Save the local slices of the particle data and the attributes
            posx_h5[start_local:end_local] = self.particles.posx_mv[:N_local]
            posy_h5[start_local:end_local] = self.particles.posy_mv[:N_local]
            posz_h5[start_local:end_local] = self.particles.posz_mv[:N_local]
            momx_h5[start_local:end_local] = self.particles.momx_mv[:N_local]
            momy_h5[start_local:end_local] = self.particles.momy_mv[:N_local]
            momz_h5[start_local:end_local] = self.particles.momz_mv[:N_local]
            particles_h5.attrs['mass']     = self.particles.mass
            particles_h5.attrs['species']  = self.particles.species
            particles_h5.attrs['type']     = self.particles.type
        masterprint('done')

# Class storing a Gadget snapshot. Besides holding methods for
# saving/loading, it stores particle data (positions, momenta, mass)
# and also Gadget ID's and the Gadget header.
@cython.cclass
class GadgetSnapshot:
    """Only Gadget type 1 (halo) particles, corresponding to dark matter
    particles, are supported.
    """

    # Initialization method.
    @cython.header
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the GadgetSnapshot type. It will get picked up by the
        # pyxpp script and indluded in the .pxd file.
        """
        # Data attributes
        dict header
        Particles particles
        unsigned int[::1] ID
        # Methods (f is an io.TextIOWrapper instance)
        load(self, str filename, bint write_msg=*)
        Py_ssize_t new_block(self, object f, Py_ssize_t offset)
        populate(self, Particles particles, double a)
        object read(self, object f, str fmt)  
        save(self, str filename)
        """
        self.header = {}

    # Method for loading in a GADGET snapshot of type 2 from disk
    @cython.header(# Arguments
                   filename='str',
                   write_msg='bint',
                   # Locals
                   N='Py_ssize_t',
                   N_local='Py_ssize_t',
                   N_locals='tuple',
                   file_position='Py_ssize_t',
                   gadget_H0='double',
                   gadget_a='double',
                   gadget_boxsize='double',
                   gadget_Ωm='double',
                   gadget_ΩΛ='double',
                   msg='str',
                   name='str',
                   offset='Py_ssize_t',
                   size='int',
                   start_local='Py_ssize_t',
                   tol='double',
                   unit='double',
                   )
    def load(self, filename, write_msg=True):
        """ It is assumed that the snapshot on the disk is a GADGET
        snapshot of type 2 and that it uses single precision. The
        GadgetSnapshot instance stores the data (positions and
        velocities) in double precision. Only GADGET type 1 (halo)
        particles, corresponding to dark matter particles,
        are supported.
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
            # Check if the parameters of the snapshot matches
            # those of the current simulation run. Display a warning
            # if they do not.
            tol = 1e-4
            gadget_a = self.header['Time']
            unit = units.kpc/self.header['HubbleParam']
            gadget_boxsize = self.header['BoxSize']*unit
            unit = 100*units.km/(units.s*units.Mpc)
            gadget_H0 = self.header['HubbleParam']*unit
            gadget_Ωm = self.header['Omega0']
            gadget_ΩΛ = self.header['OmegaLambda']
            if write_msg:
                msg = ''
                if abs(gadget_a/a_begin - 1) > tol:
                    msg += ('\n' + ' '*8 + 'a_begin: {} vs {}'
                            ).format(a_begin, gadget_a)
                if abs(gadget_boxsize/boxsize - 1) > tol:
                    unit = units.kpc
                    msg += ('\n' + ' '*8 + 'boxsize: {} vs {} (kpc)'
                            ).format(boxsize/unit, gadget_boxsize/unit)
                if abs(gadget_H0/H0 - 1) > tol:
                    unit = units.km/(units.s*units.Mpc)
                    msg += ('\n' + ' '*8 + 'H0: {} vs {} (km/s/Mpc)'
                            ).format(H0/unit, gadget_H0/unit)
                if abs(gadget_Ωm/Ωm - 1) > tol:
                    msg += ('\n' + ' '*8 + '\N{GREEK CAPITAL LETTER OMEGA}m: '
                            + '{} vs {}').format(Ωm, gadget_Ωm)
                if abs(gadget_ΩΛ/ΩΛ - 1) > tol:
                    msg += ('\n' + ' '*8 + '\N{GREEK CAPITAL LETTER OMEGA}'
                            + '\N{GREEK CAPITAL LETTER LAMDA}: '
                            + '{} vs {}').format(ΩΛ, gadget_ΩΛ)
                if msg:
                    msg = ('Mismatch between current parameters and those in '
                           + 'the GADGET snapshot '
                           + '"{}":{}').format(filename, msg)
                    masterwarn(msg, indent=4)
            # Write out progress message
            N = self.header['Npart'][1]
            masterprint('    Loading', N, 'dark matter particles',
                        '(GADGET halos) ...')
            # Compute a fair distribution
            # of particle data to the processes.
            N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                        + (N//nprocs + 1, )*(N % nprocs))
            N_local = N_locals[rank]
            start_local = np.sum(N_locals[:rank], dtype=C2np['Py_ssize_t'])
            # In pure Python, the index must be a Python integer
            if not cython.compiled:
                start_local = int(start_local)
            # Construct a Particles instance
            unit = 1e+10*units.m_sun/self.header['HubbleParam']
            self.particles = construct('Gadget halos',
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
            unit = (units.km/units.s*self.particles.mass
                    *self.header['Time']**1.5)
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
            # Finalize progress message
            masterprint('done')
            # Possible additional meta data ignored
        # Scatter particles to the correct domain-specific process.
        # Setting reset_indices_send == True ensures that buffers
        # will be reset afterwards, as this initial exchange is not
        # representable for those to come.
        exchange(self.particles, reset_buffers=True)


    # Method that handles the file object's position in the snapshot
    # file during loading. Call it when the next block should be read.
    @cython.header(offset='Py_ssize_t', returns='Py_ssize_t')
    def new_block(self, f, offset):
        # Set the current position in the file
        f.seek(offset)
        # Each block is bracketed with a 4-byte int
        # containing the size of the block
        offset += 8 + self.read(f, 'i')
        return offset

    # This method populate the snapshot with particle data
    # as well as ID's (which are not used by this code) and
    # additional header information.
    @cython.header(# Arguments
                   particles='Particles',
                   a='double',
                   # Locals
                   N_locals='Py_ssize_t[::1]',
                   h='double',
                   start_local='Py_ssize_t',
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
        # The ID's of the local particles, generated such that
        # the process with the lowest rank has the lowest ID's.
        N_locals = empty(nprocs, dtype=C2np['Py_ssize_t'])
        Allgather(array(particles.N_local, dtype=C2np['Py_ssize_t']), N_locals)
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
        self.header['Massarr']       = [0.0, particles.mass/unit] + [0.0]*4
        self.header['Time']          = a  # "Time" is really the scale factor
        self.header['Redshift']      = 1/a - 1
        self.header['FlagSfr']       = 0
        self.header['FlagFeedback']  = 0
        self.header['Nall']          = [0, particles.N, 0, 0, 0, 0]
        self.header['FlagCooling']   = 0
        self.header['Numfiles']      = 1
        unit = units.kpc/h
        self.header['BoxSize']       = boxsize/unit
        self.header['Omega0']        = Ωm
        self.header['OmegaLambda']   = ΩΛ
        self.header['HubbleParam']   = h
        self.header['FlagAge']       = 0
        self.header['FlagMetals']    = 0
        self.header['NallHW']        = [0]*6
        self.header['flag_entr_ics'] = 1

    # Method used for reading series of bytes from the snapshot file
    @cython.header(# Arguments
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

    # Method for saving a GADGET snapshot of type 2 to disk
    @cython.header(# Arguments
                   filename='str',
                   # Locals
                   i='int',
                   unit='double',
                   )
    def save(self, filename):
        """The snapshot data (positions and velocities) are stored in
        single precision. Only GADGET type 1 (halo) particles,
        corresponding to dark matter particles, are supported.
        """
        masterprint('Saving GADGET snapshot "{}" ...'.format(filename))
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
                    (asarray(np.vstack((self.particles.posx_mv[:N_local],
                                        self.particles.posy_mv[:N_local],
                                        self.particles.posz_mv[:N_local])
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
                    (asarray(np.vstack((self.particles.momx_mv[:N_local],
                                        self.particles.momy_mv[:N_local],
                                        self.particles.momz_mv[:N_local])
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
        masterprint('done')

# Function that loads in a standard (HDF5) snapshot. Particles in the
# snapshot are distriuvuted fairly among all processes.
@cython.header(# Argument
               filename='str',
               write_msg='bint',
               # Locals
               snapshot='StandardSnapshot',
               returns='StandardSnapshot',
               )
def load_standard(filename, write_msg=True):
    masterprint('Loading snapshot "{}"'.format(filename))
    snapshot = StandardSnapshot()
    snapshot.load(filename, write_msg)
    return snapshot

# Function that loads in particles from a standard (HDF5) snapshot
@cython.header(# Argument
               filename='str',
               write_msg='bint',
               # Locals
               snapshot='StandardSnapshot',
               returns='Particles',
               )
def load_standard_particles(filename, write_msg=True):
    snapshot = load_standard(filename, write_msg)
    return snapshot.particles

# Function for saving the current state as a GADGET snapshot
@cython.header(# Arguments
               particles='Particles',
               a='double',
               filename='str',
               # Locals
               snapshot='StandardSnapshot',
               )
def save_standard(particles, a, filename):
    # Instantiate snapshot
    snapshot = StandardSnapshot()
    snapshot.populate(particles, a)
    # Write snapshot to disk
    snapshot.save(filename)

# Function for loading a complete GADGET snapshot
@cython.header(# Arguments
               filename='str',
               write_msg='bint',
               # Locals
               snapshot='GadgetSnapshot',
               returns='GadgetSnapshot',
               )
def load_gadget(filename, write_msg=True):
    masterprint('Loading GADGET snapshot "{}"'.format(filename))
    snapshot = GadgetSnapshot()
    snapshot.load(filename, write_msg)
    return snapshot

# Function for loading a GADGET snapshot into a Particles instance
@cython.header(# Arguments
               filename='str',
               write_msg='bint',
               # Locals
               snapshot='GadgetSnapshot',
               returns='Particles',
               )
def load_gadget_particles(filename, write_msg=True):
    snapshot = load_gadget(filename, write_msg)
    return snapshot.particles

# Function for saving the current state as a GADGET snapshot
@cython.header(# Arguments
               particles='Particles',
               a='double',
               filename='str',
               # Locals
               N='Py_ssize_t',
               i='int',
               snapshot='GadgetSnapshot',
               unit='double',
               )
def save_gadget(particles, a, filename):
    # Instantiate GADGET snapshot
    snapshot = GadgetSnapshot()
    snapshot.populate(particles, a)
    # Write GADGET snapshot to disk
    snapshot.save(filename)

# Function which can load any snapshot,
# but always returns an instance of StandardSnapshot.
@cython.header(# Arguments
               filename='str',
               write_msg='bint',
               # Locals
               gadget_snapshot='GadgetSnapshot',
               input_type='str',
               params_dict='dict',
               snapshot='StandardSnapshot',
               unit_boxsize='double',
               unit_H0='double',
               returns='StandardSnapshot'
               )
def load_into_standard(filename, write_msg=True):
    # If no snapshot should be loaded, return immediately
    if not filename:
        return
    # Determine snapshot type
    input_type = get_snapshot_type(filename)
    if master and input_type is None:
        raise Exception(('Cannot recognize "{}" as neither a standard nor a '
                         + 'gadget2 snapshot').format(filename))
    # Dispatches the work to the appropriate function
    if input_type == 'standard':
        snapshot = load_standard(filename, write_msg)
    elif input_type == 'gadget2':
        gadget_snapshot = load_gadget(filename, write_msg)
        snapshot = StandardSnapshot()
        unit_boxsize = units.kpc/gadget_snapshot.header['HubbleParam']
        unit_H0 = 100*units.km/(units.s*units.Mpc)
        params_dict = {'a'      : gadget_snapshot.header['Time'],
                       'boxsize': (gadget_snapshot.header['BoxSize']
                                   *unit_boxsize),
                       'H0'     : (gadget_snapshot.header['HubbleParam']
                                   *unit_H0),
                       'Ωm'     : gadget_snapshot.header['Omega0'],
                       'ΩΛ'     : gadget_snapshot.header['OmegaLambda'],
                       }
        snapshot.populate(gadget_snapshot.particles,
                          params_dict['a'],
                          params_dict)
    return snapshot
