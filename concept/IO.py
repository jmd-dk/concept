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



# Class storing a standard snapshot. Besides holding methods for
# saving/loading, it stores particle data (positions, momenta, mass)
@cython.cclass
class StandardSnapshot:
    # Initialization method
    @cython.header
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the StandardSnapshot type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        str contains
        dict params
        Particles particles
        """
        # Label telling how much of the Snapshot that has been loaded
        # ('nothing', 'params', 'params and particles').
        self.contains = 'nothing'
        # Dict containing parameters of the cosmology and time as well
        # as particle attributes.
        self.params = {}
        # The actual particle data
        self.particles = None

    # Method for loading in a standard snapshot from disk
    @cython.header(# Argument
                   filename='str',
                   compare_params='bint',
                   only_params='bint',
                   # Locals
                   N_local='Py_ssize_t',
                   N_locals='tuple',
                   end_local='Py_ssize_t',
                   msg='str',
                   particle_N='Py_ssize_t',
                   particle_attribute='dict',
                   particle_mass='double',
                   particle_species='str',
                   particle_type='str',
                   snapshot_unit_length='double',
                   snapshot_unit_mass='double',
                   snapshot_unit_time='double',
                   start_local='Py_ssize_t',
                   tol='double',
                   unit='double',
                   )
    def load(self, filename, compare_params=True, only_params=False):
        # Load all particles
        with h5py.File(filename,
                       mode='r',
                       driver='mpio',
                       comm=comm,
                       ) as hdf5_file:
            # Load used base units
            snapshot_unit_length = eval(hdf5_file.attrs['unit length'], units_dict)
            snapshot_unit_time   = eval(hdf5_file.attrs['unit time'],   units_dict)
            snapshot_unit_mass   = eval(hdf5_file.attrs['unit mass'],   units_dict)
            # Load global attributes
            self.params['a']       = hdf5_file.attrs['a']
            self.params['boxsize'] = hdf5_file.attrs['boxsize']*snapshot_unit_length
            self.params['H0']      = hdf5_file.attrs['H0']*(1/snapshot_unit_time)
            self.params['Ωm']      = hdf5_file.attrs[unicode('Ω') + 'm']
            self.params['ΩΛ']      = hdf5_file.attrs[unicode('Ω') + unicode('Λ')]
            # Check if the parameters of the snapshot
            # matches those of the current simulation run.
            # Display a warning if they do not.
            if compare_params:
                tol = 1e-4
                msg = ''
                if np.abs(self.params['a']/a_begin - 1) > tol:
                    msg += ('\n' + ' '*8 + 'a_begin: {} vs {}'
                            ).format(a_begin, self.params['a'])
                if np.abs(self.params['boxsize']/boxsize - 1) > tol:
                    msg += ('\n' + ' '*8 + 'boxsize: {} vs {} ({})').format(boxsize,
                                                                            self.params['boxsize'],
                                                                            units.length)
                if np.abs(self.params['H0']/H0 - 1) > tol:
                    unit = units.km/(units.s*units.Mpc)
                    msg += ('\n' + ' '*8 + 'H0: {} vs {} ({})').format(H0/unit,
                                                                       self.params['H0']/unit,
                                                                       'km s⁻¹ Mpc⁻¹')
                if np.abs(self.params['Ωm']/Ωm - 1) > tol:
                    msg += ('\n' + ' '*8 + unicode('Ω') + 'm: {} vs {}').format(Ωm,
                                                                                self.params['Ωm'])
                if np.abs(self.params['ΩΛ']/ΩΛ - 1) > tol:
                    msg += ('\n' + ' '*8 + unicode('Ω')
                                         + unicode('Λ') + ': {} vs {}').format(ΩΛ,
                                                                               self.params['ΩΛ'])
                if msg:
                    msg = ('Mismatch between current parameters and those in'
                           + 'the snapshot "{}":{}').format(filename, msg)
                    masterwarn(msg, indent=4)
            # Initialize the particle_attributes dict
            self.params['particle_attributes'] = {}
            # Load particle data
            all_particles = hdf5_file['particles']
            for particle_type in all_particles:
                # Load particle attributes
                particles_h5 = all_particles[particle_type]
                particle_N = particles_h5['posx'].size
                particle_mass = particles_h5.attrs['mass']*snapshot_unit_mass
                particle_species = particles_h5.attrs['species']
                # The keys in the particle_attributes dict are the
                # particle types, and the values are new dicts,
                # containing the information for each type.
                self.params['particle_attributes'][particle_type] = {}
                particle_attribute = self.params['particle_attributes'][particle_type]
                particle_attribute['N']       = particle_N
                particle_attribute['mass']    = particle_mass
                particle_attribute['species'] = particle_species
                # Done loading particle attributes
                if only_params:
                    continue
                # Write out progress message
                masterprint('Loading', particle_N, particle_type,
                            '({}) ...'.format(particle_species),
                            indent=4)
                # Extract HDF5 datasets
                posx_h5 = particles_h5['posx']
                posy_h5 = particles_h5['posy']
                posz_h5 = particles_h5['posz']
                momx_h5 = particles_h5['momx']
                momy_h5 = particles_h5['momy']
                momz_h5 = particles_h5['momz']
                # Compute a fair distribution of 
                # particle data to the processes.
                N_locals = ((particle_N//nprocs, )
                            *(nprocs - (particle_N % nprocs))
                            + (particle_N//nprocs + 1, )*(particle_N % nprocs))
                N_local = N_locals[rank]
                start_local = np.sum(N_locals[:rank], dtype=C2np['Py_ssize_t'])
                end_local = start_local + N_local
                # Construct a Particles instance
                self.particles = construct(particle_type,
                                           particle_species,
                                           mass=particle_mass,
                                           N=particle_N)
                # Populate the Particles instance with data from the file
                self.particles.populate(posx_h5[start_local:end_local], 'posx')
                self.particles.populate(posy_h5[start_local:end_local], 'posy')
                self.particles.populate(posz_h5[start_local:end_local], 'posz')
                self.particles.populate(momx_h5[start_local:end_local], 'momx')
                self.particles.populate(momy_h5[start_local:end_local], 'momy')
                self.particles.populate(momz_h5[start_local:end_local], 'momz')
                # If the snapshot and the current run uses different
                # systems of units, mulitply the particle positions
                # and momenta by the snapshot units.
                if snapshot_unit_length != 1:
                    self.particles.posx_mv = asarray(self.particles.posx_mv)*snapshot_unit_length
                    self.particles.posy_mv = asarray(self.particles.posy_mv)*snapshot_unit_length
                    self.particles.posz_mv = asarray(self.particles.posz_mv)*snapshot_unit_length
                unit = snapshot_unit_length/snapshot_unit_time*snapshot_unit_mass
                if unit != 1:
                    self.particles.momx_mv = asarray(self.particles.momx_mv)*unit
                    self.particles.momy_mv = asarray(self.particles.momy_mv)*unit
                    self.particles.momz_mv = asarray(self.particles.momz_mv)*unit
                # Finalize progress message
                masterprint('done')
        # Update the "contains" string
        if only_params:
            self.contains = 'params'
        else:
            self.contains = 'params and particles'
        # Scatter particles to the correct domain-specific process.
        # Setting reset_indices_send == True ensures that buffers
        # will be reset afterwards, as this initial exchange is not
        # representable for those to come.
        if 'particles' in self.contains:
            exchange(self.particles, reset_buffers=True)

    # This method populate the snapshot with particle data
    # and additional parameters.
    @cython.header(# Arguments
                   particles='Particles',
                   a='double',
                   params='dict',
                   )
    def populate(self, particles, a, params=None):
        # Populate snapshot with the passed scalefactor and global parameters
        self.params['a']       = a
        self.params['boxsize'] = boxsize
        self.params['H0']      = H0
        self.params['Ωm']      = Ωm
        self.params['ΩΛ']      = ΩΛ
        # Pupulate snapshot with the particles
        self.particles = particles
        # Overwrite parameters with those from the passed params dict
        if params:
            for key, val in params.items():
                self.params[key] = val
        # Update the "contains" string
        self.contains = 'params and particles'
           
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
        with h5py.File(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
            # Save used base units
            hdf5_file.attrs['unit length'] = units.length
            hdf5_file.attrs['unit time']   = units.time
            hdf5_file.attrs['unit mass']   = units.mass
            # Save global attributes
            hdf5_file.attrs['H0']                        = self.params['H0']
            hdf5_file.attrs['a']                         = self.params['a']
            hdf5_file.attrs['boxsize']                   = self.params['boxsize']
            hdf5_file.attrs[unicode('Ω') + 'm']          = self.params['Ωm']
            hdf5_file.attrs[unicode('Ω') + unicode('Λ')] = self.params['ΩΛ']
            # Create HDF5 group and datasets
            N = self.particles.N
            particles_h5 = hdf5_file.create_group('particles/' + self.particles.type)
            posx_h5 = particles_h5.create_dataset('posx', [N], dtype=C2np['double'])
            posy_h5 = particles_h5.create_dataset('posy', [N], dtype=C2np['double'])
            posz_h5 = particles_h5.create_dataset('posz', [N], dtype=C2np['double'])
            momx_h5 = particles_h5.create_dataset('momx', [N], dtype=C2np['double'])
            momy_h5 = particles_h5.create_dataset('momy', [N], dtype=C2np['double'])
            momz_h5 = particles_h5.create_dataset('momz', [N], dtype=C2np['double'])
            # Get local indices of the particle data
            N_local = self.particles.N_local
            N_locals = empty(nprocs, dtype=C2np['Py_ssize_t'])
            Allgather(array(N_local, dtype=C2np['Py_ssize_t']), N_locals)
            start_local = sum(N_locals[:rank])
            end_local = start_local + N_local
            # Save the local slices of the particle data and the attributes
            posx_h5[start_local:end_local] = self.particles.posx_mv[:N_local]
            posy_h5[start_local:end_local] = self.particles.posy_mv[:N_local]
            posz_h5[start_local:end_local] = self.particles.posz_mv[:N_local]
            momx_h5[start_local:end_local] = self.particles.momx_mv[:N_local]
            momy_h5[start_local:end_local] = self.particles.momy_mv[:N_local]
            momz_h5[start_local:end_local] = self.particles.momz_mv[:N_local]
            particles_h5.attrs['mass']     = self.particles.mass
            particles_h5.attrs['species']  = self.particles.species
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
        # for the data attributes of the GadgetSnapshot type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        unsigned int[::1] ID
        str contains
        dict params
        Particles particles
        """
        # The ID of each particle (not used by the CO𝘕CEPT code)
        self.ID = None
        # Label telling how much of the Snapshot that has been loaded
        # ('nothing', 'params', 'params and particles').
        self.contains = 'nothing'
        # Dict containing all the fields of the Gadget header
        self.params = {}
        # The actual particle data
        self.particles = None

    # Method for loading in a GADGET snapshot of type 2 from disk
    @cython.header(# Arguments
                   filename='str',
                   compare_params='bint',
                   only_params='bint',
                   # Locals
                   N='Py_ssize_t',
                   N_local='Py_ssize_t',
                   N_locals='tuple',
                   file_position='Py_ssize_t',
                   mass='double',
                   msg='str',
                   name='str',
                   offset='Py_ssize_t',
                   particle_attribute='dict',
                   particle_species='str',
                   particle_type='str',
                   size='int',
                   start_local='Py_ssize_t',
                   tol='double',
                   unit='double',
                   )
    def load(self, filename, compare_params=True, only_params=False):
        """ It is assumed that the snapshot on the disk is a GADGET
        snapshot of type 2 and that it uses single precision. The
        GadgetSnapshot instance stores the data (positions and
        velocities) in double precision. Only GADGET type 1 (halo)
        particles, corresponding to dark matter particles,
        are supported.
        """
        # Only type 1 (halo) particles are supported
        particle_species = 'dark matter'
        particle_type = 'GADGET halos'
        # Read in the snapshot
        offset = 0
        with open(filename, 'rb') as f:
            # Read the HEAD block into a params['header'] dict.
            # No unit conversion will be done.
            offset = self.new_block(f, offset)
            name = self.read(f, '4s').decode('utf8')  # "HEAD"
            size = self.read(f, 'i')  # 264
            offset = self.new_block(f, offset)
            self.params['header'] = collections.OrderedDict()
            header = self.params['header']
            header['Npart']         = self.read(f, '6I')
            header['Massarr']       = self.read(f, '6d')
            header['Time']          = self.read(f, 'd')
            header['Redshift']      = self.read(f, 'd')
            header['FlagSfr']       = self.read(f, 'i')
            header['FlagFeedback']  = self.read(f, 'i')
            header['Nall']          = self.read(f, '6i')
            header['FlagCooling']   = self.read(f, 'i')
            header['NumFiles']      = self.read(f, 'i')
            header['BoxSize']       = self.read(f, 'd')
            header['Omega0']        = self.read(f, 'd')
            header['OmegaLambda']   = self.read(f, 'd')
            header['HubbleParam']   = self.read(f, 'd')
            header['FlagAge']       = self.read(f, 'i')
            header['FlagMetals']    = self.read(f, 'i')
            header['NallHW']        = self.read(f, '6i')
            header['flag_entr_ics'] = self.read(f, 'i')
            # Also include some of the header fields as parameters
            # directly in the params dict.
            self.params['a']       = header['Time']
            unit = units.kpc/header['HubbleParam']
            self.params['boxsize'] = header['BoxSize']*unit
            unit = 100*units.km/(units.s*units.Mpc)
            self.params['H0']      = header['HubbleParam']*unit
            self.params['Ωm']      = header['Omega0']
            self.params['ΩΛ']      = header['OmegaLambda']
            # Check if the parameters of the snapshot matches
            # those of the current simulation run. Display a warning
            # if they do not.
            if compare_params:
                tol = 1e-4
                msg = ''
                if np.abs(self.params['a']/a_begin - 1) > tol:
                    msg += '\n' + ' '*8 + 'a_begin: {} vs {}'.format(a_begin, self.params['a'])
                if np.abs(self.params['boxsize']/boxsize - 1) > tol:
                    msg += ('\n' + ' '*8 + 'boxsize: {} vs {} ({})').format(boxsize,
                                                                            self.params['boxsize'],
                                                                            units.length)
                if np.abs(self.params['H0']/H0 - 1) > tol:
                    unit = units.km/(units.s*units.Mpc)
                    msg += ('\n' + ' '*8 + 'H0: {} vs {} ({})').format(H0/unit,
                                                                       self.params['H0']/unit,
                                                                       'km s⁻¹ Mpc⁻¹')
                if np.abs(self.params['Ωm']/Ωm - 1) > tol:
                    msg += ('\n' + ' '*8 + unicode('Ω') + 'm: {} vs {}').format(Ωm,
                                                                                self.params['Ωm'])
                if np.abs(self.params['ΩΛ']/ΩΛ - 1) > tol:
                    msg += ('\n' + ' '*8 + unicode('Ω')
                                         + unicode('Λ') + '{} vs {}').format(ΩΛ,
                                                                             self.params['ΩΛ'])
                if msg:
                    msg = ('Mismatch between current parameters and those in '
                           + 'the GADGET snapshot '
                           + '"{}":{}').format(filename, msg)
                    masterwarn(msg, indent=4)
            # Initialize the particle_attributes dict
            self.params['particle_attributes'] = {}
            # The keys in the particle_attributes dict are the
            # particle types, and the values are new dicts,
            # containing the information for each type.
            self.params['particle_attributes'][particle_type] = {}
            particle_attribute = (self.params['particle_attributes']
                                             [particle_type])
            N = header['Npart'][1]
            particle_attribute['N'] = N
            unit = 1e+10*units.m_sun/header['HubbleParam']
            mass = header['Massarr'][1]*unit
            particle_attribute['mass'] = mass
            particle_attribute['species'] = particle_species
            # Done loading particle attributes
            if only_params:
                self.contains = 'params'
                return
            # Write out progress message
            masterprint('Loading', N, particle_type, '({}) ...'.format(particle_species), indent=4)
            # Compute a fair distribution
            # of particle data to the processes.
            N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                        + (N//nprocs + 1, )*(N % nprocs))
            N_local = N_locals[rank]
            start_local = np.sum(N_locals[:rank], dtype=C2np['Py_ssize_t'])
            # Construct a Particles instance
            self.particles = construct(particle_type, particle_species, mass=mass, N=N)
            # Read in the POS block. The positions are given in kpc/h.
            offset = self.new_block(f, offset)
            unit = units.kpc/header['HubbleParam']
            name = self.read(f, '4s').decode('utf8')  # "POS "
            size = self.read(f, 'i')
            offset = self.new_block(f, offset)
            f.seek(12*start_local, 1)  # 12 = sizeof(float)*Ndims
            file_position = f.tell()
            self.particles.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [0::3], dtype=C2np['double'])*unit,
                                    'posx')
            f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [1::3], dtype=C2np['double'])*unit,
                                    'posy')
            f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [2::3], dtype=C2np['double'])*unit,
                                    'posz')
            # Read in the VEL block. The velocities are peculiar
            # velocities u=a*dx/dt divided by sqrt(a), given in km/s.
            offset = self.new_block(f, offset)
            unit = units.km/units.s*mass*header['Time']**1.5
            name = self.read(f, '4s').decode('utf8')  # "VEL "
            size = self.read(f, 'i')
            offset = self.new_block(f, offset)
            f.seek(12*start_local, 1)  # 12 = sizeof(float)*Ndims
            file_position = f.tell()
            self.particles.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [0::3], dtype=C2np['double'])*unit,
                                    'momx')
            f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [1::3], dtype=C2np['double'])*unit,
                                    'momy')
            f.seek(file_position)
            self.particles.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [2::3], dtype=C2np['double'])*unit,
                                    'momz')
            # Read in the ID block.
            # The ID's will be distributed among all processes.
            offset = self.new_block(f, offset)
            name = self.read(f, '4s').decode('utf8')  # "ID  "
            size = self.read(f, 'i')
            offset = self.new_block(f, offset)
            f.seek(4*start_local, 1)  # 4 = sizeof(unsigned int)
            file_position = f.tell()
            self.ID = np.fromfile(f, dtype=C2np['unsigned int'], count=N_local)
            # Finalize progress message
            masterprint('done')
            # Possible additional meta data ignored
        # Update the "contains" string
        self.contains = 'params and particles'
        # Scatter particles to the correct domain-specific process.
        # Setting reset_indices_send == True ensures that buffers
        # will be reset afterwards, as this initial exchange is not
        # representable for those to come.
        exchange(self.particles, reset_buffers=True)


    # Method that handles the file object's position in the snapshot
    # file during loading. Call it when the next block should be read.
    @cython.header(offset='Py_ssize_t',
                   f='object',
                   returns='Py_ssize_t')
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
        self.ID = arange(start_local, start_local + particles.N_local, dtype=C2np['unsigned int'])
        # The header data
        self.params['header'] = collections.OrderedDict()
        header = self.params['header']
        header['Npart'] = [0, particles.N, 0, 0, 0, 0]
        h = H0/(100*units.km/(units.s*units.Mpc))
        unit = 1e+10*units.m_sun/h
        header['Massarr']       = [0.0, particles.mass/unit] + [0.0]*4
        header['Time']          = a
        header['Redshift']      = 1/a - 1
        header['FlagSfr']       = 0
        header['FlagFeedback']  = 0
        header['Nall']          = [0, particles.N, 0, 0, 0, 0]
        header['FlagCooling']   = 0
        header['Numfiles']      = 1
        unit = units.kpc/h
        header['BoxSize']       = boxsize/unit
        header['Omega0']        = Ωm
        header['OmegaLambda']   = ΩΛ
        header['HubbleParam']   = h
        header['FlagAge']       = 0
        header['FlagMetals']    = 0
        header['NallHW']        = [0]*6
        header['flag_entr_ics'] = 1

    # Method used for reading series of bytes from the snapshot file
    @cython.header(# Arguments
                   f='object',
                   fmt='str',
                   # Locals
                   t='tuple',
                   returns='object',
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
                   N='Py_ssize_t',
                   N_local='Py_ssize_t',
                   i='int',
                   unit='double',
                   )
    def save(self, filename):
        """The snapshot data (positions and velocities) are stored in
        single precision. Only GADGET type 1 (halo) particles,
        corresponding to dark matter particles, are supported.
        """
        masterprint('Saving GADGET snapshot "{}" ...'.format(filename))
        N = self.particles.N
        N_local = self.particles.N_local
        header = self.params['header']
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
                f.write(struct.pack('6I', *header['Npart']))
                f.write(struct.pack('6d', *header['Massarr']))
                f.write(struct.pack('d',   header['Time']))
                f.write(struct.pack('d',   header['Redshift']))
                f.write(struct.pack('i',   header['FlagSfr']))
                f.write(struct.pack('i',   header['FlagFeedback']))
                f.write(struct.pack('6i', *header['Nall']))
                f.write(struct.pack('i',   header['FlagCooling']))
                f.write(struct.pack('i',   header['Numfiles']))
                f.write(struct.pack('d',   header['BoxSize']))
                f.write(struct.pack('d',   header['Omega0']))
                f.write(struct.pack('d',   header['OmegaLambda']))
                f.write(struct.pack('d',   header['HubbleParam']))
                f.write(struct.pack('i',   header['FlagAge']))
                f.write(struct.pack('i',   header['FlagMetals']))
                f.write(struct.pack('6i', *header['NallHW']))
                f.write(struct.pack('i',   header['flag_entr_ics']))
                # Padding to fill out the 256 bytes
                f.write(struct.pack('60s', b' '*60))
                f.write(struct.pack('i', 256))
        # Write the POS block in serial, one process at a time
        unit = units.kpc/header['HubbleParam']
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
                             dtype=C2np['float'])/unit).tofile(f)
                    # The closing int
                    if i == nprocs - 1:
                        f.write(struct.pack('i', 3*N*4))
        # Write the VEL block in serial, one process at a time
        unit = units.km/units.s*self.particles.mass*header['Time']**1.5
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
                             dtype=C2np['float'])/unit).tofile(f)
                    # The closing int
                    if i == nprocs - 1:
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
                    asarray(self.ID, dtype=C2np['unsigned int']).tofile(f)
                    # The closing int
                    if i == nprocs - 1:
                        f.write(struct.pack('i', N*4))
        masterprint('done')

# Function that loads in a standard (HDF5) snapshot. Particles in the
# snapshot are distributed fairly among all processes.
@cython.header(# Argument
               filename='str',
               compare_params='bint',
               only_params='bint',
               # Locals
               snapshot='StandardSnapshot',
               returns='StandardSnapshot',
               )
def load_standard(filename, compare_params=True, only_params=False):
    if not only_params:
        masterprint('Loading snapshot "{}"'.format(filename))
    snapshot = StandardSnapshot()
    snapshot.load(filename, compare_params=compare_params,
                            only_params=only_params)
    return snapshot

# Function that loads in particles from a standard (HDF5) snapshot
@cython.header(# Argument
               filename='str',
               compare_params='bint',
               # Locals
               snapshot='StandardSnapshot',
               returns='Particles',
               )
def load_standard_particles(filename, compare_params=True):
    snapshot = load_standard(filename, compare_params=compare_params)
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
               compare_params='bint',
               only_params='bint',
               # Locals
               snapshot='GadgetSnapshot',
               returns='GadgetSnapshot',
               )
def load_gadget(filename, compare_params=True, only_params=False):
    if not only_params:
        masterprint('Loading GADGET snapshot "{}"'.format(filename))
    snapshot = GadgetSnapshot()
    snapshot.load(filename, compare_params=compare_params,
                            only_params=only_params)
    return snapshot

# Function for loading a GADGET snapshot into a Particles instance
@cython.header(# Arguments
               filename='str',
               compare_params='bint',
               # Locals
               snapshot='GadgetSnapshot',
               returns='Particles',
               )
def load_gadget_particles(filename, compare_params=True):
    snapshot = load_gadget(filename, compare_params=compare_params)
    return snapshot.particles

# Function for saving the current state as a GADGET snapshot
@cython.header(# Arguments
               particles='Particles',
               a='double',
               filename='str',
               # Locals
               snapshot='GadgetSnapshot',
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
               compare_params='bint',
               only_params='bint',
               # Locals
               gadget_snapshot='GadgetSnapshot',
               input_type='str',
               snapshot='StandardSnapshot',
               returns='StandardSnapshot'
               )
def load_into_standard(filename, compare_params=True, only_params=False):
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
        snapshot = load_standard(filename, compare_params=compare_params,
                                           only_params=only_params)
    elif input_type == 'gadget2':
        gadget_snapshot = load_gadget(filename, compare_params=compare_params,
                                                only_params=only_params)
        # Create a corresponding standard snapshot
        snapshot = StandardSnapshot()
        snapshot.populate(gadget_snapshot.particles,
                          gadget_snapshot.params['a'],
                          gadget_snapshot.params)
    return snapshot

# Function that loads particle data from a snapshot file and
# instantiate a Particles instance on each process,
# storing the particles within its domain.
@cython.header(# Argument
               filename='str',
               compare_params='bint',
               # Locals
               snapshot='StandardSnapshot',
               returns='Particles',
               )
def load_particles(filename, compare_params=True):
    # If no snapshot should be loaded, return immediately
    if not filename:
        return
    # Load in particles from snapshot
    snapshot = load_into_standard(filename, compare_params=compare_params)
    return snapshot.particles


# Function which loads in parameters from a snapshot without loading
# in the particle data.
@cython.header(# Arguments
               filename='str',
               compare_params='bint',
               # Locals
               snapshot='StandardSnapshot',
               returns='dict',
               )
def load_params(filename, compare_params=True):
    snapshot = load_into_standard(filename, compare_params=compare_params,
                                            only_params=True)
    return snapshot.params

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
        raise Exception('Does not recognize output type "{}"'.format(snapshot_type))

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


