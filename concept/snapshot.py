# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2016 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
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
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
from communication import smart_mpi
cimport('from species import Component, get_representation')
cimport('from communication import domain_layout_local_indices, domain_subdivisions, exchange')

# Pure Python imports
import struct



# Class storing a standard snapshot. Besides holding methods for
# saving/loading, it stores component data.
@cython.cclass
class StandardSnapshot:
    """The params dict contain the following entries:
    'H0'
    'a'
    'boxsize'
    'Ωm'
    'ΩΛ'
    """
    # The filename extension for this type of snapshot
    extension = '.hdf5'

    # Initialization method
    @cython.header
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the StandardSnapshot type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        public dict params
        public list components
        public dict units
        """
        # Dict containing all the parameters of the snapshot
        self.params = {}
        # List of components
        self.components = []
        # Dict containing the base units in str format
        self.units = {}

    # Methd that saves the snapshot to an hdf5 file
    @cython.pheader(# Argument
                   filename='str',
                   # Locals
                   component='Component',
                   domain_end_i='Py_ssize_t',
                   domain_end_j='Py_ssize_t',
                   domain_end_k='Py_ssize_t',
                   domain_size_i='Py_ssize_t',
                   domain_size_j='Py_ssize_t',
                   domain_size_k='Py_ssize_t',
                   domain_start_i='Py_ssize_t',
                   domain_start_j='Py_ssize_t',
                   domain_start_k='Py_ssize_t',
                   end_local='Py_ssize_t',
                   shape='tuple',
                   start_local='Py_ssize_t',
                   )
    def save(self, filename):
        # Attach missing extension to filename
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        # Print out message
        masterprint('Saving standard snapshot "{}":'.format(filename))
        with h5py.File(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
            # Save used base units
            hdf5_file.attrs['unit length'] = self.units['length']
            hdf5_file.attrs['unit time']   = self.units['time']
            hdf5_file.attrs['unit mass']   = self.units['mass']
            # Save global attributes
            hdf5_file.attrs['H0']          = self.params['H0']
            hdf5_file.attrs['a']           = self.params['a']
            hdf5_file.attrs['boxsize']     = self.params['boxsize']
            hdf5_file.attrs[unicode('Ωm')] = self.params['Ωm']
            hdf5_file.attrs[unicode('ΩΛ')] = self.params['ΩΛ']
            # Store each component as a seperate group
            # within /components.
            for component in self.components:
                component_h5 = hdf5_file.create_group('components/' + component.name)
                # Save the general component attributes
                component_h5.attrs['species'] = component.species
                component_h5.attrs['mass'] = component.mass
                if component.representation == 'particles':
                    # Write out progress message
                    masterprint('Writing out {} ({} {}) ...'.format(component.name,
                                                                    component.N,
                                                                    component.species),
                                indent=4)
                    # Save particle attributes
                    component_h5.attrs['N']    = component.N
                    # Get local indices of the particle data
                    start_local = int(np.sum(smart_mpi(component.N_local,
                                                       mpifun='allgather')[:rank]))
                    end_local = start_local + component.N_local
                    # Save particle data
                    shape = (component.N, )
                    posx_h5 = component_h5.create_dataset('posx', shape, dtype=C2np['double'])
                    posy_h5 = component_h5.create_dataset('posy', shape, dtype=C2np['double'])
                    posz_h5 = component_h5.create_dataset('posz', shape, dtype=C2np['double'])
                    momx_h5 = component_h5.create_dataset('momx', shape, dtype=C2np['double'])
                    momy_h5 = component_h5.create_dataset('momy', shape, dtype=C2np['double'])
                    momz_h5 = component_h5.create_dataset('momz', shape, dtype=C2np['double'])
                    posx_h5[start_local:end_local] = component.posx_mv[:component.N_local]
                    posy_h5[start_local:end_local] = component.posy_mv[:component.N_local]
                    posz_h5[start_local:end_local] = component.posz_mv[:component.N_local]
                    momx_h5[start_local:end_local] = component.momx_mv[:component.N_local]
                    momy_h5[start_local:end_local] = component.momy_mv[:component.N_local]
                    momz_h5[start_local:end_local] = component.momz_mv[:component.N_local]
                    # Finalize progress message
                    masterprint('done')
                elif component.representation == 'fluid':
                    # Write out progress message
                    masterprint('Writing out {} ({} with gridsize {}) ...'
                                .format(component.name, component.species, component.gridsize),
                                indent=4)
                    # Save fluid attributes
                    component_h5.attrs['gridsize'] = component.gridsize
                    # Save fluid grids
                    shape = (component.gridsize, )*3
                    δ_h5  = component_h5.create_dataset(unicode('δ'), shape, dtype=C2np['double'])
                    ux_h5 = component_h5.create_dataset('ux', shape, dtype=C2np['double'])
                    uy_h5 = component_h5.create_dataset('uy', shape, dtype=C2np['double'])
                    uz_h5 = component_h5.create_dataset('uz', shape, dtype=C2np['double'])
                    domain_size_i = component.δ_mv.shape[0] - (1 + 2*2)
                    domain_size_j = component.δ_mv.shape[1] - (1 + 2*2)
                    domain_size_k = component.δ_mv.shape[2] - (1 + 2*2)
                    domain_start_i = domain_layout_local_indices[0]*domain_size_i
                    domain_start_j = domain_layout_local_indices[1]*domain_size_j
                    domain_start_k = domain_layout_local_indices[2]*domain_size_k
                    domain_end_i = domain_start_i + domain_size_i
                    domain_end_j = domain_start_j + domain_size_j
                    domain_end_k = domain_start_k + domain_size_k
                    δ_h5[domain_start_i:domain_end_i,
                         domain_start_j:domain_end_j,
                         domain_start_k:domain_end_k] = component.δ_mv[2:(2 + domain_size_i),
                                                                       2:(2 + domain_size_j),
                                                                       2:(2 + domain_size_k)]
                    ux_h5[domain_start_i:domain_end_i,
                          domain_start_j:domain_end_j,
                          domain_start_k:domain_end_k] = component.ux_mv[2:(2 + domain_size_i),
                                                                         2:(2 + domain_size_j),
                                                                         2:(2 + domain_size_k)]
                    uy_h5[domain_start_i:domain_end_i,
                          domain_start_j:domain_end_j,
                          domain_start_k:domain_end_k] = component.uy_mv[2:(2 + domain_size_i),
                                                                         2:(2 + domain_size_j),
                                                                         2:(2 + domain_size_k)]
                    uz_h5[domain_start_i:domain_end_i,
                          domain_start_j:domain_end_j,
                          domain_start_k:domain_end_k] = component.uz_mv[2:(2 + domain_size_i),
                                                                         2:(2 + domain_size_j),
                                                                         2:(2 + domain_size_k)]
                    # Finalize progress message
                    masterprint('done')
                elif master:
                    abort('Does not know how to save component "{}" with representation "{}"'
                          .format(component.name, component.representation))

    # Method for loading in a standard snapshot from disk
    @cython.pheader(# Argument
                    filename='str',
                    compare_params='bint',
                    only_params='bint',
                    do_exchange='bint',
                    # Locals
                    N_local='Py_ssize_t',
                    N_locals='tuple',
                    end_local='Py_ssize_t',
                    gridsize='Py_ssize_t',
                    N='Py_ssize_t',
                    mass='double',
                    representation='str',
                    species='str',
                    name='str',
                    component='Component',
                    snapshot_unit_length='double',
                    snapshot_unit_mass='double',
                    snapshot_unit_time='double',
                    start_local='Py_ssize_t',
                    unit='double',
                    dim='int',
                    domain_end_i='Py_ssize_t',
                    domain_end_j='Py_ssize_t',
                    domain_end_k='Py_ssize_t',
                    domain_size_i='Py_ssize_t',
                    domain_size_j='Py_ssize_t',
                    domain_size_k='Py_ssize_t',
                    domain_start_i='Py_ssize_t',
                    domain_start_j='Py_ssize_t',
                    domain_start_k='Py_ssize_t',
                    )
    def load(self, filename, compare_params=True, only_params=False, do_exchange=True):
        if not only_params:
            masterprint('Loading snapshot "{}":'.format(filename))
        # Load all components
        with h5py.File(filename, mode='r', driver='mpio', comm=comm) as hdf5_file:
            # Load used base units
            self.units['length'] = hdf5_file.attrs['unit length']
            self.units['time']   = hdf5_file.attrs['unit time']
            self.units['mass']   = hdf5_file.attrs['unit mass']
            snapshot_unit_length = eval(self.units['length'], units_dict)
            snapshot_unit_time   = eval(self.units['time'],   units_dict)
            snapshot_unit_mass   = eval(self.units['mass'],   units_dict)
            # Load global attributes
            self.params['H0']      = hdf5_file.attrs['H0']*(1/snapshot_unit_time)
            self.params['a']       = hdf5_file.attrs['a']
            self.params['boxsize'] = hdf5_file.attrs['boxsize']*snapshot_unit_length
            self.params['Ωm']      = hdf5_file.attrs[unicode('Ωm')]
            self.params['ΩΛ']      = hdf5_file.attrs[unicode('ΩΛ')]
            # Check if the parameters of the snapshot
            # matches those of the current simulation run.
            # Display a warning if they do not.
            if compare_params:
                compare_parameters(self.params, filename)
            # Load component data
            for name, component_h5 in hdf5_file['components'].items():
                # Load the general component attributes
                species = component_h5.attrs['species']
                representation = get_representation(species)
                mass = component_h5.attrs['mass']*snapshot_unit_mass
                if representation == 'particles':
                    # Construct a Component instance and append it
                    # to this snapshot's list of components.
                    N = component_h5.attrs['N']
                    component = Component(name, species, N, mass)
                    self.components.append(component)
                    # Done loading component attributes
                    if only_params:
                        continue
                    # Write out progress message
                    if not only_params:
                        masterprint('Reading in {} ({} {}) ...'.format(name, N, species), indent=4)
                    # Extract HDF5 datasets
                    posx_h5 = component_h5['posx']
                    posy_h5 = component_h5['posy']
                    posz_h5 = component_h5['posz']
                    momx_h5 = component_h5['momx']
                    momy_h5 = component_h5['momy']
                    momz_h5 = component_h5['momz']
                    # Compute a fair distribution of 
                    # particle data to the processes.
                    N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                                + (N//nprocs + 1, )*(N % nprocs))
                    N_local = N_locals[rank]
                    start_local = np.sum(N_locals[:rank], dtype=C2np['Py_ssize_t'])
                    end_local = start_local + N_local
                    # Populate the Component instance with data from the file
                    component.populate(posx_h5[start_local:end_local], 'posx')
                    component.populate(posy_h5[start_local:end_local], 'posy')
                    component.populate(posz_h5[start_local:end_local], 'posz')
                    component.populate(momx_h5[start_local:end_local], 'momx')
                    component.populate(momy_h5[start_local:end_local], 'momy')
                    component.populate(momz_h5[start_local:end_local], 'momz')
                    # If the snapshot and the current run uses different
                    # systems of units, mulitply the component positions
                    # and momenta by the snapshot units.
                    if snapshot_unit_length != 1:
                        for i in range(N_local):
                            component.posx[i] *= snapshot_unit_length
                            component.posy[i] *= snapshot_unit_length
                            component.posz[i] *= snapshot_unit_length
                    unit = snapshot_unit_length/snapshot_unit_time*snapshot_unit_mass
                    if unit != 1:
                        for i in range(N_local):
                            component.momx[i] *= unit
                            component.momy[i] *= unit
                            component.momz[i] *= unit
                    # Finalize progress message
                    masterprint('done')
                elif representation == 'fluid':
                    # Construct a Component instance and append it
                    # to this snapshot's list of components.
                    gridsize = component_h5.attrs['gridsize']
                    component = Component(name, species, gridsize, mass)
                    self.components.append(component)
                    # Done loading component attributes
                    if only_params:
                        continue
                    # Write out progress message
                    if not only_params:
                        masterprint('Reading in {} ({} with gridsize {}) ...'
                                    .format(name, species, gridsize), indent=4)
                    # Extract HDF5 datasets
                    δ_h5 = component_h5[unicode('δ')]
                    ux_h5 = component_h5['ux']
                    uy_h5 = component_h5['uy']
                    uz_h5 = component_h5['uz']
                    # Compute local indices of fluid grids
                    domain_size_i = gridsize//domain_subdivisions[0]
                    domain_size_j = gridsize//domain_subdivisions[1]
                    domain_size_k = gridsize//domain_subdivisions[2]
                    if master and (   gridsize != domain_subdivisions[0]*domain_size_i
                                   or gridsize != domain_subdivisions[1]*domain_size_j
                                   or gridsize != domain_subdivisions[2]*domain_size_k):
                        abort('The gridsize of the {} component is {}\n'
                              'which cannot be equally shared among {} processes'
                              .format(name, gridsize, nprocs))
                    domain_start_i = domain_layout_local_indices[0]*domain_size_i
                    domain_start_j = domain_layout_local_indices[1]*domain_size_j
                    domain_start_k = domain_layout_local_indices[2]*domain_size_k
                    domain_end_i = domain_start_i + domain_size_i
                    domain_end_j = domain_start_j + domain_size_j
                    domain_end_k = domain_start_k + domain_size_k
                    # Populate the Component instance with data from the file
                    component.populate(δ_h5[domain_start_i:domain_end_i,
                                            domain_start_j:domain_end_j,
                                            domain_start_k:domain_end_k], 'δ')
                    component.populate(ux_h5[domain_start_i:domain_end_i,
                                             domain_start_j:domain_end_j,
                                             domain_start_k:domain_end_k], 'ux')
                    component.populate(uy_h5[domain_start_i:domain_end_i,
                                             domain_start_j:domain_end_j,
                                             domain_start_k:domain_end_k], 'uy')
                    component.populate(uz_h5[domain_start_i:domain_end_i,
                                             domain_start_j:domain_end_j,
                                             domain_start_k:domain_end_k], 'uz')
                    # If the snapshot and the current run uses different
                    # systems of units, mulitply the component
                    # velocities (δ has no units) by the snapshot units.
                    unit = snapshot_unit_length/snapshot_unit_time
                    if unit != 1:
                        for i in range(gridsize**3):
                            component.ux[i] *= unit
                            component.uy[i] *= unit
                            component.uz[i] *= unit
                    # Finalize progress message
                    masterprint('done')
                elif master:
                    abort('Does not know how to load component "{}" with representation "{}"'
                          .format(name, representation))
        # Scatter particles to the correct domain-specific process.
        # Setting reset_indices_send == True ensures that buffers
        # will be reset afterwards, as this initial exchange is not
        # representable for those to come.
        if not only_params and do_exchange:
            for component in self.components[:(len(self.components) - 1)]:
                exchange(component, reset_buffers=False)
            # Reset communication buffers after last exchange
            exchange(component, reset_buffers=True)

    # This method populate the snapshot with component data
    # and additional parameters.
    @cython.pheader(# Arguments
                    components='list',
                    a='double',
                    params='dict',
                    # Locals
                    key='str',
                    )
    def populate(self, components, a, params={}):
        # Pupulate snapshot with the components
        self.components = components
        # Populate snapshot with the parsed scalefactor
        # and global parameters. If a params dict is parsed,
        # use values from this instead.
        self.params['H0']      = params.get('H0',      H0)
        self.params['a']       = params.get('a',       a)
        self.params['boxsize'] = params.get('boxsize', boxsize)
        self.params['Ωm']      = params.get('Ωm',      Ωm)
        self.params['ΩΛ']      = params.get('ΩΛ',      ΩΛ)
        # Populate the base units with the global base units
        self.units['length'] = unit_length
        self.units['time']   = unit_time
        self.units['mass']   = unit_mass

# Class storing a Gadget2 snapshot. Besides holding methods for
# saving/loading, it stores particle data (positions, momenta, mass)
# and also Gadget ID's and the Gadget header.
@cython.cclass
class Gadget2Snapshot:
    """Only Gadget2 type 1 (halo) particles,
    corresponding to dark matter particles, are supported.
    The params dict contain the following entries:
    'H0'
    'a'
    'boxsize'
    'Ωm'
    'ΩΛ'
    'header'
    """
    # The filename extension for this type of snapshot
    extension = ''

    # Initialization method.
    @cython.header
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Gadget2Snapshot type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        public dict params
        public list components
        Component component
        unsigned int[::1] ID
        """
        # Dict containing all the parameters of the snapshot
        self.params = {}
        # List of Component instances (will only ever hold
        # self.component, which can only be GADGET halos).
        self.components = []
        # The actual component data
        self.component = None
        # The ID of each particle (not used by the CO𝘕CEPT code)
        self.ID = None

    # Method for saving a GADGET2 snapshot of type 2 to disk
    @cython.pheader(# Arguments
                    filename='str',
                    # Locals
                    N='Py_ssize_t',
                    N_local='Py_ssize_t',
                    i='int',
                    component='Component',
                    unit='double',
                    )
    def save(self, filename):
        """The snapshot data (positions and velocities) are stored in
        single precision. Only GADGET2 type 1 (halo) particles,
        corresponding to dark matter particles, are supported.
        """
        component = self.component
        if master and component.species != 'dark matter particles':
            abort('The GAGDET2 snapshot type can only store dark matter particles\n'
                  '(the species of the {} component is "{}")'
                  .format(component.name, component.species))
        masterprint('Saving GADGET2 snapshot "{}":'.format(filename))
        masterprint('Writing out {} ({} {}) ...'.format(component.name,
                                                        component.N,
                                                        component.species),
                    indent=4)
        N = component.N
        N_local = component.N_local
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
                    (asarray(np.vstack((component.posx_mv[:N_local],
                                        component.posy_mv[:N_local],
                                        component.posz_mv[:N_local])
                                       ).T.flatten(),
                             dtype=C2np['float'])/unit).tofile(f)
                    # The closing int
                    if i == nprocs - 1:
                        f.write(struct.pack('i', 3*N*4))
        # Write the VEL block in serial, one process at a time
        unit = units.km/units.s*component.mass*header['Time']**1.5
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
                    (asarray(np.vstack((component.momx_mv[:N_local],
                                        component.momy_mv[:N_local],
                                        component.momz_mv[:N_local])
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
        # Finalize progress message
        masterprint('done')

    # Method for loading in a GADGET2 snapshot of type 2 from disk
    @cython.pheader(# Arguments
                    filename='str',
                    compare_params='bint',
                    only_params='bint',
                    do_exchange='bint',
                    # Locals
                    N='Py_ssize_t',
                    N_local='Py_ssize_t',
                    N_locals='tuple',
                    blockname='str',
                    file_position='Py_ssize_t',
                    header='object',  # collections.OrderedDict
                    mass='double',
                    name='str',
                    offset='Py_ssize_t',
                    size='int',
                    species='str',
                    start_local='Py_ssize_t',
                    unit='double',
                    )
    def load(self, filename, compare_params=True, only_params=False, do_exchange=True):
        """ It is assumed that the snapshot on the disk is a GADGET2
        snapshot of type 2 and that it uses single precision. The
        Gadget2Snapshot instance stores the data (positions and
        velocities) in double precision. Only GADGET type 1 (halo)
        particles, corresponding to dark matter particles,
        are supported.
        """
        if not only_params:
            masterprint('Loading GADGET2 snapshot "{}":'.format(filename))
        # Only type 1 (halo) particles are supported
        name = 'GADGET halos'
        species = 'dark matter particles'
        # Read in the snapshot
        offset = 0
        with open(filename, 'rb') as f:
            # Read the HEAD block into a params['header'] dict.
            # No unit conversion will be done.
            offset = self.new_block(f, offset)
            blockname = self.read(f, '4s').decode('utf8')  # "HEAD"
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
            # directly in the params dict. These are the same as
            # those included in the params dict of
            # standard type snapshots.
            unit = 100*units.km/(units.s*units.Mpc)
            self.params['H0']      = header['HubbleParam']*unit
            self.params['a']       = header['Time']
            unit = units.kpc/header['HubbleParam']
            self.params['boxsize'] = header['BoxSize']*unit
            self.params['Ωm']      = header['Omega0']
            self.params['ΩΛ']      = header['OmegaLambda']
            # Check if the parameters of the snapshot matches
            # those of the current simulation run. Display a warning
            # if they do not.
            if compare_params:
                compare_parameters(self.params, filename)
            # Construct a Component instance and pack it
            # into this snapshot's list of components.
            N = header['Npart'][1]
            unit = 1e+10*units.m_sun/header['HubbleParam']
            mass = header['Massarr'][1]*unit
            self.component = Component(name, species, N, mass)
            self.components = [self.component]
            # Done loading component attributes
            if only_params:
                return
            # Write out progress message
            masterprint('Reading in {} ({} {}) ...'.format(name, N, species), indent=4)
            # Compute a fair distribution
            # of component data to the processes.
            N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                        + (N//nprocs + 1, )*(N % nprocs))
            N_local = N_locals[rank]
            start_local = np.sum(N_locals[:rank], dtype=C2np['Py_ssize_t'])          
            # Read in the POS block. The positions are given in kpc/h.
            offset = self.new_block(f, offset)
            unit = units.kpc/header['HubbleParam']
            name = self.read(f, '4s').decode('utf8')  # "POS "
            size = self.read(f, 'i')
            offset = self.new_block(f, offset)
            f.seek(12*start_local, 1)  # 12 = sizeof(float)*Ndims
            file_position = f.tell()
            self.component.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [0::3], dtype=C2np['double'])*unit,
                                    'posx')
            f.seek(file_position)
            self.component.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [1::3], dtype=C2np['double'])*unit,
                                    'posy')
            f.seek(file_position)
            self.component.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [2::3], dtype=C2np['double'])*unit,
                                    'posz')
            # Read in the VEL block. The velocities are peculiar
            # velocities u=a*dx/dt divided by sqrt(a), given in km/s.
            offset = self.new_block(f, offset)
            unit = units.km/units.s*mass*header['Time']**1.5
            blockname = self.read(f, '4s').decode('utf8')  # "VEL "
            size = self.read(f, 'i')
            offset = self.new_block(f, offset)
            f.seek(12*start_local, 1)  # 12 = sizeof(float)*Ndims
            file_position = f.tell()
            self.component.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [0::3], dtype=C2np['double'])*unit,
                                    'momx')
            f.seek(file_position)
            self.component.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [1::3], dtype=C2np['double'])*unit,
                                    'momy')
            f.seek(file_position)
            self.component.populate(asarray(np.fromfile(f, dtype=C2np['float'], count=3*N_local)
                                            [2::3], dtype=C2np['double'])*unit,
                                    'momz')
            # Read in the ID block.
            # The ID's will be distributed among all processes.
            offset = self.new_block(f, offset)
            blockname = self.read(f, '4s').decode('utf8')  # "ID  "
            size = self.read(f, 'i')
            offset = self.new_block(f, offset)
            f.seek(4*start_local, 1)  # 4 = sizeof(unsigned int)
            file_position = f.tell()
            self.ID = np.fromfile(f, dtype=C2np['unsigned int'], count=N_local)
            # Finalize progress message
            masterprint('done')
            # Possible additional meta data ignored
        # Scatter particles to the correct domain-specific process.
        # Setting reset_indices_send == True ensures that buffers
        # will be reset afterwards, as this initial exchange is not
        # representable for those to come.
        if do_exchange:
            exchange(self.component, reset_buffers=True)

    # This method populate the snapshot with component data
    # as well as ID's (which are not used by this code) and
    # additional header information.
    @cython.pheader(# Arguments
                    components='list',
                    a='double',
                    params='dict',
                    # Locals
                    component='Component',
                    h='double',
                    start_local='Py_ssize_t',
                    unit='double',
                    )
    def populate(self, components, a, params={}):
        """The following header fields depend on the particles:
            Npart, Massarr, Nall.
        The following header fields depend on the current time:
            Time, Redshift.
        The following header fields correspond to the parameters
        used in the current run:
            BoxSize, Omega0, OmegaLambda, HubbleParam.
        All other fields get generic values.
        """
        # Pupulate snapshot with the GADGTE halos
        component = components[0]
        self.component = component
        self.components = [component]
        # The ID's of the local particles, generated such that
        # the process with the lowest rank has the lowest ID's.
        start_local = int(np.sum(smart_mpi(component.N_local, mpifun='allgather')[:rank]))
        self.ID = arange(start_local, start_local + component.N_local, dtype=C2np['unsigned int'])
        # Populate snapshot with the parsed scalefactor
        # and global parameters. If a params dict is parsed,
        # use values from this instead.
        self.params['H0']      = params.get('H0',      H0)
        self.params['a']       = params.get('a',       a)
        self.params['boxsize'] = params.get('boxsize', boxsize)
        self.params['Ωm']      = params.get('Ωm',      Ωm)
        self.params['ΩΛ']      = params.get('ΩΛ',      ΩΛ)
        # Build the GADGET header
        self.update_header()

    # Method for constructing the GADGET header from the other
    # parameters in the params dict.
    @cython.header(# Locals
                   component='Component',
                   h='double',
                   header='object',  # collections.OrderedDict
                   params='dict',
                   unit='double',
                   )
    def update_header(self):
        # Extract variabled
        component = self.component
        params = self.params
        # The GADGET header is constructed from scratch
        params['header'] = collections.OrderedDict()
        header = params['header']
        # Fill the header
        header['Npart'] = [0, component.N, 0, 0, 0, 0]
        unit = 100*units.km/(units.s*units.Mpc)
        h = params['H0']/unit
        unit = 1e+10*units.m_sun/h
        header['Massarr']       = [0.0, component.mass/unit, 0.0, 0.0, 0.0, 0.0]
        header['Time']          = params['a']
        header['Redshift']      = 1/params['a'] - 1
        header['FlagSfr']       = 0
        header['FlagFeedback']  = 0
        header['Nall']          = [0, component.N, 0, 0, 0, 0]
        header['FlagCooling']   = 0
        header['Numfiles']      = 1
        unit = units.kpc/h
        header['BoxSize']       = params['boxsize']/unit
        header['Omega0']        = params['Ωm']
        header['OmegaLambda']   = params['ΩΛ']
        header['HubbleParam']   = h
        header['FlagAge']       = 0
        header['FlagMetals']    = 0
        header['NallHW']        = [0, 0, 0, 0, 0, 0]
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
        # It is nicer to use mutable lists than immutable tuples
        return list(t)

    # Method that handles the file object's position in the snapshot
    # file during loading. Call it when the next block should be read.
    @cython.header(offset='Py_ssize_t',
                   f='object',
                   returns='Py_ssize_t',
                   )
    def new_block(self, f, offset):
        # Set the current position in the file
        f.seek(offset)
        # Each block is bracketed with a 4-byte int
        # containing the size of the block
        offset += 8 + self.read(f, 'i')
        return offset



# Function whick takes in a dict of parameters and compare their
# values to those of the current run. If any disagreement is found,
# write a warning message.
@cython.header(# Arguments
               params='dict',
               filename='str',
               indent='bint',
               # Locals
               indent_str='str',
               msg='str',
               reltol='double',
               vs='str',
               )
def compare_parameters(params, filename, indent=4):
    """Specifically, the following parameters are compared:
    a (compared against a_begin)
    boxsize
    H0
    Ωm
    ΩΛ
    """
    # The relative tolerence by which the parameters are compared
    reltol = 1e-6
    # Format strings
    vs = '{:.' + str(int(1 - log10(reltol))) + 'g}'
    vs += ' vs ' + vs
    indent_str = '\n' + ' '*2*indent
    msg = ''
    # Do the comparisons one by one
    if not isclose(a_begin, float(params['a']), reltol):
        msg += (indent_str + 'a_begin: ' + vs).format(a_begin, params['a'])
    if not isclose(boxsize, float(params['boxsize']), reltol):
        msg += (indent_str + 'boxsize: ' + vs + ' [{}]').format(boxsize,
                                                                params['boxsize'],
                                                                unit_length)
    if not isclose(H0, float(params['H0']), reltol):
        unit = units.km/(units.s*units.Mpc)
        msg += (indent_str + 'H0: ' + vs + ' [{}]'
                ).format(H0/unit,
                         params['H0']/unit,
                         'km s{inv} Mpc{inv}'.format(inv=(unicode('⁻¹')))
                         )
    if not isclose(Ωm, float(params['Ωm']), reltol):
        msg += (indent_str + unicode('Ωm: ') + vs).format(Ωm, params['Ωm'])
    if not isclose(ΩΛ, float(params['ΩΛ']), reltol):
        msg += (indent_str + unicode('ΩΛ: ') + vs).format(ΩΛ, params['ΩΛ'])
    if msg:
        msg = ('Mismatch between current parameters and those in the snapshot "{}":{}'
               ).format(filename, msg)
        masterwarn(msg, indent=indent)

# Function for determining the snapshot type of a file
@cython.header(# Arguments
               filename='str',
               # Locals
               head='tuple',
               returns='str',
               )
def get_snapshot_type(filename):
    # Abort if the file does not exist
    if master and not os.path.exists(filename):
        abort('The snapshot file "{}" does not exist'.format(filename))
    # Test for standard HDF5 format by looking up ΩΛ
    try:
        with h5py.File(filename, mode='r') as f:
            f.attrs[unicode('ΩΛ')]
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

# Function that saves the current state of the simulation
# - consisting of global parameters as well as the list of components -
# to a snapshot file. The type of snapshot to be saved is determined by
# the snapshot_type parameter.
@cython.header(# Argument
               components='list',
               a='double',
               filename='str',
               # Locals
               snapshot='object'  # Any implemented snapshot type
               )
def save(components, a, filename):
    # Instantiate snapshot of the appropriate type
    snapshot = eval(snapshot_type.capitalize() + 'Snapshot()')
    # Populate the snapshot with data and save it to disk
    snapshot.populate(components, a)
    snapshot.save(filename)

# Function that loads a snapshot file.
# The type of snapshot can be any of the implemented.
@cython.pheader(# Argument
                filename='str',
                compare_params='bint',
                only_params='bint',
                only_components='bint',
                do_exchange='bint',
                as_if='str',
                # Locals
                input_type='str',
                snapshot='object',          # Some snapshot type
                snapshot_newtype='object',  # Some snapshot type
                returns='object',           # Snapshot or list
                )
def load(filename, compare_params=True,
                   only_params=False,
                   only_components=False,
                   do_exchange=True,
                   as_if=''):
    """When only_params == False and only_components == False,
    the return type is simply a snapshot object containing all the
    data in the snapshot on disk.
    When only_components == True, a list of components within
    the snapshot will be returned.
    When only_params == True, a snapshot object will be returned,
    containing both parameters (.params) and components (.components),
    just as when only_params == False. These components will have
    correctly specified attributes, but no no actual component data.
    """
    # If no snapshot should be loaded, return immediately
    if not filename:
        return
    # Determine snapshot type
    input_type = get_snapshot_type(filename)
    if master and input_type is None:
        abort('Cannot recognize "{}" as neither a standard nor a GADGET2 snapshot'
              .format(filename))
    # Instantiate snapshot of the appropriate type
    snapshot = eval(input_type.capitalize() + 'Snapshot()')
    # Load the snapshot from disk
    snapshot.load(filename, compare_params=compare_params,
                            only_params=only_params,
                            do_exchange=do_exchange)
    # If the caller is interested in the components only,
    # return the list of components.
    if only_components:
        return snapshot.components
    # If a specific snapshot type is required, build this snapshot
    # and populate it with the loaded data.
    if as_if and as_if != input_type:
        snapshot_newtype = eval(as_if.capitalize() + 'Snapshot()')
        snapshot_newtype.populate(snapshot.components,
                                  snapshot.params['a'],
                                  snapshot.params)
        return snapshot_newtype
    # Return the loaded snapshot
    return snapshot



# Possible filename extensions for snapshots
cython.declare(snapshot_extensions='tuple')
snapshot_extensions = tuple([snapshot_class.extension for snapshot_class in (StandardSnapshot,
                                                                             Gadget2Snapshot,
                                                                             )
                             if snapshot_class.extension])

