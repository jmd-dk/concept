# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2017 Jeppe Mosgaard Dakin.
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
cimport('from communication import communicate_domain, exchange')
cimport('from fluid import maccormack')



# Class which serves as the data structure for a scalar fluid grid
# (each component of a fluid variable is stored as a collection of
# scalar grids). Each scalar fluid has its own name, e.g.
# ρ     (varnum == 0, multi_index == 0),
# ρu[0] (varnum == 1, multi_index == 0),
# ρu[1] (varnum == 1, multi_index == 1),
# ρu[2] (varnum == 1, multi_index == 2),
@cython.cclass
class FluidScalar:
    # Initialization method
    @cython.header(# Arguments
                   varnum='int',
                   multi_index='object',  # tuple or int-like
                   # Locals
                   shape='tuple',
                   size='Py_ssize_t',
                   )
    def __init__(self, varnum, multi_index=()):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the FluidScalar type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        # Fluid variable number and index of fluid scalar
        public int varnum
        public tuple multi_index
        # The data itself
        double* grid
        public double[:, :, ::1] grid_mv
        public double[:, :, :] grid_noghosts
        # The starred buffer
        double* gridˣ
        public double[:, :, ::1] gridˣ_mv
        public double[:, :, :] gridˣ_noghosts
        # The Δ buffer
        double* Δ
        public double[:, :, ::1] Δ_mv
        public double[:, :, :] Δ_noghosts
        """
        # Number and index of fluid variable
        self.varnum = varnum
        self.multi_index = tuple(any2iter(multi_index))
        # Minimal starting layout
        shape = (1, 1, 1)
        size = np.prod(shape)
        # The data itself
        self.grid = malloc(size*sizeof('double'))
        self.grid_mv = cast(self.grid, 'double[:shape[0], :shape[1], :shape[2]]')
        self.grid_noghosts = self.grid_mv[:, :, :]
        # The starred buffer
        self.gridˣ = malloc(size*sizeof('double'))
        self.gridˣ_mv = cast(self.gridˣ, 'double[:shape[0], :shape[1], :shape[2]]')
        self.gridˣ_noghosts = self.gridˣ_mv[:, :, :]
        # The Δ buffer
        self.Δ = malloc(size*sizeof('double'))
        self.Δ_mv = cast(self.Δ, 'double[:shape[0], :shape[1], :shape[2]]')
        self.Δ_noghosts = self.Δ_mv[:, :, :]

    # Method for resizing all grids of this scalar fluid
    @cython.header(# Arguments
                   shape_nopseudo_noghost='tuple',
                   # Locals
                   s='Py_ssize_t',
                   shape='tuple',
                   shape_noghosts='tuple',
                   size='Py_ssize_t',
                   size_noghosts='Py_ssize_t',
                   )
    def resize(self, shape_nopseudo_noghost):
        """After resizing the fluid scalar,
        all fluid elements will be nullified.
        """
        # The full shape and size of the grid,
        # with pseudo and ghost points.
        shape = tuple([2 + s + 1 + 2 for s in shape_nopseudo_noghost])
        size = np.prod(shape)
        # The shape and size of the grid
        # with no ghost points but with pseudo points.
        shape_noghosts = tuple([s + 1 for s in shape_nopseudo_noghost])
        size_noghosts = np.prod(shape_noghosts)
        # The data itself
        self.grid = realloc(self.grid, size*sizeof('double'))
        self.grid_mv = cast(self.grid, 'double[:shape[0], :shape[1], :shape[2]]')
        self.grid_noghosts = self.grid_mv[2:(self.grid_mv.shape[0] - 2),
                                          2:(self.grid_mv.shape[1] - 2),
                                          2:(self.grid_mv.shape[2] - 2)]
        # Nullify the newly allocated data grid
        self.nullify_grid()
        # The starred buffer
        self.gridˣ = realloc(self.gridˣ, size*sizeof('double'))
        self.gridˣ_mv = cast(self.gridˣ, 'double[:shape[0], :shape[1], :shape[2]]')
        self.gridˣ_noghosts = self.gridˣ_mv[2:(self.gridˣ_mv.shape[0] - 2),
                                            2:(self.gridˣ_mv.shape[1] - 2),
                                            2:(self.gridˣ_mv.shape[2] - 2)]
        # Nullify the newly allocated starred buffer
        self.nullify_gridˣ()
        # The starred buffer
        self.Δ = realloc(self.Δ, size*sizeof('double'))
        self.Δ_mv = cast(self.Δ, 'double[:shape[0], :shape[1], :shape[2]]')
        self.Δ_noghosts = self.Δ_mv[2:(self.Δ_mv.shape[0] - 2),
                                    2:(self.Δ_mv.shape[1] - 2),
                                    2:(self.Δ_mv.shape[2] - 2)]
        # Nullify the newly allocated Δ buffer
        self.nullify_Δ()

    # Method for nullifying the data grid
    @cython.pheader(# Locals
                    i='Py_ssize_t',
                    grid='double*',
                    shape='Py_ssize_t*',
                    )
    def nullify_grid(self):
        # Extract data pointer
        grid = self.grid
        # Nullify starred buffer
        shape = self.grid_mv.shape
        for i in range(shape[0]*shape[1]*shape[2]):
            grid[i] = 0

    # Method for nullifying the starred grid
    @cython.pheader(# Locals
                    i='Py_ssize_t',
                    gridˣ='double*',
                    shape='Py_ssize_t*',
                    )
    def nullify_gridˣ(self):
        # Extract starred buffer pointer
        gridˣ = self.gridˣ
        # Nullify starred buffer
        shape = self.gridˣ_mv.shape
        for i in range(shape[0]*shape[1]*shape[2]):
            gridˣ[i] = 0

    # Method for nullifying the Δ buffer
    @cython.pheader(# Locals
                    i='Py_ssize_t',
                    Δ='double*',
                    shape='Py_ssize_t*',
                    )
    def nullify_Δ(self):
        # Extract Δ buffer pointer
        Δ = self.Δ
        # Nullify Δ buffer
        shape = self.Δ_mv.shape
        for i in range(shape[0]*shape[1]*shape[2]):
            Δ[i] = 0

    # This method is automaticlly called when a FluidScalar instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        free(self.grid)
        free(self.gridˣ)

    # String representation
    def __repr__(self):
        if self.varnum < len(fluidvarnames):
            name = fluidvarnames[self.varnum]
        else:
            name = str(self.varnum)
        if self.varnum > 0 and self.multi_index:
            return '<fluidscalar {}[{}]>'.format(name,
                                                 ', '.join([str(i) for i in self.multi_index]))
        else:
            return '<fluidscalar {}>'.format(name)
    def __str__(self):
        return self.__repr__()


# The class governing any component of the universe
@cython.cclass
class Component:
    """An instance of this class represents either a collection of
    particles or a grid of fluid values. A Component instance should be
    present on all processes.
    """

    # Initialization method
    @cython.header(# Arguments
                   name='str',
                   species='str',
                   N_or_gridsize='Py_ssize_t',
                   mass='double',
                   fluid_properties='dict',
                   )
    def __init__(self, name, species, N_or_gridsize, mass, fluid_properties={}):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Component type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        # General component attributes
        public dict forces
        public double mass
        public str name
        public str representation
        public str species
        # Particle component attributes
        public Py_ssize_t N
        Py_ssize_t N_allocated
        Py_ssize_t N_local
        double softening
        # Particle data
        double* posx
        double* posy
        double* posz
        double* momx
        double* momy
        double* momz
        double** pos
        double** mom
        double[::1] posx_mv
        double[::1] posy_mv
        double[::1] posz_mv
        double[::1] momx_mv
        double[::1] momy_mv
        double[::1] momz_mv
        list pos_mv
        list mom_mv
        # Fluid component attributes
        public Py_ssize_t gridsize
        # Dict storing the fluid grids
        public dict fluidvars
        """
        # General component attributes
        self.name    = name
        self.species = species
        self.mass    = mass
        # Attatch information about what forces (including the method
        # used to compute these forces) act on this species in this
        # particular simulation to the component instance.
        self.forces = {}
        for species in (self.species, 'all'):
            if species in forces:
                for species_force in forces[species]:
                    if not species_force[0] in self.forces:
                        self.forces[species_force[0]] = species_force[1]
        # Determine the representation based on the species
        self.representation = get_representation(self.species)
        # Particle component attributes
        self.N_allocated = 1
        self.N_local     = 1
        if self.representation == 'particles':
            self.N = N_or_gridsize
            if self.species in softeningfactors:
                self.softening = softeningfactors[self.species]*boxsize/cbrt(self.N)
            else:
                abort('Species "{}" do not have an assigned softening factor!'
                      .format(self.species))
        else:
            self.N         = 1
            self.softening = 1
        # Populate the fluid arguments dict
        fluid_properties['N_fluidvars'] = fluid_properties.get('N_fluidvars', 2)
        fluid_properties['cs']          = fluid_properties.get('cs'         , 0)
        # Particle data
        self.posx = malloc(self.N_allocated*sizeof('double'))
        self.posy = malloc(self.N_allocated*sizeof('double'))
        self.posz = malloc(self.N_allocated*sizeof('double'))
        self.momx = malloc(self.N_allocated*sizeof('double'))
        self.momy = malloc(self.N_allocated*sizeof('double'))
        self.momz = malloc(self.N_allocated*sizeof('double'))
        self.posx_mv = cast(self.posx, 'double[:self.N_allocated]')
        self.posy_mv = cast(self.posy, 'double[:self.N_allocated]')
        self.posz_mv = cast(self.posz, 'double[:self.N_allocated]')
        self.momx_mv = cast(self.momx, 'double[:self.N_allocated]')
        self.momy_mv = cast(self.momy, 'double[:self.N_allocated]')
        self.momz_mv = cast(self.momz, 'double[:self.N_allocated]')
        # Pack particle data into a pointer array of pointers
        # and a list of memoryviews.
        self.pos = malloc(3*sizeof('double*'))
        self.pos[0] = self.posx
        self.pos[1] = self.posy
        self.pos[2] = self.posz
        self.mom = malloc(3*sizeof('double*'))
        self.mom[0] = self.momx
        self.mom[1] = self.momy
        self.mom[2] = self.momz
        self.pos_mv = [self.posx_mv, self.posy_mv, self.posz_mv]
        self.mom_mv = [self.momx_mv, self.momy_mv, self.momz_mv]
        # Fluid component attributes
        if self.representation == 'fluid':
            self.gridsize = N_or_gridsize
        else:
            self.gridsize = 1
        # Construct the fluidvars dict, storing all fluid variables
        # and meta data.
        self.fluidvars = {# Grid shape
                          'shape'         : (1, 1, 1),
                          'size'          : 1,
                          'shape_noghosts': (1, 1, 1),
                          'size_noghosts' : 1,
                          # Fluid properties
                          'N_fluidvars': fluid_properties['N_fluidvars'],
                          'cs'         : fluid_properties['cs'],
                          }
        # Create the two lowest order fluid variables (ρ and ρu)
        # SHOULD BE DEPENDENT ON N_fluidvars !!!
        self.fluidvars[0] = asarray([FluidScalar(0)], dtype='object')
        self.fluidvars[1] = asarray([FluidScalar(1, dim) for dim in range(3)], dtype='object')
        # Also assign some convenient names for the fluid grids
        self.assign_fluidnames()

    # This method populate the Component pos/mom arrays (for a
    # particles representation) or the fluid variable arrays (for a
    # fluid representation) with data. It is deliberately designed so
    # that you have to make a call for each attribute. You should
    # construct the data array within the call itself, as this will
    # minimize memory usage. This data array is 1D for particle data and
    # 3D for fluid data.
    @cython.header(# Arguments
                   data='object',  # 1D/3D (particles/fluid) memoryview
                   var='object',   # str or int-like
                   indices='object',  # Int-like or tuple
                   # Locals
                   fluidscalar='FluidScalar',
                   mv1D='double[::1]',
                   mv3D='double[:, :, ::1]',
                   tmp='object',  # FluidScalar or np.ndarray
                   )
    def populate(self, data, var, indices=0):
        if self.representation == 'particles':
            mv1D = data
            self.N_local = mv1D.shape[0]
            # Enlarge data attributes if necessary
            if self.N_allocated < self.N_local:
                self.resize(self.N_local)
            # Update the data corresponding to the parsed string
            if var == 'posx':
                self.posx_mv[...] = mv1D[...]
            elif var == 'posy':
                self.posy_mv[...] = mv1D[...]
            elif var == 'posz':
                self.posz_mv[...] = mv1D[...]
            elif var == 'momx':
                self.momx_mv[...] = mv1D[...]
            elif var == 'momy':
                self.momy_mv[...] = mv1D[...]
            elif var == 'momz':
                self.momz_mv[...] = mv1D[...]
            elif master:
                abort('Wrong component attribute name "{}"!'.format(var))
        elif self.representation == 'fluid':
            mv3D = data
            if master and var not in self.fluidvars:
                if isinstance(var, str):
                    abort('The "{}" component does not contain a fluid variable with the name "{}"'
                          .format(self.name, var))
                else:
                    abort('The "{}" component does not have fluid variable with number {}'
                          .format(self.name, var))
            # Reallocate fluid grids if necessary           
            self.resize(asarray(mv3D).shape)
            # Populate the scalar grid given by 'indices' of the fluid
            # given by 'var' with the parsed data. This data should not
            # include pseudo or ghost points.
            tmp = self.fluidvars[var]
            if isinstance(tmp, FluidScalar):
                fluidscalar = tmp
            else:
                fluidscalar = tmp[indices]
            fluidscalar.grid_noghosts[:mv3D.shape[0], :mv3D.shape[1], :mv3D.shape[2]] = mv3D[...]
            # Populate pseudo and ghost points
            communicate_domain(fluidscalar.grid_mv, mode='populate')

    # This method will grow/shrink the data attributes.
    # Note that it will update N_allocated but not N_local.
    @cython.header(# Arguments
                   size_or_shape_nopseudo_noghosts='object',  # Py_ssize_t or tuple
                   # Locals
                   N_allocated='Py_ssize_t',
                   fluidscalar='FluidScalar',
                   s='Py_ssize_t',
                   shape_nopseudo_noghosts='tuple',
                   size='Py_ssize_t',
                   s_old='Py_ssize_t',
                   )
    def resize(self, size_or_shape_nopseudo_noghosts):
        if self.representation == 'particles':
            size = size_or_shape_nopseudo_noghosts
            if size != self.N_allocated:
                self.N_allocated = size
                # Reallocate particle data
                self.posx = realloc(self.posx, self.N_allocated*sizeof('double'))
                self.posy = realloc(self.posy, self.N_allocated*sizeof('double'))
                self.posz = realloc(self.posz, self.N_allocated*sizeof('double'))
                self.momx = realloc(self.momx, self.N_allocated*sizeof('double'))
                self.momy = realloc(self.momy, self.N_allocated*sizeof('double'))
                self.momz = realloc(self.momz, self.N_allocated*sizeof('double'))
                # Reassign particle data memory views
                self.posx_mv = cast(self.posx, 'double[:self.N_allocated]')
                self.posy_mv = cast(self.posy, 'double[:self.N_allocated]')
                self.posz_mv = cast(self.posz, 'double[:self.N_allocated]')
                self.momx_mv = cast(self.momx, 'double[:self.N_allocated]')
                self.momy_mv = cast(self.momy, 'double[:self.N_allocated]')
                self.momz_mv = cast(self.momz, 'double[:self.N_allocated]')
                # Repack particle data into a pointer array of pointers
                # and a list of memoryviews.
                self.pos[0], self.pos[1], self.pos[2] = self.posx, self.posy, self.posz
                self.mom[0], self.mom[1], self.mom[2] = self.momx, self.momy, self.momz
                self.pos_mv = [self.posx_mv, self.posy_mv, self.posz_mv]
                self.mom_mv = [self.momx_mv, self.momy_mv, self.momz_mv]
        elif self.representation == 'fluid':
            shape_nopseudo_noghosts = size_or_shape_nopseudo_noghosts
            # The allocated shape of the fluid grids are 5 points
            # (one layer of pseudo points and two layers of ghost points
            # before and after the local grid) longer than the local
            # shape, in each direction.
            if not any([2 + s + 1 + 2 != s_old for s, s_old in zip(shape_nopseudo_noghosts,
                                                                   self.fluidvars['shape'])]):
                return
            if any([s < 1 for s in shape_nopseudo_noghosts]):
                abort('Attempted to resize fluid grids of the {} component'
                      'to a shape of {}'.format(self.name, shape_nopseudo_noghosts))
            # Recalculate and -assign meta data
            self.fluidvars['shape']          = tuple([2 + s + 1 + 2 for s in shape_nopseudo_noghosts])
            self.fluidvars['shape_noghosts'] = tuple([    s + 1     for s in shape_nopseudo_noghosts])
            self.fluidvars['size']           = np.prod(self.fluidvars['shape'])
            self.fluidvars['size_noghosts']  = np.prod(self.fluidvars['shape_noghosts'])
            # Reallocate fluid data
            for fluidscalar in self.iterate_fluidscalars():
                fluidscalar.resize(shape_nopseudo_noghosts)

    # Method for integrating particle positions/fluid values
    # forward in time.
    @cython.header(# Arguments
                   ᔑdt='dict',
                   # Locals
                   fac='double',
                   fluidscalar='FluidScalar',
                   i='Py_ssize_t',
                   momx='double*',
                   momy='double*',
                   momz='double*',
                   posx='double*',
                   posy='double*',
                   posz='double*',
                   )
    def drift(self, ᔑdt):
        if self.representation == 'particles':
            masterprint('Drifting {} ...'.format(self.name))
            posx = self.posx
            posy = self.posy
            posz = self.posz
            momx = self.momx
            momy = self.momy
            momz = self.momz
            # The factor 1/mass*∫_{t}^{t + Δt}dta⁻²
            fac = ᔑdt['a⁻²']/self.mass
            # Update positions
            for i in range(self.N_local):
                posx[i] += momx[i]*fac
                posy[i] += momy[i]*fac
                posz[i] += momz[i]*fac
                # Toroidal boundaries
                posx[i] = mod(posx[i], boxsize)
                posy[i] = mod(posy[i], boxsize)
                posz[i] = mod(posz[i], boxsize)
            masterprint('done')
            # Some partiles may have drifted out of the local domain.
            # Exchange particles to the correct processes.
            exchange(self)
        elif self.representation == 'fluid':
            # Evolve the fluid using the MacCormack method
            masterprint('Evolving fluid variables of {} ...'.format(self.name))
            maccormack(self, ᔑdt)
            masterprint('done')

    # Method which assigns convenient names to some
    # fluid variables and fluid scalars.
    @cython.header(# Locals
                   dim='int',
                   l='Py_ssize_t',
                   )
    def assign_fluidnames(self):
        """After running this method,
        fluid scalars can be accessed as e.g.
        self.fluidvars['ρ']   == self.fluidvars[0][0]
        self.fluidvars['ρux'] == self.fluidvars['ρu'][0] == self.fluidvars[1][0]
        self.fluidvars['ρuy'] == self.fluidvars['ρu'][1] == self.fluidvars[1][1]
        self.fluidvars['ρuz'] == self.fluidvars['ρu'][2] == self.fluidvars[1][2]
        """
        # As higher fluid variables may not have names in general,
        # enclose the assignments in a try block. The assignments
        # should be ordered accoring to the fluid variable number.
        try:
            for l in range(self.fluidvars['N_fluidvars']):
                if l == 0:
                    # ρ
                    self.fluidvars[fluidvarnames[l]] = self.fluidvars[l][0]
                elif l == 1:
                    # ρu
                    self.fluidvars[fluidvarnames[l]] = self.fluidvars[l]
                    for dim in range(3):
                        self.fluidvars[fluidvarnames[l] + 'xyz'[dim]] = self.fluidvars[l][dim]
        except:
            ...

    # Generator for looping over all the scalar fluid grids within
    # the component.
    def iterate_fluidscalars(self):
        fluidvars = self.fluidvars
        for l in range(fluidvars['N_fluidvars']):
            fluidvar = fluidvars[l]
            for multi_index in self.iterate_fluidscalar_indices(fluidvar):
                yield fluidvar[multi_index]

    # Generator for looping over all multi-indices of a fluid variable
    @staticmethod
    def iterate_fluidscalar_indices(fluidvar):
        it = np.nditer(fluidvar, flags=('refs_ok', 'multi_index'))
        while not it.finished:
            yield it.multi_index
            it.iternext()

    # Method for communicating pseudo and ghost points
    # of all fluid variables.
    @cython.header(# Arguments
                   mode='str',
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def communicate_fluid_grids(self, mode=''):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            communicate_domain(fluidscalar.grid_mv, mode=mode)

    # Method for communicating pseudo and ghost points
    # of all starred fluid variables.
    @cython.header(# Arguments
                   mode='str',
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def communicate_fluid_gridsˣ(self, mode=''):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            communicate_domain(fluidscalar.gridˣ_mv, mode=mode)

    # Method for communicating pseudo and ghost points
    # of all fluid Δ buffers.
    @cython.header(# Arguments
                   mode='str',
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def communicate_fluid_Δ(self, mode=''):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            communicate_domain(fluidscalar.Δ_mv, mode=mode)

    # Method which calls the nullify_grid on all fluid scalars
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_fluid_grid(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            fluidscalar.nullify_grid()

    # Method which calls the nullify_gridˣ on all fluid scalars
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_fluid_gridˣ(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            fluidscalar.nullify_gridˣ()

    # Method which calls the nullify_Δ on all fluid scalars
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_fluid_Δ(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            fluidscalar.nullify_Δ()

    # This method is automaticlly called when a Component instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        free(self.pos)
        free(self.posx)
        free(self.posy)
        free(self.posz)
        free(self.mom)
        free(self.momx)
        free(self.momy)
        free(self.momz)

    # String representation
    def __repr__(self):
        return '<component "{}" of species "{}">'.format(self.name, self.species)
    def __str__(self):
        return self.__repr__()



# Function for getting the component representation based on the species
@cython.header(# Arguments
               species='str',
               # Locals
               key='tuple',
               representation='str',
               returns='str'
               )
def get_representation(species):
    for key, representation in representation_of_species.items():
        if species in key:
            return representation
    abort('Species "{}" not implemented'.format(species))
# Mapping from valid species to their representations
cython.declare(representation_of_species='dict')
representation_of_species = {('baryons',
                              'dark energy particles',
                              'dark matter particles',
                              'neutrinos',
                              ): 'particles',
                             ('baryon fluid',
                              'dark matter fluid',
                              'dark energy fluid',
                              'neutrino fluid',
                              ): 'fluid',
                             }
# Mapping from fluid variable number to name
cython.declare(fluidvarnames='tuple')
fluidvarnames = ('ρ', 'ρu')
