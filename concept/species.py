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
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from analysis import measure')
cimport('from communication import communicate_domain, domain_subdivisions, exchange, smart_mpi')
cimport('from fluid import maccormack, apply_internal_sources')
cimport('from integration import Spline, cosmic_time, scale_factor, ȧ')
cimport('from linear import compute_cosmo, compute_transfers, realize')



# Class which serves as the data structure for a scalar fluid grid
# (each component of a fluid variable is stored as a collection of
# scalar grids). Each scalar fluid has its own name, e.g.
# ϱ    (varnum == 0, multi_index == 0),
# J[0] (varnum == 1, multi_index == 0),
# J[1] (varnum == 1, multi_index == 1),
# J[2] (varnum == 1, multi_index == 2),
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

    # Method for scaling the data grid
    @cython.pheader(# Argument
                    a='double',
                    # Locals
                    i='Py_ssize_t',
                    grid='double*',
                    shape='Py_ssize_t*',
                    )
    def scale_grid(self, a):
        # Extract data pointer
        grid = self.grid
        # Scale data buffer
        shape = self.grid_mv.shape
        for i in range(shape[0]*shape[1]*shape[2]):
            grid[i] *= a

    # Method for nullifying the data grid
    @cython.pheader(# Locals
                    i='Py_ssize_t',
                    grid='double*',
                    shape='Py_ssize_t*',
                    )
    def nullify_grid(self):
        # Extract data pointer
        grid = self.grid
        # Nullify data buffer
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
        return '<fluidscalar {}[{}]>'.format(self.varnum,
                                             ', '.join([str(mi) for mi in self.multi_index]))
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
    @cython.pheader(# Arguments
                    name='str',
                    species='str',
                    N_or_gridsize='Py_ssize_t',
                    species_class='str',
                    mass='double',
                    N_fluidvars='Py_ssize_t',
                    w='object',  # NoneType, float, int, str or dict
                    # Locals
                    fluidvar='object',  # np.ndarray with dtype object
                    fluidvar_shape='tuple',
                    i='Py_ssize_t',
                    index='Py_ssize_t',
                    multi_index='tuple',
                    )
    def __init__(self, name, species, N_or_gridsize, species_class='',
                 # Particle-specific arguments
                 mass=-1,
                 # Fluid-specific arguments
                 N_fluidvars=2, w=None,
                 ):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Component type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        # General component attributes
        public dict forces
        public str name
        public str representation
        public str species
        public str species_class
        # Particle attributes
        public Py_ssize_t N
        Py_ssize_t N_allocated
        Py_ssize_t N_local
        public double mass
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
        public double[::1] posx_mv
        public double[::1] posy_mv
        public double[::1] posz_mv
        public double[::1] momx_mv
        public double[::1] momy_mv
        public double[::1] momz_mv
        public list pos_mv
        public list mom_mv
        # Particle Δ buffers
        double* Δposx
        double* Δposy
        double* Δposz
        double* Δmomx
        double* Δmomy
        double* Δmomz
        double** Δpos
        double** Δmom
        public double[::1] Δposx_mv
        public double[::1] Δposy_mv
        public double[::1] Δposz_mv
        public double[::1] Δmomx_mv
        public double[::1] Δmomy_mv
        public double[::1] Δmomz_mv
        public list Δpos_mv
        public list Δmom_mv
        # Fluid attributes
        public Py_ssize_t gridsize
        public tuple shape
        public tuple shape_noghosts
        public Py_ssize_t size
        public Py_ssize_t size_noghosts
        public dict fluid_names
        str w_type
        double w_constant
        double[:, ::1] w_tabulated
        str w_expression
        Spline w_spline
        double Σmass_present
        # Fluid data
        public list fluidvars
        FluidScalar ϱ
        object J
        FluidScalar Jx
        FluidScalar Jy
        FluidScalar Jz
        object σ
        FluidScalar σxx
        FluidScalar σxy
        FluidScalar σxz
        FluidScalar σyx
        FluidScalar σyy
        FluidScalar σyz
        FluidScalar σzx
        FluidScalar σzy
        FluidScalar σzz
        """
        # Check that the name does not conflict with
        # one of the special names used internally.
        if name in internally_defined_names:
            masterwarn('A species by the name of "{}" is to be created. '
                       'As this name is used internally by the code, '
                       'this may lead to erroneous behaviour.'
                       .format(name)
                       )
        # General attributes
        self.name    = name
        self.species = species
        if not species_class:
            # If the CLASS species is not given, use the default name
            # corresponding to the (CO𝘕CEPT) species. If no such default
            # name is defined, continue on without an error and assume
            # that this name will not be used.
            if species in default_species_class:
                species_class = default_species_class[self.species]
        self.species_class = species_class
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
        # Particle attributes
        self.mass        = mass
        self.N_allocated = 1
        self.N_local     = 1
        if self.representation == 'particles':
            self.N = N_or_gridsize
            if self.species in softeningfactors:
                self.softening = softeningfactors[self.species]*boxsize/cbrt(self.N)
            elif self.species in default_softeningfactors:
                self.softening = default_softeningfactors[self.species]*boxsize/cbrt(self.N)
            else:
                abort('Species "{}" has neither an assigned nor a default softening factor!'
                      .format(self.species))
        else:
            self.N         = 1
            self.softening = 1
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
        # Particle data buffers
        self.Δposx = malloc(self.N_allocated*sizeof('double'))
        self.Δposy = malloc(self.N_allocated*sizeof('double'))
        self.Δposz = malloc(self.N_allocated*sizeof('double'))
        self.Δmomx = malloc(self.N_allocated*sizeof('double'))
        self.Δmomy = malloc(self.N_allocated*sizeof('double'))
        self.Δmomz = malloc(self.N_allocated*sizeof('double'))
        self.Δposx_mv = cast(self.Δposx, 'double[:self.N_allocated]')
        self.Δposy_mv = cast(self.Δposy, 'double[:self.N_allocated]')
        self.Δposz_mv = cast(self.Δposz, 'double[:self.N_allocated]')
        self.Δmomx_mv = cast(self.Δmomx, 'double[:self.N_allocated]')
        self.Δmomy_mv = cast(self.Δmomy, 'double[:self.N_allocated]')
        self.Δmomz_mv = cast(self.Δmomz, 'double[:self.N_allocated]')
        # Pack particle data buffers into a pointer array of pointers
        # and a list of memoryviews.
        self.Δpos = malloc(3*sizeof('double*'))
        self.Δpos[0] = self.Δposx
        self.Δpos[1] = self.Δposy
        self.Δpos[2] = self.Δposz
        self.Δmom = malloc(3*sizeof('double*'))
        self.Δmom[0] = self.Δmomx
        self.Δmom[1] = self.Δmomy
        self.Δmom[2] = self.Δmomz
        self.Δpos_mv = [self.Δposx_mv, self.Δposy_mv, self.Δposz_mv]
        self.Δmom_mv = [self.Δmomx_mv, self.Δmomy_mv, self.Δmomz_mv]
        # Fluid attributes
        self.shape = (1, 1, 1)
        self.shape_noghosts = (1, 1, 1)
        self.size = 1
        self.size_noghosts = 1
        if self.representation == 'fluid':
            self.gridsize = N_or_gridsize
            # All gridsizes should be even
            if self.gridsize%2 != 0:
                masterwarn('The fluid component "{}" have an odd gridsize ({}). '
                           'Some operations may not function correctly.'
                           .format(self.name, self.gridsize))
        else:
            self.gridsize = 1
        # Initialize the equation of state parameter w
        self.initialize_w(w)
        # Fluid data.
        # Create the N_fluidvars fluid variables
        # and store them in the list fluidvars.
        if N_fluidvars != 2 and self.representation == 'particles':
            abort('Particle components must have N_fluidvars = 2, '
                  'but N_fluidvars = {} was specified for "{}"'
                  .format(N_fluidvars, self.name))
        self.fluidvars = []
        for i in range(N_fluidvars):
            # The shape of the i'th fluid variable,
            # when thought of as a tensor.
            if i == 0:
                # Special case: The density ϱ is a scalar
                # (rank 0 tensor), not an empty tensor.
                fluidvar_shape = (1, )
            else:
                # The general shape is 3×3×3×...×3 (i times)
                fluidvar_shape = (3, )*i
            # Instantiate the tensor
            fluidvar = empty(fluidvar_shape, dtype='object')
            # Populate the tensor with fluid scalar fields
            for multi_index in self.iterate_fluidscalar_indices(fluidvar):
                fluidvar[multi_index] = FluidScalar(i, multi_index)
            # Add the fluid variable to the list
            self.fluidvars.append(fluidvar)
        # Construct mapping from names of fluid variables (e.g. J)
        # to their indices in self.fluidvars, and also from names of
        # fluid scalars (e.g. ϱ, Jx) to tuple of the form
        # (index, multi_index). The fluid scalar is then given
        # by self.fluidvars[index][multi_index].
        # Also include trivial mappings from indices to themselves,
        # and the special "reverse" mapping from indices to names
        # given by the 'ordered' key.
        fluidvar_names = ('ϱ', 'J', 'σ')
        self.fluid_names = {'ordered': fluidvar_names}
        for index, (fluidvar, fluidvar_name) in enumerate(zip(self.fluidvars, fluidvar_names)):
            # The fluid variable
            self.fluid_names[        fluidvar_name ] = index
            self.fluid_names[unicode(fluidvar_name)] = index
            self.fluid_names[index                 ] = index  # Trivial mapping
            # The fluid scalar
            for multi_index in self.iterate_fluidscalar_indices(fluidvar):
                fluidscalar_name = fluidvar_name
                if index > 0:
                    fluidscalar_name += ''.join(['xyz'[mi] for mi in multi_index])
                self.fluid_names[        fluidscalar_name ] = (index, multi_index)
                self.fluid_names[unicode(fluidscalar_name)] = (index, multi_index)
                self.fluid_names[index, multi_index       ] = (index, multi_index)  # Trivial mapping
        # Also include particle variable names in the fluid_names dict
        self.fluid_names['pos'] = 0
        self.fluid_names['mom'] = 1
        # Assign the fluid variables and scalars as convenient named
        # attributes on the Component instance.
        # Use the same naming scheme as above.
        self.ϱ  = self.fluidvars[0][0]
        self.J  = self.fluidvars[1]
        self.Jx = self.fluidvars[1][0]
        self.Jy = self.fluidvars[1][1]
        self.Jz = self.fluidvars[1][2]
        if N_fluidvars > 2:
            self.σ   = self.fluidvars[2]            
            self.σxx = self.fluidvars[2][0, 0]
            self.σxy = self.fluidvars[2][0, 1]
            self.σxz = self.fluidvars[2][0, 2]
            self.σyx = self.fluidvars[2][1, 0]
            self.σyy = self.fluidvars[2][1, 1]
            self.σyz = self.fluidvars[2][1, 2]
            self.σzx = self.fluidvars[2][2, 0]
            self.σzy = self.fluidvars[2][2, 1]
            self.σzz = self.fluidvars[2][2, 2]
        if N_fluidvars > 3:
            ...

    # This method populate the Component pos/mom arrays (for a
    # particles representation) or the fluid scalar grids (for a
    # fluid representation) with data. It is deliberately designed so
    # that you have to make a call for each attribute (posx, posy, ...
    # for particle components, ϱ, Jx, Jy, ... for fluid components).
    # You should construct the data array within the call itself,
    # as this will minimize memory usage. This data array is 1D for
    # particle data and 3D for fluid data.
    @cython.pheader(# Arguments
                    data='object',  # 1D/3D (particles/fluid) memoryview
                    var='object',   # int-like or str
                    multi_index='object',  # int-like or tuple
                    buffer='bint',
                    # Locals
                    a='double',
                    fluid_indices='object',  # tuple or int-like
                    fluidscalar='FluidScalar',
                    index='Py_ssize_t',
                    mv1D='double[::1]',
                    mv3D='double[:, :, ::1]',
                    )
    def populate(self, data, var, multi_index=0, buffer=False):
        """If buffer is True, the Δ buffers will be populated
        instead of the data arrays.
        """
        if self.representation == 'particles':
            mv1D = data
            self.N_local = mv1D.shape[0]
            # Enlarge data attributes if necessary
            if self.N_allocated < self.N_local:
                self.resize(self.N_local)
            # Update the data corresponding to the passed string
            if var == 'posx':
                if buffer:
                    self.Δposx_mv[:self.N_local] = mv1D[:]
                else:
                    self.posx_mv [:self.N_local] = mv1D[:]
            elif var == 'posy':
                if buffer:
                    self.Δposy_mv[:self.N_local] = mv1D[:]
                else:
                    self.posy_mv [:self.N_local] = mv1D[:]
            elif var == 'posz':
                if buffer:
                    self.Δposz_mv[:self.N_local] = mv1D[:]
                else:
                    self.posz_mv [:self.N_local] = mv1D[:]
            elif var == 'momx':
                if buffer:
                    self.Δmomx_mv[:self.N_local] = mv1D[:]
                else:
                    self.momx_mv [:self.N_local] = mv1D[:]
            elif var == 'momy':
                if buffer:
                    self.Δmomy_mv[:self.N_local] = mv1D[:]
                else:
                    self.momy_mv [:self.N_local] = mv1D[:]
            elif var == 'momz':
                if buffer:
                    self.Δmomz_mv[:self.N_local] = mv1D[:]
                else:
                    self.momz_mv [:self.N_local] = mv1D[:]
            elif master:
                abort('Wrong component attribute name "{}"!'.format(var))
        elif self.representation == 'fluid':
            mv3D = data
            # The fluid scalar will be given as
            # self.fluidvars[index][multi_index],
            # where index is an int and multi_index is a tuple of ints.
            # These may be given directly as var and multi_index.
            # Alternatively, var may be a str, in which case it can be
            # the name of a fluid variable or a fluid scalar.
            # For each possibility, find index and multi_index.
            if isinstance(var, int):
                if not (0 <= var < len(self.fluidvars)):
                    abort('The "{}" component does not have fluid variable with number {}'
                          .format(self.name, var))
                # The fluid scalar is given as
                # self.fluidvars[index][multi_index].
                index = var
            if isinstance(var, str):
                if var not in self.fluid_names:
                    abort('The "{}" component does not contain a fluid variable with the name "{}"'
                          .format(self.name, var))
                # Lookup the fluid indices corresponding to var.
                # This can either be a tuple of the form
                # (index, multi_index) (for a passed fluid scalar name)
                # or just an index (for a passed fluid name).
                fluid_indices = self.fluid_names[var]
                if isinstance(fluid_indices, int):
                    index = fluid_indices
                else:  # fluid_indices is a tuple
                    if multi_index != 0:
                        masterwarn('Overwriting passed multi_index ({}) with {}, '
                                   'deduced from the passed var = {}.'
                                   .format(multi_index, fluid_indices[1], var)
                                   )
                    index, multi_index = fluid_indices
            # Type check on the multi_index
            if not isinstance(multi_index, (int, tuple)):
                abort('A multi_index of type "{}" was supplied. '
                      'This should be either an int or a tuple.'
                      .format(type(multi_index))
                      )
            # Reallocate fluid grids if necessary           
            self.resize(asarray(mv3D).shape)
            # Populate the scalar grid given by index and multi_index
            # with the passed data. This data should not
            # include pseudo or ghost points.
            fluidscalar = self.fluidvars[index][multi_index]
            if buffer:
                fluidscalar.Δ_noghosts   [:mv3D.shape[0], :mv3D.shape[1], :mv3D.shape[2]] = mv3D[...]
            else:
                fluidscalar.grid_noghosts[:mv3D.shape[0], :mv3D.shape[1], :mv3D.shape[2]] = mv3D[...]       
            # Populate pseudo and ghost points
            if buffer:
                communicate_domain(fluidscalar.Δ_mv   , mode='populate')
            else:
                communicate_domain(fluidscalar.grid_mv, mode='populate')
            # If the ϱ grid has been populated, measure and store the
            # total mass. For w ≠ 0, this mass is not constant in an
            # expanding universe. The mass at the present time (a = 1)
            # is the one which is measured and stored.
            if index == 0:
                a = universals.a
                universals.a = 1
                self.Σmass_present = measure(self, 'mass')
                universals.a = a

    # This method will grow/shrink the data attributes.
    # Note that it will update N_allocated but not N_local.
    @cython.pheader(# Arguments
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
                # Repack particle data into pointer arrays of pointers
                # and lists of memoryviews.
                self.pos[0], self.pos[1], self.pos[2] = self.posx, self.posy, self.posz
                self.mom[0], self.mom[1], self.mom[2] = self.momx, self.momy, self.momz
                self.pos_mv = [self.posx_mv, self.posy_mv, self.posz_mv]
                self.mom_mv = [self.momx_mv, self.momy_mv, self.momz_mv]
                # Reallocate particle buffers
                #self.Δposx = realloc(self.Δposx, self.N_allocated*sizeof('double'))
                #self.Δposy = realloc(self.Δposy, self.N_allocated*sizeof('double'))
                #self.Δposz = realloc(self.Δposz, self.N_allocated*sizeof('double'))
                self.Δmomx = realloc(self.Δmomx, self.N_allocated*sizeof('double'))
                self.Δmomy = realloc(self.Δmomy, self.N_allocated*sizeof('double'))
                self.Δmomz = realloc(self.Δmomz, self.N_allocated*sizeof('double'))
                # Reassign particle buffer memory views
                #self.Δposx_mv = cast(self.Δposx, 'double[:self.N_allocated]')
                #self.Δposy_mv = cast(self.Δposy, 'double[:self.N_allocated]')
                #self.Δposz_mv = cast(self.Δposz, 'double[:self.N_allocated]')
                self.Δmomx_mv = cast(self.Δmomx, 'double[:self.N_allocated]')
                self.Δmomy_mv = cast(self.Δmomy, 'double[:self.N_allocated]')
                self.Δmomz_mv = cast(self.Δmomz, 'double[:self.N_allocated]')
                # Repack particle buffers into pointer arrays of
                # pointers and lists of memoryviews.
                self.Δpos[0], self.Δpos[1], self.Δpos[2] = self.Δposx, self.Δposy, self.Δposz
                self.Δmom[0], self.Δmom[1], self.Δmom[2] = self.Δmomx, self.Δmomy, self.Δmomz
                self.Δpos_mv = [self.Δposx_mv, self.Δposy_mv, self.Δposz_mv]
                self.Δmom_mv = [self.Δmomx_mv, self.Δmomy_mv, self.Δmomz_mv]
                # Nullify the newly allocated Δ buffer
                self.nullify_Δ()
        elif self.representation == 'fluid':
            shape_nopseudo_noghosts = size_or_shape_nopseudo_noghosts
            # The allocated shape of the fluid grids are 5 points
            # (one layer of pseudo points and two layers of ghost points
            # before and after the local grid) longer than the local
            # shape, in each direction.
            if not any([2 + s + 1 + 2 != s_old for s, s_old in zip(shape_nopseudo_noghosts,
                                                                   self.shape)]):
                return
            if any([s < 1 for s in shape_nopseudo_noghosts]):
                abort('Attempted to resize fluid grids of the {} component'
                      'to a shape of {}'.format(self.name, shape_nopseudo_noghosts))
            # Recalculate and reassign meta data
            self.shape          = tuple([2 + s + 1 + 2 for s in shape_nopseudo_noghosts])
            self.shape_noghosts = tuple([    s + 1     for s in shape_nopseudo_noghosts])
            self.size           = np.prod(self.shape)
            self.size_noghosts  = np.prod(self.shape_noghosts)
            # Reallocate fluid data
            for fluidscalar in self.iterate_fluidscalars():
                fluidscalar.resize(shape_nopseudo_noghosts)

    # Method which populate the grids of fluid variables with real-space
    # 3D realisation of linear transfer function.
    @cython.pheader(# Arguments
                    variables='object',  # str or int or container of str's and ints
                    transfer_spline='Spline',
                    cosmoresults='object',  # CosmoResults
                    a='double',
                    # Locals
                    N_vars='Py_ssize_t',
                    dim='int',
                    gridsize='Py_ssize_t',
                    i='Py_ssize_t',
                    k_gridsize='Py_ssize_t',
                    k_max='double',
                    k_min='double',
                    shape='tuple',
                    transfer_splines='list',
                    )
    def realize(self, variables=None, transfer_spline=None, cosmoresults=None, a=-1):
        """This method will realise a given fluid/particle variable from
        a given transfer function. Any existing data for the variable
        in question will be lost.
        The variables argument specifies which variable(s) of the
        component to realise. Valid formats of this argument can be seen
        in varnames2indices. The transfer_spline is a Spline object of
        the transfer function of the variable which should be realised.
        Finally, cosmoresults is a linear.CosmoResults object containing
        all results from the CLASS run which produced the transfer
        function, from which further information can be obtained.
        If only the variables argument is given, the transfer_spline and
        cosmoresults (matching the current time) will be produced by
        calling CLASS. You can supply multiple variables in one go,
        but then you have to leave the other arguments unspecified
        (as you can only pass in a single transfer_spline).
        If no arguments are passed at all, transfer functions for each
        variable will be computed via CLASS and all of them
        will be realized.
        Specify the scale factor a if you want to realize the variables
        at a time different from the present time.
        """
        # Define the gridsize used by the realization (gridsize for
        # fluid components and ∛N for particle components) and resize
        # the data attributes if needed.
        # Also do some particles-only checks.
        if self.representation == 'particles':
            if self.N%nprocs != 0:
                abort(f'Cannot perform realization of particle component "{self.name}" '
                      f'with N = {self.N}, as N is not evenly divisible by {nprocs} processes.'
                      )
            if not isint(cbrt(self.N)):
                abort(f'Cannot perform realization of particle component "{self.name}" '
                      f'with N = {self.N}, as N is not a cubic number.'
                      )
            gridsize = int(round(cbrt(self.N)))
            self.N_local = self.N//nprocs
            self.resize(self.N_local)
        elif self.representation == 'fluid':
            gridsize = self.gridsize
            shape = tuple([gridsize//domain_subdivisions[dim] for dim in range(3)])
            self.resize(shape)
        # Check that the gridsize fulfills the requirements for FFT
        # and therefore for realizations.
        if gridsize%nprocs != 0:
            abort(f'Cannot perform realization of component "{self.name}" '
                  f'with gridsize = {gridsize}, as gridsize is not '
                  f'evenly divisible by {nprocs} processes.'
                  )
        for dim in range(3):
            if gridsize%domain_subdivisions[dim] != 0:
                abort(f'Cannot perform realization of component "{self.name}" '
                      f'with gridsize = {gridsize}, as the global grid of shape '
                      f'({gridsize}, {gridsize}, {gridsize}) cannot be divided '
                      f'according to the domain decomposition ({domain_subdivisions[0]}, '
                      f'{domain_subdivisions[1]}, {domain_subdivisions[2]}).'
                      )
        # Argument processing
        if transfer_spline is None and cosmoresults is not None:
            abort('The realize method was called with a transfer_spline but no cosmoresults')
        if transfer_spline is not None and cosmoresults is None:
            abort('The realize method was called with cosmoresults but no transfer_spline')
        if self.representation == 'particles':
            # Particles use the Zeldovich approximation for realization,
            # which realizes both positions and momenta from the δ (pos)
            # transfer function. Thus for particles, regardless of what
            # variables are passed, a value of 'pos' should always
            # be used.
            variables = 'pos'
        if variables is None:
            # Realize all variables
            if transfer_spline is not None:
                masterwarn('The realize method was called without specifying a variable, '
                           'though a transfer_spline is passed. '
                           'This transfer_spline will be ignored.')
            if cosmoresults is not None:
                masterwarn('The realize method was called without specifying a variable, '
                           'though a cosmoresults is passed. This cosmoresults will be ignored.')
            variables = arange(len(self.fluidvars))
            N_vars = len(variables)
        else:
            # Realize one or more variables
            variables = any2iter(variables)
            N_vars = len(variables)
            if N_vars > 1:
                # Realize multiple variables
                if transfer_spline is not None:
                    abort(f'The realize method was called with {N_vars} variables '
                          'while a transfer_spline was supplied as well')
                if cosmoresults is not None:
                    abort(f'The realize method was called with {N_vars} variables '
                          'while cosmoresults was supplied as well')
        # Get transfer functions and cosmoresults from CLASS,
        # if not passed as arguments.
        if transfer_spline is None:
            k_min = ℝ[2*π/boxsize]
            k_max = ℝ[2*π/boxsize]*sqrt(3*(gridsize//2)**2)
            k_gridsize = 10*gridsize
            transfer_splines, cosmoresults = compute_transfers(self,
                                                               variables,
                                                               k_min, k_max, k_gridsize,
                                                               )
        else:
            transfer_splines = [transfer_spline]
        # Realize each of the variables in turn
        for i in range(N_vars):
            realize(self, variables[i], transfer_splines[i], cosmoresults, a)
            if self.representation == 'particles':
                # Particles use the Zeldovich approximation for
                # realization, which realizes both positions and
                # momenta. Thus for particles, a single realization is
                # all that is neeed. Importantly, the passed transfer
                # function must be that of δ, retrieved from
                # compute_transfers using e.g. 'pos' as the passed
                # variables argument. The value of the variable argument
                # to the realize function does not matter in the case of
                # a particle component.
                break



    # Method for integrating particle positions/fluid values
    # forward in time.
    # For fluid components, source terms are not included.
    @cython.header(# Arguments
                   ᔑdt='dict',
                   # Locals
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
            # Update positions
            for i in range(self.N_local):
                posx[i] += momx[i]*ℝ[ᔑdt['a⁻²']/self.mass]
                posy[i] += momy[i]*ℝ[ᔑdt['a⁻²']/self.mass]
                posz[i] += momz[i]*ℝ[ᔑdt['a⁻²']/self.mass]
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

    # Method for integrating fluid values forward in time
    # due to "internal" source terms, meaning source terms that do not
    # result from interactions with other components,
    # or even interactions between fluid elements within a component.
    @cython.header(# Arguments
                   ᔑdt='dict',
                   # Locals
                   ẇ='double',
                   )
    def apply_internal_sources(self, ᔑdt):
        if self.representation == 'particles':
            return
        # All internal source terms in the fluid equations
        # vanishes when ẇ = 0.
        ẇ = self.ẇ()
        if ẇ != 0:
            masterprint('Evolving fluid variables (source terms) of {} ...'.format(self.name))
            apply_internal_sources(self, ᔑdt)
            masterprint('done')

    # Method for computing the equation of state parameter w
    # at a certain time t or value of the scale factor a.
    # This has to be a pure Python function, othewise it cannot
    # be called as w(a=a).
    def w(self, t=-1, a=-1):
        """This method should not be called before w has been
        initialized by the initialize_w method.
        """
        # If no time or scale factor value is passed,
        # use the current time and scale factor value.
        if t == -1 == a:
            t = universals.t
            a = universals.a
        # Compute the current w dependent on its type
        if self.w_type == 'constant':
            value = self.w_constant
        elif self.w_type == 'tabulated (t)':
            if t == -1:
                t = cosmic_time(a)
            value = self.w_spline.eval(t)
        elif self.w_type == 'tabulated (a)':
            if a == -1:
                a = scale_factor(t)
            value = self.w_spline.eval(a)
        elif self.w_type == 'expression':
            if t == -1:
                t = cosmic_time(a)
            elif a == -1:
                a = scale_factor(t)
            units_dict['t'] = t
            units_dict['a'] = a            
            value = eval_unit(self.w_expression, units_dict)
            units_dict.pop('t')
            units_dict.pop('a')
        else:
            abort('Did not recognize w type "{}"'.format(self.w_type))
        # It may happen that w becomes slightly negative due to
        # the spline (when given as tabulated data) or rounding errors
        # (when given as an expression). Prevent this.
        if value < 0:
            if value > -1e+6*machine_ϵ:
                value = 0
            else:
                abort('Got w(t = {}, a = {}) = {}. Negative w is not implemented.'
                      .format(t, a, value)
                      )
        return value

    # Method for computing the proper time derivative
    # of the equation of state parameter w
    # at a certain time t or value of the scale factor a.
    @cython.pheader(# Arguments
                    t='double',
                    a='double',
                    # Locals
                    w_after='double',
                    w_before='double',
                    Δx='double',
                    returns='double',
                    )
    def ẇ(self, t=-1, a=-1):
        """This method should not be called before w has been
        initialized by the initialize_w method.
        """
        # If no time or scale factor value is passed,
        # use the current time and scale factor value.
        if t == -1 == a:
            t = universals.t
            a = universals.a
        # Compute the current ẇ dependent on its type
        if self.w_type == 'constant':
            return 0
        if self.w_type == 'tabulated (t)':
            return self.w_spline.eval_deriv(t)
        if self.w_type == 'tabulated (a)':
            # The chain rule: dw/dt = dw/da * da/dt
            return self.w_spline.eval_deriv(a)*ȧ(t, a)
        if self.w_type == 'expression':
            # Approximate the derivative via symmetric difference
            Δx = 1e+6*machine_ϵ
            units_dict['t'] = t - Δx
            units_dict['a'] = a - Δx           
            w_before = eval_unit(self.w_expression, units_dict)
            units_dict['t'] = t + Δx
            units_dict['a'] = a + Δx           
            w_after = eval_unit(self.w_expression, units_dict)
            units_dict.pop('t')
            units_dict.pop('a')
            return (w_after - w_before)/(2*Δx)
        abort('Did not recognize w type "{}"'.format(self.w_type))

    # Method which initializes the equation of state parameter w.
    # Call this before calling the w method.
    @cython.header(# Arguments
                   w='object',  # NoneType, float-like, str or dict
                   # Locals
                   delim_left='str',
                   delim_right='str',
                   done_reading_w='bint',
                   i='int',
                   i_tabulated='double[:]',
                   key='str',
                   line='str',
                   pattern='str',
                   spline='Spline',
                   unit='double',
                   w_data='double[:, :]',
                   w_tabulated='double[:]',
                   returns='Spline',
                   )
    def initialize_w(self, w=None):
        """The w argument can be one of the following (Python) types:
        - NoneType  : Assign a constant w depending on the species
                      of the component.
                      The w will be stored in self.w_constant and
                      self.w_type will be set to 'constant'.
        - float-like: Designates a constant w.
                      The w will be stored in self.w_constant and
                      self.w_type will be set to 'constant'.
        - str       : If w is given as a str, it can mean any of
                      three things:
                      - w may be the word 'CLASS'.
                      - w may be a filename.
                      - w may be some analytical expression.
                      If w == 'CLASS', CLASS should be used to compute
                      w throughout time. The value of species_class
                      will be used to pick out w(a) for the correct
                      species. The result from CLASS are tabulated
                      w(a), where the a values will be stored as
                      self.w_tabulated[0, :], while the tabulated values
                      of w will be stored as self.w_tabulated[1, :].
                      The self.w_type will be set to 'tabulated (a)'.
                      If w is a filename, the file should contain
                      tabulated values of w and either t or a. The file
                      must be in a format understood by numpy.loadtxt,
                      and a header must be present stating whether w(t)
                      or w(a) is tabulated. For w(t), the header should
                      also specify the units for the t column.
                      Legal formats for the header are:
                      # a    w(a)
                      # a    w
                      # t [Gyr]    w
                      # t [Gyr]    w(t)
                      In addition, the two columns may be specified in
                      the reverse order, parentheses and brackets may be
                      replaced with any of (), [], {}, <> and the number
                      of spaces/tabs does not matter.
                      The tabulated values for t or a will be stored as
                      self.w_tabulated as described above when using
                      CLASS, though self.w_type will be set to
                      'tabulated (t)' or 'tabulated (a)' depending on
                      whether t or a is used.
                      If w is some arbitrary expression,
                      this should include eather t or a.
                      The w will be stored in self.w_expression and
                      self.w_type will be set to 'expression'.
        - dict      : The dict has to be of the form
                      {'t': iterable, 'w': iterable} or
                      {'a': iterable, 'w': iterable}, where the
                      iterables are some iterables of matching
                      tabulated values.
                      The tabulated values for t or a wil be stored as
                      self.w_tabulated[0, :], the tabulated values for w
                      as self.w_tabulated[1, :] and self.w_type will be
                      set to 'tabulated (t)' or 'tabulated (a)'.
        If the user parameter w_eos contains the species, whatever is
        specified as the value there is used instead of the passed w.
        """
        # If the species has been given a w in the user
        # parameter w_eos, this will overwrite the passed w.
        if w is not None and self.species in w_eos:
            masterprint('Overwriting w = {} for the "{}" component with w = {}.'
                        .format(w, self.name, w_eos[self.species])
                        )
        w = w_eos.get(self.species, w)
        # Initialize w dependent on its type
        try:
            w = float(w)
        except:
            ...
        if w is None:
            # Assign w constant value based on the species
            self.w_type = 'constant'
            if self.species in default_w:
                self.w_constant = default_w[self.species]
            elif self.representation == 'particles':
                # The equation of state parameter w should be 0
                # for any particle component.
                self.w_constant = 0
            else:
                abort('No default w is defined for the "{}" species'.format(self.species))
        elif isinstance(w, float):
            # Assign passed constant w
            self.w_type = 'constant'
            self.w_constant = w
        elif isinstance(w, str) and w.lower() == 'class':
            # Get w as P/ρ from CLASS
            self.w_type = 'tabulated (a)'
            cosmoresults = compute_cosmo()
            # The CLASS results are only available to the master
            # process. Let it alone do the final computation.
            if master:
                background = cosmoresults.background
                i_tabulated = 1/(background['z'] + 1)
                # Cold dark matter and baryons have no pressure,
                # and CLASS does not give this as a result.
                if self.species_class in ('cdm', 'b', 'cdm+b'):
                    w_tabulated = zeros(i_tabulated.shape[0], dtype=C2np['double'])
                else:
                    w_tabulated = ( background[f'(.)p_{self.species_class}']
                                   /background[f'(.)rho_{self.species_class}'])
        elif isinstance(w, str) and os.path.isfile(w):
            # Load tabulated w from file.
            self.w_type = 'tabulated (?)'
            # Let only the master process read in the file
            if master:
                w_data = np.loadtxt(w)
                # Transpose w_data so that it consists of two rows
                w_data = w_data.T
                # For varying w it is crucial to know whether w is a
                # function of t or a.
                # This should be written in the header of the file.
                pattern = r'[a-zA-Z]*\s*(?:\(\s*[a-zA-Z0-9\.]+\s*\))?'
                done_reading_w = False
                with open(w, 'r', encoding='utf-8') as w_file:
                    while True:
                        line = w_file.readline().lstrip()
                        if line and not line.startswith('#'):
                            break
                        line = line.replace('#', '').replace('\t', ' ').replace('\n', ' ')
                        for delim_left, delim_right in zip('[{<', ']}>'):
                            line = line.replace(delim_left , '(')
                            line = line.replace(delim_right, ')')
                        match = re.search(r'\s*({pattern})\s+({pattern})\s*(.*)'.format(pattern=pattern),
                                          line)                
                        if match and not match.group(3):
                            # Header line containing the relevant
                            # information found.
                            for i in range(2):
                                var = match.group(1 + i)
                                if var[0] == 't':
                                    self.w_type = 'tabulated (t)'
                                    unit_match = re.search('\((.*)\)', var)  # Will be applied later                               
                                elif var[0] == 'a':
                                    self.w_type = 'tabulated (a)'
                                elif var[0] == 'w':
                                    # Extract the two rows
                                    w_tabulated = w_data[i, :]
                                    i_tabulated = w_data[(i + 1) % 2, :]
                                    done_reading_w = True
                        if done_reading_w and '(?)' not in self.w_type:
                            break
                # Multiply unit on the time
                if '(t)' in self.w_type:
                    if unit_match and unit_match.group(1) in units_dict:
                        unit = eval_unit(unit_match.group(1)) 
                        i_tabulated = asarray(i_tabulated)*unit
                    elif unit_match:
                        abort('Time unit "{}" in header of "{}" not understood'.format(unit_match.group(1), w))
                    else:
                        abort('Could not find time unit in header of "{}"'.format(w))
        elif isinstance(w, str):
            # Save the passed expression for w
            self.w_type = 'expression'
            self.w_expression = w
            # Test that the expression is parsable
            try:
                self.w()
            except:
                abort(f'Cannot parse w = "{self.w_expression}" as an expression.\n'
                      'No file with that name can be found either.')
        elif isinstance(w, dict):
            # Use the tabulated w given by the two dict key-value pairs
            self.w_type = 'tabulated (?)'
            for key, val in w.items():
                if key.startswith('t'):
                    self.w_type = 'tabulated (t)'
                    i_tabulated = asarray(val, dtype=C2np['double'])
                elif key.startswith('a'):
                    self.w_type = 'tabulated (a)'
                    i_tabulated = asarray(val, dtype=C2np['double'])
                elif key.startswith('w'):
                    w_tabulated = asarray(val, dtype=C2np['double'])
        else:
            abort('Cannot handle w of type {}'.format(type(w)))
        # In the case of a tabulated w, only the master process
        # have the needed i_tabulated and w_tabulated. Broadcast these
        # to the slaves and instantiate a spline on all processes.
        if 'tabulated' in self.w_type:
            if master:
                # Check that tabulated values have been found
                if '(?)' in self.w_type:
                    abort('Could not detect the independent variable (should be \'a\' or \'t\')')
                # Make sure that the values of i_tabulated are in increasing order
                order = np.argsort(i_tabulated)
                i_tabulated = asarray(i_tabulated)[order]
                w_tabulated = asarray(w_tabulated)[order]
                # Pack the two rows together
                self.w_tabulated = empty((2, w_tabulated.shape[0]))
                self.w_tabulated[0, :] = i_tabulated
                self.w_tabulated[1, :] = w_tabulated
            # Broadcast the tabulated w
            self.w_type = bcast(self.w_type)
            self.w_tabulated = smart_mpi(self.w_tabulated if master else (), mpifun='bcast')
            # Construct a Spline object from the tabulated data
            self.w_spline = Spline(self.w_tabulated[0, :], self.w_tabulated[1, :])

    # Method which convert named fluid/particle variables to indices
    @cython.header(# Arguments
                   varnames='object',  # str, int or container of str's and ints 
                   single='bint',
                   # Locals
                   N_vars='Py_ssize_t',
                   i='Py_ssize_t',
                   indices='Py_ssize_t[::1]',
                   varname='object', # str or int
                   varnames_list='list',
                   returns='object',  # Py_ssize_t[::1] or Py_ssize_t
                   )
    def varnames2indices(self, varnames, single=False):
        """This method conveniently transform any reasonable input
        to an array of variable indices. Some examples:
        varnames2indices('ϱ') → asarray([0])
        varnames2indices(['J', 'ϱ']) → asarray([1, 0])
        varnames2indices(['pos', 'mom']) → asarray([0, 1])
        varnames2indices(2) → asarray([2])
        varnames2indices(['σ', 1]) → asarray([2, 1])
        """
        if isinstance(varnames, str):
            # Single variable name given
            indices = asarray([self.fluid_names[varnames]], dtype=C2np['Py_ssize_t'])
        else:
            # One or more variable names/indices given
            varnames_list = any2iter(varnames)
            indices = empty(len(varnames_list), dtype=C2np['Py_ssize_t'])
            for i, varname in enumerate(varnames_list):
                indices[i] = self.fluid_names[varname]
        if single:
            N_vars = indices.shape[0]
            if N_vars > 1:
                abort(f'The varnames2indices method was called '
                      'with {N_vars} variables while single == True')
            return indices[0]
        return indices

    # Generator for looping over all
    # scalar fluid grids within the component.
    def iterate_fluidscalars(self):
        for fluidvar in self.fluidvars:
            yield from self.iterate_fluidscalar(fluidvar)

    # Generator for looping over all
    # scalar fluid grids of a fluid variable.
    def iterate_fluidscalar(self, fluidvar):
        for multi_index in self.iterate_fluidscalar_indices(fluidvar):
            yield fluidvar[multi_index]

    # Generator for looping over all multi-indices of a fluid variable.
    # The yielded multi_index is always a tuple.
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

    # Method which calls the scale_grid on all fluid scalars
    @cython.header(# Arguments
                   a='double',
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def scale_fluid_grid(self, a):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            fluidscalar.scale_grid(a)

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
    @cython.header(# Arguments
                   specifically='object',  # str og container of str's
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_Δ(self, specifically=None):
        if self.representation == 'particles':
            if specifically is None:
                self.Δposx_mv[...] = 0
                self.Δposy_mv[...] = 0
                self.Δposz_mv[...] = 0
                self.Δmomx_mv[...] = 0
                self.Δmomy_mv[...] = 0
                self.Δmomz_mv[...] = 0
            else:
                if 'pos' in specifically:
                    self.Δposx_mv[...] = 0
                    self.Δposy_mv[...] = 0
                    self.Δposz_mv[...] = 0
                if 'mom' in specifically:
                    self.Δmomx_mv[...] = 0
                    self.Δmomy_mv[...] = 0
                    self.Δmomz_mv[...] = 0
        elif self.representation == 'fluid':
            for fluidscalar in self.iterate_fluidscalars():
                fluidscalar.nullify_Δ()

    # This method is automaticlly called when a Component instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        # Free particle data
        # (fluid data lives on FluidScalar instances).
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



# Mapping from species to their representations
cython.declare(representation_of_species='dict')
representation_of_species = {('dark matter particles',
                              'baryons',
                              'matter particles',
                              'neutrinos',
                              ): 'particles',
                             ('dark matter fluid',
                              'baryon fluid',
                              'matter fluid',
                              'neutrino fluid',
                              ): 'fluid',
                             }
# Mapping from particle species to default softening factors
cython.declare(default_softeningfactors='dict')
default_softeningfactors = {'dark matter particles': 0.03,
                            'baryons'              : 0.03,
                            'matter particles'     : 0.03,
                            'neutrinos'            : 0.03,
                            }
# Mapping from species to default w values
cython.declare(default_w='dict')
default_w = {'dark matter particles': 0,
             'dark matter fluid'    : 0,
             'baryons'              : 0,
             'baryon fluid'         : 0,
             'matter particles'     : 0,
             'matter fluid'         : 0,
             'neutrinos'            : 1/3,
             'neutrino fluid'       : 1/3,
             }
# Mapping from species to default species names in CLASS
cython.declare(default_species_class='dict')
default_species_class = {'dark matter particles': 'cdm',
                         'dark matter fluid'    : 'cdm',
                         'baryons'              : 'b',
                         'baryon fluid'         : 'b',
                         'matter particles'     : 'cdm+b',
                         'matter fluid'         : 'cdm+b',
                         'neutrinos'            : 'ncdm[0]',
                         'neutrino fluid'       : 'ncdm[0]',
                         }
# Set of all component names used internally by the code,
# and which the user should generally avoid.
cython.declare(internally_defined_names='set')
internally_defined_names = {'buffer', 'total'}

