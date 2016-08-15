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
import gravity
cimport('from communication import communicate_domain_boundaries, communicate_domain_ghosts, exchange')
cimport('from gravity import PM, kick_fluid')
cimport('from integration import maccormack')



# Class which serves as the data structure for a scalar fluid grid
# (each component of a fluid variable is stored as a collection of
# scalar grids). Each scalar fluid has its own name, e.g.
# ϱ     (varnum == 0, multi_index == 0),
# ϱu[0] (varnum == 1, multi_index == 0),
# ϱu[1] (varnum == 1, multi_index == 1),
# ϱu[2] (varnum == 1, multi_index == 2),
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
        # The source term buffer
        double* source
        public double[:, :, ::1] source_mv
        public double[:, :, :] source_noghosts
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
        # Source term buffer
        self.source = malloc(size*sizeof('double'))
        self.source_mv = cast(self.source, 'double[:shape[0], :shape[1], :shape[2]]')
        self.source_noghosts = self.source_mv[:, :, :]

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
        # The starred buffer
        self.gridˣ = realloc(self.gridˣ, size*sizeof('double'))
        self.gridˣ_mv = cast(self.gridˣ, 'double[:shape[0], :shape[1], :shape[2]]')
        self.gridˣ_noghosts = self.gridˣ_mv[2:(self.gridˣ_mv.shape[0] - 2),
                                            2:(self.gridˣ_mv.shape[1] - 2),
                                            2:(self.gridˣ_mv.shape[2] - 2)]
        # The source term buffer for the ϱu variable
        if self.varnum == 1:
            self.source = realloc(self.source, size*sizeof('double'))
            self.source_mv = cast(self.source, 'double[:shape[0], :shape[1], :shape[2]]')
            self.source_noghosts = self.source_mv[2:(self.source_mv.shape[0] - 2),
                                                  2:(self.source_mv.shape[1] - 2),
                                                  2:(self.source_mv.shape[2] - 2)]

    # Method for communicating pseudo and ghost points of grids
    @cython.pheader()
    def communicate_grid(self):
        """The entire local grid should already be constructed.
        Whether the pseudo and grid points presently holds
        correct values or not is irrelevant.
        """
        communicate_domain_boundaries(self.grid_mv, mode=1)
        communicate_domain_ghosts(self.grid_mv)

    # Method for communicating pseudo and ghost points of source buffers
    @cython.pheader()
    def communicate_source(self):
        """The entire local grid of source values should already be
        constructed. Whether the pseudo and grid points presently holds
        correct values or not is irrelevant.
        """
        communicate_domain_boundaries(self.source_mv, mode=1)
        communicate_domain_ghosts(self.source_mv)

    # Method for nullifying the starred grid
    @cython.pheader(# Locals
                    i='Py_ssize_t',
                    gridˣ='double*',
                    )
    def nullify_gridˣ(self):
        # Extract starred buffer pointer
        gridˣ = self.gridˣ
        # Nullify starred buffer
        for i in range(np.prod(asarray(self.gridˣ_mv).shape)):
            gridˣ[i] = 0

    # Method for nullifying the source term buffer
    @cython.pheader(# Locals
                    i='Py_ssize_t',
                    source='double*',
                    )
    def nullify_source(self):
        # Extract source term buffer pointer
        source = self.source
        # Nullify source term buffer
        for i in range(np.prod(asarray(self.source_mv).shape)):
            source[i] = 0

    # Method for nullifying all buffers
    @cython.pheader()
    def nullify_buffers(self):
        self.nullify_gridˣ()
        self.nullify_source()

    # This method is automaticlly called when a FluidScalar instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        free(self.grid)
        free(self.gridˣ)
        free(self.source)

    # String representation
    def __str__(self):
        if self.varnum < len(fluidvarnames):
            name = fluidvarnames[self.varnum]
        else:
            name = str(self.varnum)
        if self.varnum > 0 and self.multi_index:
            return '<fluidscalar {}[{}]>'.format(name,
                                                 ', '.join([str(i) for i in self.multi_index]))
        else:
            return '<fluidscalar {}>'.format(name)


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
                   N_fluidvars='Py_ssize_t',
                   )
    def __init__(self, name, species, N_or_gridsize, mass, N_fluidvars=2):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Component type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        # General component attributes
        public str name
        public str species
        public str representation
        public double mass
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
        self.name           = name
        self.species        = species
        self.mass           = mass
        # Determine the representation based on the species
        self.representation = get_representation(self.species)
        # Particle component attributes
        self.N_allocated = 1
        self.N_local     = 1
        if self.representation == 'particles':
            self.N = N_or_gridsize
            if self.species in softeningfactors:
                self.softening = softeningfactors[self.species]*boxsize/(self.N**ℝ[1/3])
            elif master:
                abort('Species "{}" do not have an assigned softening factor!'
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
        # Fluid component attributes
        if self.representation == 'fluid':
            self.gridsize = N_or_gridsize
        else:
            self.gridsize = 1
        # Construct the fluidvars dict, storing all fluid variables
        # and meta data. 'N' is the number of fluid variables and
        # 'shape' and 'shape_noghosts' are the shape of the fluid scalar
        # grids, with and without ghost points, respectively (both
        # include pseudo points).
        self.fluidvars = {'N'             : N_fluidvars,
                          'shape'         : (1, 1, 1),
                          'shape_noghosts': (1, 1, 1),
                          }
        # Create the two lowest order fluid variables (ϱ and ϱu)
        # SHOULD BE DEPENDENT ON THE PARSED N_fluidvars !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
            # inclde pseudo or ghost points.
            tmp = self.fluidvars[var]
            if isinstance(tmp, FluidScalar):
                fluidscalar = tmp
            else:
                fluidscalar = tmp[indices]
            fluidscalar.grid_noghosts[:mv3D.shape[0], :mv3D.shape[1], :mv3D.shape[2]] = mv3D[...]

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
            if any([2 + s + 1 + 2 != s_old for s, s_old in zip(shape_nopseudo_noghosts,
                                                               self.fluidvars['shape'])]):
                if any([s < 1 for s in shape_nopseudo_noghosts]):
                    abort('Attempted to resize fluid grids of the {} component'
                          'to a shape of {}'.format(self.name, shape_nopseudo_noghosts))
                # Reassign the shape meta data
                self.fluidvars['shape']          = tuple([2 + s + 1 + 2 for s in shape_nopseudo_noghosts])
                self.fluidvars['shape_noghosts'] = tuple([    s + 1     for s in shape_nopseudo_noghosts])
                # Reallocate fluid data
                for fluidscalar in self.iterate_fluidscalars():
                    fluidscalar.resize(shape_nopseudo_noghosts)

    # Method for integrating particle positions/fluid values
    # forward in time.
    @cython.header(# Arguments
                   ᔑdt='dict',
                   # Locals
                   dim='int',
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
            # The factor 1/mass*∫_t^(t + Δt)a⁻²dt
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
            # Communicate the pseudo and ghost points
            # of all fluid variables.
            self.communicate_fluid_grids()
            masterprint('done')

    # Method for kicking particles and fluids
    @cython.pheader(# Arguments
                    ᔑdt='dict',
                    meshbuf_mv='double[:, :, ::1]',
                    dim='int',
                    # Locals
                    kick_algorithm='str',
                    ϱu_dim='FluidScalar',
                    )
    def kick(self, ᔑdt, meshbuf_mv=None, dim=-1):
        """In the case of a component carrying particles, a 'kick' is a
        complete update of all the particles momenta (momx, momy
        and momz). When the assigned kick algorithm is 'PM', a complete
        kick is achieved by calling this function three times, one for
        each dimension (dim == 0, dim == 1, dim == 2). For all other
        kick algorithms, a complete kick is achieved by a single call.
        These other algorithms do not need anything other than ᔑdt,
        while 'PM' also needs meshbuf (storing the values of
        [-∇φ]_dim) and dim.
        In the case of a fluid component, a 'kick' is not a complete
        update of the velocity field, or any other fluid variable. It is
        merely the computation of the source term -∇φ*ϱ.
        Fluid components do not have a choice for the kick algorithm,
        as this is the only implemented way for a fluid to receive
        gravitational forces. As in the 'PM' method for particles,
        meshbuf_mv and dim need to be parsed and only the dim'th
        component are updated (ϱux, ϱuy or ϱuz). These updates are
        stored in the special source buffers on the three fluid scalars.
        """
        if self.representation == 'particles':
            # The assigned kick algorithm
            kick_algorithm = kick_algorithms[self.species]
            if kick_algorithm == 'PM':
                # For the PM algoritm, call the PM function with
                # all the supplied arguments.
                PM(self, ᔑdt, meshbuf_mv, dim)
            else:
                # For all other kick algorithms, the kick is done
                # completely in one go.
                # Write out progess message and do the kick.
                masterprint('Kicking ({}) {} ...'.format(kick_algorithm, self.name))
                getattr(gravity, kick_algorithm)(self, ᔑdt)
                masterprint('done')
        elif self.representation == 'fluid':
            # Extract fluid scalar ϱu[dim], the source buffer of which
            # should be updated.
            ϱu_dim = self.fluidvars['ϱu'][dim]
            # Nullify source buffer
            ϱu_dim.nullify_source()
            # Interpolate (-∇φ)[dim] to the source buffer of ϱu_dim
            kick_fluid(self, meshbuf_mv, dim)
            # Communicate the pseudo and ghost points of source buffer
            # of ϱu_dim.
            ϱu_dim.communicate_source()

    # Method which assigns convenient names to some
    # fluid variables and fluid scalars.
    @cython.header(# Locals
                   dim='int',
                   l='Py_ssize_t',
                   )
    def assign_fluidnames(self):
        """After running this method,
        fluid scalars can be accessed as e.g.
        self.fluidvars['ϱ']   == self.fluidvars[0][0]
        self.fluidvars['ϱux'] == self.fluidvars['ϱu'][0] == self.fluidvars[1][0]
        self.fluidvars['ϱuy'] == self.fluidvars['ϱu'][1] == self.fluidvars[1][1]
        self.fluidvars['ϱuz'] == self.fluidvars['ϱu'][2] == self.fluidvars[1][2]
        """
        # As higher fluid variables may not have names in general,
        # enclose the assignments in a try block. The assignments
        # should be ordered accoring to the fluid variable number.
        try:
            for l in range(self.fluidvars['N']):
                if l == 0:
                    # ϱ
                    self.fluidvars[fluidvarnames[l]] = self.fluidvars[l][0]
                elif l == 1:
                    # ϱu
                    self.fluidvars[fluidvarnames[l]] = self.fluidvars[l]
                    for dim in range(3):
                        self.fluidvars[fluidvarnames[l] + 'xyz'[dim]] = self.fluidvars[l][dim]
        except:
            ...

    # Generator for looping over all the scalar fluid grids within
    # the component.
    def iterate_fluidscalars(self):
        fluidvars = self.fluidvars
        for l in range(fluidvars['N']):
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
    # on all fluid scalars.
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def communicate_fluid_grids(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            fluidscalar.communicate_grid()

    # Method for communicating pseudo and ghost points
    # on all fluid scalars.
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def communicate_fluid_sources(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            fluidscalar.communicate_source()

    # Method which calls the nullify_gridˣ on all fluid scalars
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_fluid_gridˣ(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            fluidscalar.nullify_gridˣ()

    # Method which calls the nullify_source on all fluid scalars
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_fluid_source(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            fluidscalar.nullify_source()

    # Method for nullifying all buffers on all fluid scalars
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_fluid_buffers(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            fluidscalar.nullify_buffers()

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
fluidvarnames = ('ϱ', 'ϱu')
