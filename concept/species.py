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
cimport('from communication import exchange')
cimport('from mesh import diff')
cimport('from gravity import PM, kick_fluid')


# Class which serves as the data structure for a scalar fluid grid
# (each component of a fluid variable is stored as a collection of
# scalar grids).
@cython.cclass
class FluidScalar:
    # Initialization method
    @cython.header(# Locals
                   shape='tuple',
                   size='Py_ssize_t',
                   )
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the FluidScalar type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        # The data itself
        double* grid
        public double[:, :, ::1] grid_mv
        public double[:, :, :] grid_noghosts
        # The update buffer
        double* Δ
        public double[:, :, ::1] Δ_mv
        # The differentiation buffers
        double* diffx
        double* diffy
        double* diffz
        public double[:, :, ::1] diffx_mv
        public double[:, :, ::1] diffy_mv
        public double[:, :, ::1] diffz_mv
        """
        shape = tuple([1]*3)
        size = shape[0]*shape[1]*shape[2]
        # The data itself
        self.grid = malloc(size*sizeof('double'))
        self.grid_mv = cast(self.grid, 'double[:shape[0], :shape[1], :shape[2]]')
        self.grid_noghosts = self.grid_mv[:, :, :]
        # The update buffer
        self.Δ = malloc(size*sizeof('double'))
        self.Δ_mv = cast(self.Δ, 'double[:shape[0], :shape[1], :shape[2]]')
        # The differentiation buffers
        self.diffx = malloc(size*sizeof('double'))
        self.diffy = malloc(size*sizeof('double'))
        self.diffz = malloc(size*sizeof('double'))
        self.diffx_mv = cast(self.diffx, 'double[:shape[0], :shape[1], :shape[2]]')
        self.diffy_mv = cast(self.diffy, 'double[:shape[0], :shape[1], :shape[2]]')
        self.diffz_mv = cast(self.diffz, 'double[:shape[0], :shape[1], :shape[2]]')

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
        shape = tuple([s + 1 + 2*2 for s in shape_nopseudo_noghost])
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
        # The update buffer (no ghost points)
        self.Δ = realloc(self.Δ, size_noghosts*sizeof('double'))
        self.Δ_mv = cast(self.Δ, 
                         'double[:shape_noghosts[0], :shape_noghosts[1], :shape_noghosts[2]]')
        # The differentiation buffers (no ghost points)
        self.diffx = realloc(self.diffx, size_noghosts*sizeof('double'))
        self.diffy = realloc(self.diffy, size_noghosts*sizeof('double'))
        self.diffz = realloc(self.diffz, size_noghosts*sizeof('double'))
        self.diffx_mv = cast(self.diffx,
                             'double[:shape_noghosts[0], :shape_noghosts[1], :shape_noghosts[2]]')
        self.diffy_mv = cast(self.diffy,
                             'double[:shape_noghosts[0], :shape_noghosts[1], :shape_noghosts[2]]')
        self.diffz_mv = cast(self.diffz,
                             'double[:shape_noghosts[0], :shape_noghosts[1], :shape_noghosts[2]]')

    # Method for nullifying the buffers attached to the fluid scalar
    # (the update buffer Δ and the three differentiation buffers
    # diffx, diffy, diffz).
    @cython.header(# Locals
                   diffx='double*',
                   diffy='double*',
                   diffz='double*',
                   i='Py_ssize_t',
                   size_noghosts='Py_ssize_t',
                   Δ='double*',
                   )
    def nullify_buffers(self):
        # Extract buffer pointers
        Δ = self.Δ
        diffx = self.diffx
        diffy = self.diffy
        diffz = self.diffz
        # All of the buffers have the same size
        size_noghosts = np.prod(asarray(self.Δ_mv).shape)
        # Nullify
        for i in range(size_noghosts):
            Δ[i] = 0
            diffx[i] = 0
            diffy[i] = 0
            diffz[i] = 0

    # This method is automaticlly called when a FluidScalar instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        free(self.grid)
        free(self.Δ)
        free(self.diffx)
        free(self.diffy)
        free(self.diffz)

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
                   )
    def __init__(self, name, species, N_or_gridsize, mass):
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
        double[::1] posx_mv
        double[::1] posy_mv
        double[::1] posz_mv
        double[::1] momx_mv
        double[::1] momy_mv
        double[::1] momz_mv
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
        # Fluid component attributes
        if self.representation == 'fluid':
            self.gridsize = N_or_gridsize
        else:
            self.gridsize = 1
        # Construct the two lowest order fluid variables (δ and u)
        self.fluidvars = {'N': 2}  # N is the number of fluid variables
        self.fluidvars[0] = asarray([FluidScalar()], dtype='object')
        self.fluidvars[1] = asarray([FluidScalar() for i in range(3)], dtype='object')
        # Also assign some convenient names for the fluid grids
        self.assign_fluidnames()

    # This method populate the Component pos/mom arrays (for a
    # particles representation) or the δ/u arrays (for a fluid
    # representation) with data. It is deliberately designed so that
    # you have to make a call for each attribute. You should construct
    # the data array within the call itself, as this will minimize
    # memory usage. This data array is 1D for particle data and 3D
    # for fluid data.
    @cython.header(# Arguments
                   data='object',  # 1D/3D (particles/fluid) memoryview
                   var='object',   # str or int-like
                   indices='object',  # Int-like or tuple
                   # Locals
                   fluidscalar='FluidScalar',
                   mv1D='double[::1]',
                   mv3D='double[:, :, ::1]',
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
            if isinstance(var, str):
                fluidscalar = self.fluidvars[var]
            else:
                fluidscalar = self.fluidvars[var][indices]
            fluidscalar.grid_noghosts[:mv3D.shape[0], :mv3D.shape[1], :mv3D.shape[2]] = mv3D[...]

    # This method will grow/shrink the data attributes.
    # Note that it will update N_allocated but not N_local.
    @cython.header(# Arguments
                   shape='object',  # Py_ssize_t or tuple
                   # Locals
                   N_allocated='Py_ssize_t',
                   fluidscalar='FluidScalar',
                   msg='str',
                   s='Py_ssize_t',
                   size='Py_ssize_t',
                   δ_mv='double[:, :, ::1]',
                   )
    def resize(self, shape):
        if self.representation == 'particles':
            if shape != self.N_allocated:
                self.N_allocated = shape
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
        elif self.representation == 'fluid':
            # The allocated shape of the fluid grids are 5 points
            # (one layer of pseudo points and two layers of ghost points
            # before and after the logical grid) longer than the logical
            # shape, in each direction.
            δ_mv = self.fluidvars['δ'].grid_mv
            if (   shape[0] + 1 + 2*2 != δ_mv.shape[0]
                or shape[1] + 1 + 2*2 != δ_mv.shape[1]
                or shape[2] + 1 + 2*2 != δ_mv.shape[2]):
                if any([s < 1 for s in shape]):
                    msg = ('Attempted to resize fluid grids of the {} component\n'
                           'to a shape of {}.').format(self.name, shape)
                    abort(msg)
                # Reallocate fluid data
                for fluidscalar in self.iterate_fluidvars():
                    fluidscalar.resize(shape)

    # Method for integrating particle positions/fluid values
    # forward in time.
    @cython.header(# Arguments
                   a_integrals='dict',
                   # Locals
                   diffx_mv='double[:, :, ::1]',
                   diffy_mv='double[:, :, ::1]',
                   diffz_mv='double[:, :, ::1]',
                   fac='double',
                   fluidscalar='FluidScalar',
                   grid_mv='double[:, :, ::1]',
                   h='double',
                   i='Py_ssize_t',
                   posx='double*',
                   posy='double*',
                   posz='double*',
                   momx='double*',
                   momy='double*',
                   momz='double*',
                   )
    def drift(self, a_integrals):
        """a_integral is ∫_t^(t + Δt)a⁻²dt
        """
        if self.representation == 'particles':
            masterprint('Drifting {} ...'.format(self.name))
            posx = self.posx
            posy = self.posy
            posz = self.posz
            momx = self.momx
            momy = self.momy
            momz = self.momz
            # The factor 1/mass*∫_t^(t + Δt)a⁻²dt
            fac = a_integrals['a⁻²']/self.mass
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
            masterprint('Evolving fluid variables of {} ...'.format(self.name))
            # Communicate pseudo and ghost points of fluid grids


            # Pre-tabulate all three differentiations of each fluid
            # scalar and store the results in the
            # designated diff buffers.
            # The physical grid spacing h is the same in all directions.
            h = boxsize/self.gridsize
            for fluidscalar in self.iterate_fluidvars():
                # Extract grids
                grid_mv = fluidscalar.grid_mv
                diffx_mv = fluidscalar.diffx_mv
                diffy_mv = fluidscalar.diffy_mv
                diffz_mv = fluidscalar.diffz_mv
                # Do the differentiation
                diff(grid_mv, 0, h, diffx_mv)
                diff(grid_mv, 1, h, diffy_mv)
                diff(grid_mv, 2, h, diffz_mv)
            # Evolve the fluid variables


            masterprint('done')
        
    # Method for kicking particles and fluids
    @cython.pheader(# Arguments
                    a_integrals='dict',
                    meshbuf_mv='double[:, :, ::1]',
                    dim='int',
                    # Locals
                    kick_algorithm='str',
                    )
    def kick(self, a_integrals, meshbuf_mv=None, dim=-1):
        """In the case of a component carrying particles, a 'kick' is a
        complete update of all the particles momenta (momx, momy
        and momz). When the assigned kick algorithm is 'PM', a complete
        kick is achieved by calling this function three times, one for
        each dimension (dim == 0, dim == 1, dim == 2). For all other
        kick algorithms, a complete kick is achieved by a single call.
        These other algorithms do not need anything other than
        a_integral, while 'PM' also needs meshbuf (storing the values of
        [-∇φ]_dim) and dim.
        In the case of a fluid component, a 'kick' is not a complete
        update of the velocity field, or any other fluid variable. It is
        merely the computation of the -∇φ∫_t^(t + Δt)a⁻²dt part of Δu.
        Fluid components do not have a choice for the kick algorithm,
        as this is the only implemented way for a fluid to receive
        gravitational forces. As in the 'PM' method for particles,
        meshbuf_mv and dim need to be parsed and only the dim'th
        component (Δux, Δuy or Δuz) are updated.
        """
        if self.representation == 'particles':
            # The assigned kick algorithm
            kick_algorithm = kick_algorithms[self.species]
            if kick_algorithm == 'PM':
                # For the PM algoritm, call the PM function with
                # all the supplied arguments.
                PM(self, a_integrals['a⁻¹'], meshbuf_mv, dim)
            else:
                # For all other kick algorithms, the kick is done
                # completely in one go.
                # Write out progess message and do the kick.
                masterprint('Kicking ({}) {} ...'.format(kick_algorithm, self.name))
                getattr(gravity, kick_algorithm)(self, a_integrals['a⁻¹'])
                masterprint('done')
        elif self.representation == 'fluid':
            # Interpolate [-∇φa⁻²]_dim to the dim'th Δu grid
            kick_fluid(self, a_integrals['a⁻²'], meshbuf_mv, dim)

    # Method which assigns convenient names to some
    # fluid variables and fluid scalars.
    @cython.header()
    def assign_fluidnames(self):
        # As some names fluid variables do not exist in general,
        # enclose the assignments in a try block. The assignments
        # should be ordered accoring to the fluid variable number.
        try:
            self.fluidvars['δ']  = self.fluidvars[0][0]
            self.fluidvars['u']  = self.fluidvars[1]
            self.fluidvars['ux'] = self.fluidvars['u'][0]
            self.fluidvars['uy'] = self.fluidvars['u'][1]
            self.fluidvars['uz'] = self.fluidvars['u'][2]
        except:
            pass

    # Method for nullifying all fluid buffers of a component with the
    # fluid representation. This include the update buffer (Δ) and the
    # three differentiation buffers (diffx, diffy, diffz).
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_fluid_buffers(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidvars():
            fluidscalar.nullify_buffers()

    # Generator for looping over all the scalar fluid grids within
    # the component.
    def iterate_fluidvars(self):
        fluidvars = self.fluidvars
        for l in range(fluidvars['N']):
            fluidvar = fluidvars[l]
            for multi_index in self.iterate_fluidvar(fluidvar):
                yield fluidvar[multi_index]

    # Generator for looping over all multi-indices of a fluid variable
    @staticmethod
    def iterate_fluidvar(fluidvar):
        it = np.nditer(fluidvar, flags=('refs_ok', 'multi_index'))
        while not it.finished:
            yield it.multi_index
            it.iternext()

    # This method is automaticlly called when a Component instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        free(self.posx)
        free(self.posy)
        free(self.posz)
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

