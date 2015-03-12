# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from gravity import PP, PM
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from gravity cimport PP, PM
    """

# Hej Peter
# The class representing a collection of particles of a given type
@cython.cclass
class Particles:
    """An instance of this class represents a collection of particles
    of a definite type. Only one instance of any Particles type may be
    instantiated in a run. A Particles instance of a given type should be
    present on all processes.
    All species share the same class (this one). The difference is purely in
    their "type" and "species" attributes. The "species" attribute is used as
    a flag to allow different species to behave differently.
    """

    # Initialization method.
    # Note that data attributes are declared in the .pxd file.
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   N='size_t',
                   )
    def __init__(self, N):
        # Store particle meta data
        self.N = N
        self.N_allocated = 1
        self.N_local = 1
        self.mass = 1*units.m_sun
        self.softening = 1*units.kpc
        self.species = 'generic particle species'
        self.type = 'generic particle type'
        # Manually allocate memory for particle data
        self.posx = malloc(self.N_allocated*sizeof('double'))
        self.posy = malloc(self.N_allocated*sizeof('double'))
        self.posz = malloc(self.N_allocated*sizeof('double'))
        self.momx = malloc(self.N_allocated*sizeof('double'))
        self.momy = malloc(self.N_allocated*sizeof('double'))
        self.momz = malloc(self.N_allocated*sizeof('double'))
        # Memory views around the allocated data
        self.posx_mw = cast(self.posx, 'double[:self.N_allocated]')
        self.posy_mw = cast(self.posy, 'double[:self.N_allocated]')
        self.posz_mw = cast(self.posz, 'double[:self.N_allocated]')
        self.momx_mw = cast(self.momx, 'double[:self.N_allocated]')
        self.momy_mw = cast(self.momy, 'double[:self.N_allocated]')
        self.momz_mw = cast(self.momz, 'double[:self.N_allocated]')

    # This method populate the Particles pos/mom attributes with data.
    # It is deliberately designed so that you have to make a call for each
    # attribute. You should consruct the mw array within the call itself,
    # as this will minimize memory usage.
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   mw='double[::1]',
                   coord='str',
                   )
    def populate(self, mw, coord):
        self.N_allocated = mw.size
        self.N_local = self.N_allocated
        # Update the attribute corresponding to the passed string
        if coord == 'posx':
            self.posx = realloc(self.posx, self.N_allocated*sizeof('double'))
            self.posx_mw = cast(self.posx, 'double[:self.N_local]')
            self.posx_mw[...] = mw[...]
        elif coord == 'posy':
            self.posy = realloc(self.posy, self.N_allocated*sizeof('double'))
            self.posy_mw = cast(self.posy, 'double[:self.N_local]')
            self.posy_mw[...] = mw[...]
        elif coord == 'posz':
            self.posz = realloc(self.posz, self.N_allocated*sizeof('double'))
            self.posz_mw = cast(self.posz, 'double[:self.N_local]')
            self.posz_mw[...] = mw[...]
        elif coord == 'momx':
            self.momx = realloc(self.momx, self.N_allocated*sizeof('double'))
            self.momx_mw = cast(self.momx, 'double[:self.N_local]')
            self.momx_mw[...] = mw[...]
        elif coord == 'momy':
            self.momy = realloc(self.momy, self.N_allocated*sizeof('double'))
            self.momy_mw = cast(self.momy, 'double[:self.N_local]')
            self.momy_mw[...] = mw[...]
        elif coord == 'momz':
            self.momz = realloc(self.momz, self.N_allocated*sizeof('double'))
            self.momz_mw = cast(self.momz, 'double[:self.N_local]')
            self.momz_mw[...] = mw[...]
        else:
            raise ValueError('Wrong attribute name "' + coord + '"!')

    # This method will grow/shrink the data attributes.
    # Note that it will not update the N_local attribute.
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   N_allocated='size_t',
                   )
    def resize(self, N_allocated):
        self.N_allocated = N_allocated
        # Reallocate data
        self.posx = realloc(self.posx, self.N_allocated*sizeof('double'))
        self.posy = realloc(self.posy, self.N_allocated*sizeof('double'))
        self.posz = realloc(self.posz, self.N_allocated*sizeof('double'))
        self.momx = realloc(self.momx, self.N_allocated*sizeof('double'))
        self.momy = realloc(self.momy, self.N_allocated*sizeof('double'))
        self.momz = realloc(self.momz, self.N_allocated*sizeof('double'))
        # Reassign memory views
        self.posx_mw = cast(self.posx, 'double[:self.N_allocated]')
        self.posy_mw = cast(self.posy, 'double[:self.N_allocated]')
        self.posz_mw = cast(self.posz, 'double[:self.N_allocated]')
        self.momx_mw = cast(self.momx, 'double[:self.N_allocated]')
        self.momy_mw = cast(self.momy, 'double[:self.N_allocated]')
        self.momz_mw = cast(self.momz, 'double[:self.N_allocated]')

    # Method for integrating particle positions forward in time
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   Δt='double',
                   # Locals
                   fac='double',
                   posx='double*',
                   posy='double*',
                   posz='double*',
                   momx='double*',
                   momy='double*',
                   momz='double*',
                   i='size_t',
                   )
    def drift(self, Δt):
        """Note that the time step size Δt is really ∫_t^(t + Δt) dt/a**2.
        """
        # Extracting variables
        posx = self.posx
        posy = self.posy
        posz = self.posz
        momx = self.momx
        momy = self.momy
        momz = self.momz
        # The factor 1/mass*∫_t^(t + Δt) dt/a**2
        fac = Δt/self.mass
        # Update positions
        for i in range(self.N_local):
            posx[i] += momx[i]*fac
            posy[i] += momy[i]*fac
            posz[i] += momz[i]*fac
            # Toroidal boundaries
            posx[i] %= boxsize
            posy[i] %= boxsize
            posz[i] %= boxsize

    # Method for updating particle momenta
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   Δt='double',
                   )
    def kick(self, Δt):
        """Note that the time step size Δt is really ∫_t^(t + Δt) dt/a.
        """
        # Delegate the work to the appropriate function based on species
        if self.species == 'dark matter':
            PP(self, Δt)  # or PM(self, Δt)
        elif self.species == 'dark energy':
            # NOT YET IMPLEMENTED
            pass
        else:
            raise ValueError('Species "' + self.species
                             + '" do not have an assigned kick function!')

    # This method is automaticlly called when a Particles instance
    # is garbage collected. All manually allocated mmeory is freed.
    def __dealloc__(self):
        if self.posx:
            free(self.posx)
        if self.posy:
            free(self.posy)
        if self.posz:
            free(self.posz)
        if self.momx:
            free(self.momx)
        if self.momy:
            free(self.momy)
        if self.momz:
            free(self.momz)


# Constructor function for Particles instances
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               type_name='str',
               species_name='str',
               mass='double',
               N='size_t',
               # Locals
               particles='Particles',
               )
@cython.returns('Particles')
def construct(type_name, species_name, mass, N):
    # Instantiate Particles instance
    particles = Particles(N)
    # Attach information to the particles
    particles.mass = mass
    particles.species = species_name
    particles.type = type_name
    if species_name in softeningfactors:
        particles.softening = (softeningfactors[species_name]
                               *boxsize/(N**one_third))
    else:
        raise ValueError('Species "' + species_name
                         + '" do not have an assigned softening length!')
    return particles


# Function that constructs a Particles instance with random
# positions, momenta and masses. The particle data is
# scattered fair among the processes.
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               type_name='str',
               species_name='str',
               N='size_t',
               # Locals
               N_local='size_t',
               N_locals='tuple',
               mass='double',
               mom_max='double',
               particles='Particles',
               )
@cython.returns('Particles')
def construct_random(type_name, species_name, N):
    # Print out message
    if master:
        print('Initializes particles of type "' + type_name + '"')
    # Minimum and maximum mass and maximum
    # momenta (in any of the three directions)
    mass = Ωm*ϱ*boxsize**3/N
    mom_max = 1e+3*units.km/units.s*mass
    # Compute a fair distribution of particle data to the processes
    N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                + (N//nprocs + 1, )*(N % nprocs))
    N_local = N_locals[rank]
    # Construct a Particles instance
    particles = construct(type_name,
                          species_name,
                          mass=mass,
                          N=N,
                          )
    # Populate the Particles instance with random data
    particles.populate(zeros(N_local), 'acc')
    particles.populate(random(N_local)*boxsize, 'posx')
    particles.populate(random(N_local)*boxsize, 'posy')
    particles.populate(random(N_local)*boxsize, 'posz')
    particles.populate(zeros(N_local, dtype='int32'), 'rung')
    particles.populate((2*random(N_local) - 1)*mom_max, 'momx')
    particles.populate((2*random(N_local) - 1)*mom_max, 'momy')
    particles.populate((2*random(N_local) - 1)*mom_max, 'momz')
    return particles
