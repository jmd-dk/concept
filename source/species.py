# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from gravity import PP, PM
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from gravity cimport PP, PM
    """


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

    # Initialization method. Note that data attributes are declared in the .pxd file.
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   mass='double',
                   N='size_t',
                   )
    def __init__(self, mass, N):
        # Instantiate Particles instances with just a single particle member
        self.N_allocated = 1
        # Manually allocate memory for particle data
        self.posx = malloc(self.N_allocated*sizeof('double'))
        self.posy = malloc(self.N_allocated*sizeof('double'))
        self.posz = malloc(self.N_allocated*sizeof('double'))
        self.velx = malloc(self.N_allocated*sizeof('double'))
        self.vely = malloc(self.N_allocated*sizeof('double'))
        self.velz = malloc(self.N_allocated*sizeof('double'))
        # Memory views around the allocated data
        self.posx_mw = cast(self.posx, 'double[:self.N_allocated]')
        self.posy_mw = cast(self.posy, 'double[:self.N_allocated]')
        self.posz_mw = cast(self.posz, 'double[:self.N_allocated]')
        self.velx_mw = cast(self.velx, 'double[:self.N_allocated]')
        self.vely_mw = cast(self.vely, 'double[:self.N_allocated]')
        self.velz_mw = cast(self.velz, 'double[:self.N_allocated]')
        # Store particle meta data
        self.mass = mass
        self.N = N
        self.N_local = 1
        self.type = 'generic particle type'
        self.species = 'generic particle species'

    # This method populate the Particles pos/vel attributes with data.
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
        elif coord == 'velx':
            self.velx = realloc(self.velx, self.N_allocated*sizeof('double'))
            self.velx_mw = cast(self.velx, 'double[:self.N_local]')
            self.velx_mw[...] = mw[...]
        elif coord == 'vely':
            self.vely = realloc(self.vely, self.N_allocated*sizeof('double'))
            self.vely_mw = cast(self.vely, 'double[:self.N_local]')
            self.vely_mw[...] = mw[...]
        elif coord == 'velz':
            self.velz = realloc(self.velz, self.N_allocated*sizeof('double'))
            self.velz_mw = cast(self.velz, 'double[:self.N_local]')
            self.velz_mw[...] = mw[...]
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
        self.velx = realloc(self.velx, self.N_allocated*sizeof('double'))
        self.vely = realloc(self.vely, self.N_allocated*sizeof('double'))
        self.velz = realloc(self.velz, self.N_allocated*sizeof('double'))
        # Reassign memory views
        self.posx_mw = cast(self.posx, 'double[:self.N_allocated]')
        self.posy_mw = cast(self.posy, 'double[:self.N_allocated]')
        self.posz_mw = cast(self.posz, 'double[:self.N_allocated]')
        self.velx_mw = cast(self.velx, 'double[:self.N_allocated]')
        self.vely_mw = cast(self.vely, 'double[:self.N_allocated]')
        self.velz_mw = cast(self.velz, 'double[:self.N_allocated]')

    # Method for integrating particle positions forward in time
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Locals
                   posx='double*',
                   posy='double*',
                   posz='double*',
                   velx='double*',
                   vely='double*',
                   velz='double*',
                   i='size_t',
                   )
    def drift(self):
        # Extracting variables
        posx = self.posx
        posy = self.posy
        posz = self.posz
        velx = self.velx
        vely = self.vely
        velz = self.velz
        # Update positions
        for i in range(self.N_local):
            posx[i] += velx[i]*dt
            posy[i] += vely[i]*dt
            posz[i] += velz[i]*dt
            # Toroidal boundaries
            posx[i] %= boxsize
            posy[i] %= boxsize
            posz[i] %= boxsize

    # Method for updating particle velocities
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def kick(self):
        # Delegate the work to the appropriate function based on species
        if self.species == 'dark matter':
            PP(self)  # or PM(self)
        elif self.species == 'dark energy':
            # NOT YET IMPLEMENTED
            pass
        else:
            raise ValueError('Species "' + self.species + '" do not have an assigned kick method!')

    # This method is automaticlly called when a Particles instance
    # is garbage collected. All manually allocated mmeory is freed.
    def __dealloc__(self):
        if self.posx:
            free(self.posx)
        if self.posy:
            free(self.posy)
        if self.posz:
            free(self.posz)
        if self.velx:
            free(self.velx)
        if self.vely:
            free(self.vely)
        if self.velz:
            free(self.velz)


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
    particles = Particles(mass, N)
    # Attach the type and species information to the particles
    particles.type = type_name
    particles.species = species_name
    return particles

# Function that constructs a Particles instance with random
# positions, velocities and masses. The particle data is
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
               mmin='double',
               mmax='double',
               particles='Particles',
               vmax='double',
               )
@cython.returns('Particles')
def construct_random(type_name, species_name, N):
    # Print out message
    if master:
        print('Initializes particles of type "' + type_name + '"')
    # Minimum and maximum mass and maximum velocity
    mmin = 0.1
    mmax = 1
    vmax = 2
    # Compute a fair distribution of particle data to the processes
    N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                + (N//nprocs + 1, )*(N % nprocs))
    N_local = N_locals[rank]
    # Construct a Particles instance
    particles = construct(type_name,
                          species_name,
                          mass=(mmin + random()*(mmax - mmin)),
                          N=N,
                          )
    # Populate the Particles instance with random data
    particles.populate(random(N_local)*boxsize, 'posx')
    particles.populate(random(N_local)*boxsize, 'posy')
    particles.populate(random(N_local)*boxsize, 'posz')
    particles.populate((2*random(N_local) - 1)*vmax, 'velx')
    particles.populate((2*random(N_local) - 1)*vmax, 'vely')
    particles.populate((2*random(N_local) - 1)*vmax, 'velz')
    return particles
