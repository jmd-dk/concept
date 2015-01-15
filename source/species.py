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

    ########################################
    # NOTE! Allocation should happen here. Later, the data can be set to whatever needed. Free is important! Maybe alloc with C's alloc, not Pythons! See http://docs.cython.org/src/tutorial/memory_allocation.html
    ######################################## 

    # Initialization method. Note that data attributes are declared in the .pxd file.
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   pos='double[:, ::1]',
                   vel='double[:, ::1]',
                   mass='double',
                   N='size_t',
                   )
    def __init__(self, pos, vel, mass, N):
        """As always, the last dimension (of pos, vel) are chosen as the
        contiguous one (the C standard). For fast access to a specific
        coordinate of all particles, the particles are distributed over
        the last dimension, while the coordinates are distributed over
        the first. That is, posx = pos[0, :] and so on.
        """
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.N = N
        self.N_local = pos.shape[1]
        self.posx = cython.address(pos[0, :])
        self.posy = cython.address(pos[1, :])
        self.posz = cython.address(pos[2, :])
        self.velx = cython.address(vel[0, :])
        self.vely = cython.address(vel[1, :])
        self.velz = cython.address(vel[2, :])
        # Initialize and assign a meaningless type and species label
        self.type = 'generic particle type'
        self.species = 'generic particle species'

    # Method for integrating particle positions forward in time
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(posx='double*',
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
        # Dispatches the work to the appropriate function based on species
        if self.species == 'dark matter':
            PP(self)  # or PM(self)
        elif self.species == 'dark energy':
            # NOT YET IMPLEMENTED
            pass
        else:
            raise ValueError('Species "' + self.species + '" do not have an assigned kick method!')

# Constructor function for Particles instances
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               type_name='str',
               species_name='str',
               pos='double[:, ::1]',
               vel='double[:, ::1]',
               mass='double',
               N='size_t',
               # Locals
               particles='Particles',
               )
@cython.returns('Particles')
def construct(type_name, species_name, pos, vel, mass, N):
    particles = Particles(pos, vel, mass, N)
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
               vmax='double',
               distribution_sizes='tuple',
               N_local='size_t',
               particles='Particles',
               )
@cython.returns('Particles')
def construct_random(type_name, species_name, N):
    # Minimum and maximum mass and maximum velocity
    mmin = 0.1
    mmax = 1
    vmax = 1
    # Compute a fair distribution of particle data to the processes
    distribution_sizes = ((N//nprocs, )*(nprocs - (N % nprocs))
                          + (N//nprocs + 1, )*(N % nprocs))
    N_local = distribution_sizes[rank]
    # Construct a Particles instance
    particles = construct(type_name,
                          species_name,
                          pos=random((3, N_local))*boxsize,
                          vel=(2*random((3, N_local)) - 1)*vmax,
                          mass=(mmin + random()*(mmax - mmin)),
                          N=N,
                          )
    return particles
