# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from species import construct
    from communication import exchange_all
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from species cimport construct
    from communication cimport exchange_all
    """


# Function that saves particle data to an hdf5 file
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               particles='Particles',
               filename='str',
               # Locals
               N='size_t',
               N_local='size_t',
               N_locals='size_t[::1]',
               end_local='size_t',
               start_local='size_t',
               )
@cython.returns('Particles')
def save(particles, filename):
    # Print out message
    if master:
        print('Saving snapshot:', filename)
    with h5py.File(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
        # Create HDF5 group and datasets
        N = particles.N
        particles_h5 = hdf5_file.create_group('particles/' + particles.type)
        posx_h5 = particles_h5.create_dataset('posx', [N], dtype='float64')
        posy_h5 = particles_h5.create_dataset('posy', [N], dtype='float64')
        posz_h5 = particles_h5.create_dataset('posz', [N], dtype='float64')
        momx_h5 = particles_h5.create_dataset('momx', [N], dtype='float64')
        momy_h5 = particles_h5.create_dataset('momy', [N], dtype='float64')
        momz_h5 = particles_h5.create_dataset('momz', [N], dtype='float64')
        # Get local indices of the particle data
        N_local = particles.N_local
        N_locals = empty(nprocs, dtype='uintp')
        Allgather(array(N_local, dtype='uintp'), N_locals)
        start_local = sum(N_locals[:rank])
        end_local = start_local + N_local
        # In pure Python, the indices needs to be Python integers
        if not cython.compiled:
            start_local = int(start_local)
            end_local = int(end_local)
        # Save the local slices of the particle data and the attributes
        posx_h5[start_local:end_local] = particles.posx_mw[:N_local]
        posy_h5[start_local:end_local] = particles.posy_mw[:N_local]
        posz_h5[start_local:end_local] = particles.posz_mw[:N_local]
        momx_h5[start_local:end_local] = particles.momx_mw[:N_local]
        momy_h5[start_local:end_local] = particles.momy_mw[:N_local]
        momz_h5[start_local:end_local] = particles.momz_mw[:N_local]
        particles_h5.attrs['type'] = particles.type
        particles_h5.attrs['species'] = particles.species
        particles_h5.attrs['mass'] = particles.mass

# Function that loads particle data from an hdf5 file and instantiate a
# Particles instance on each process, storing the particles within its domain.
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               filename='str',
               # Locals
               N='size_t',
               N_local='size_t',
               N_locals='tuple',
               end_local='size_t',
               particle_type='str',
               particles='Particles',
               start_local='size_t',
               )
@cython.returns('Particles')
def load(filename):
    # Print out message
    if master:
        print('Loading snapshot:', filename)
    # Load all particles
    with h5py.File(filename, mode='r', driver='mpio', comm=comm) as hdf5_file:
        all_particles = hdf5_file['particles']
        for particle_type in all_particles:
            # Extract HDF5 group and datasets
            particles_h5 = all_particles[particle_type]
            posx_h5 = particles_h5['posx']
            posy_h5 = particles_h5['posy']
            posz_h5 = particles_h5['posz']
            momx_h5 = particles_h5['momx']
            momy_h5 = particles_h5['momy']
            momz_h5 = particles_h5['momz']
            # Compute a fair distribution of particle data to the processes
            N = posx_h5.size
            N_locals = ((N//nprocs, )*(nprocs - (N%nprocs))
                        + (N//nprocs + 1, )*(N%nprocs))
            N_local = N_locals[rank]
            start_local = sum(N_locals[:rank])
            end_local = start_local + N_local
            # In pure Python, the indices must be Python integers
            if not cython.compiled:
                start_local = int(start_local)
                end_local = int(end_local)
            # Construct a Particles instance
            particles = construct(particles_h5.attrs['type'],
                                  particles_h5.attrs['species'],
                                  mass=particles_h5.attrs['mass'],
                                  N=N,
                                  )
            # Populate the Particles instance with data from the file
            particles.populate(posx_h5[start_local:end_local], 'posx')
            particles.populate(posy_h5[start_local:end_local], 'posy')
            particles.populate(posz_h5[start_local:end_local], 'posz')
            particles.populate(momx_h5[start_local:end_local], 'momx')
            particles.populate(momy_h5[start_local:end_local], 'momy')
            particles.populate(momz_h5[start_local:end_local], 'momz')
    # Scatter particles to the correct domain-specific process
    exchange_all(particles)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # THE PARTICLES VARIABLE SHOULD BE A TUPLE OF PARTICLES OR SOMETHING
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return particles

