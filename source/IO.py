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

from time import sleep

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
               start_local='size_t',
               end_local='size_t',
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
        velx_h5 = particles_h5.create_dataset('velx', [N], dtype='float64')
        vely_h5 = particles_h5.create_dataset('vely', [N], dtype='float64')
        velz_h5 = particles_h5.create_dataset('velz', [N], dtype='float64')
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
        posx_h5[start_local:end_local] = particles.posx_mw
        posy_h5[start_local:end_local] = particles.posy_mw
        posz_h5[start_local:end_local] = particles.posz_mw
        velx_h5[start_local:end_local] = particles.velx_mw
        vely_h5[start_local:end_local] = particles.vely_mw
        velz_h5[start_local:end_local] = particles.velz_mw
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
               N_locals='tuple',
               N_local='size_t',
               start_local='size_t',
               end_local='size_t',
               particles='Particles',
               nr_domain_cuts='int',
               domain_size='double',
               domain_layout='int[:, :, ::1]',
               indices_send='size_t[:, ::1]',
               N_send='size_t[::1]',
               N_send_max='size_t',
               posx='double*',
               posy='double*',
               posz='double*',
               i='size_t',
               posx_domain='int',
               posy_domain='int',
               posz_domain='int',
               owner='int',
               sendbuf='double[::1]',
               N_recv='size_t[::1]',
               N_recv_max='size_t',
               free_slots='ptrdiff_t',
               N_recv_tot='ptrdiff_t',
               N_recv_cum='size_t',
               j='int',
               ID_send='int',
               ID_recv='int',
               N_send_j='size_t',
               indices_send_j='size_t[::1]',
               N_recv_j='size_t',
               index_send='size_t',
               N_send_tot='size_t',
               velx='double*',
               vely='double*',
               velz='double*',
               posx_mw='double[::1]',
               posy_mw='double[::1]',
               posz_mw='double[::1]',
               velx_mw='double[::1]',
               vely_mw='double[::1]',
               velz_mw='double[::1]',
               index_recv_j='size_t',
               N_allocated='size_t',
               N_needed='size_t',
               indices_holds='size_t[::1]',
               indices_holds_count='size_t',
               flag_hold='double',
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
            velx_h5 = particles_h5['velx']
            vely_h5 = particles_h5['vely']
            velz_h5 = particles_h5['velz']
            # Compute a fair distribution of particle data to the processes
            N = posx_h5.size
            N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                        + (N//nprocs + 1, )*(N % nprocs))
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
            particles.populate(velx_h5[start_local:end_local], 'velx')
            particles.populate(vely_h5[start_local:end_local], 'vely')
            particles.populate(velz_h5[start_local:end_local], 'velz')
    # Scatter particles to the correct domain-specific process
    exchange_all(particles)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # THE PARTICLES VARIABLE SHOULD BE A TUPLE OF PARTICLES OR SOMETHING
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return particles

