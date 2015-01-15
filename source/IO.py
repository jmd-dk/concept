# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from species import construct
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from species cimport construct
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
               start_local='size_t',
               end_local='size_t',
               )
@cython.returns('Particles')
def save(particles, filename):
    with h5py.File(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
        # Create HDF5 groups and datasets
        N = particles.N
        particles_group = hdf5_file.create_group('particles/' + particles.type)
        pos = particles_group.create_dataset('pos', (3, N), dtype='float64')
        vel = particles_group.create_dataset('vel', (3, N), dtype='float64')
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
        pos[:, start_local:end_local] = particles.pos
        vel[:, start_local:end_local] = particles.vel
        particles_group.attrs['type'] = particles.type
        particles_group.attrs['species'] = particles.species
        particles_group.attrs['mass'] = particles.mass


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
               owner='int',
               sendbuf='double[::1]',
               N_recv_max='size_t',
               recvbuf='double[::1]',
               j='int',
               ID_send='int',
               ID_recv='int',
               N_send_j='size_t',
               indices_send_j='size_t[::1]',
               k='size_t',
               N_recv_j='size_t',
               )
@cython.returns('Particles')
def load(filename):
    with h5py.File(filename, mode='r', driver='mpio', comm=comm) as hdf5_file:
        # Load all particles
        all_particles = hdf5_file['particles']
        for particle_type in all_particles:
            particles_group = all_particles[particle_type]
            pos = particles_group['pos']
            vel = particles_group['vel']
            # Compute a fair distribution of particle data to the processes
            N = pos.shape[1]
            N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                        + (N//nprocs + 1, )*(N % nprocs))
            N_local = local_size = N_locals[rank]
            start_local = sum(N_locals[:rank])
            end_local = start_local + N_local
            # In pure Python, the indices needs to be Python integers
            if not cython.compiled:
                start_local = int(start_local)
                end_local = int(end_local)
            # Construct a Particles instance
            particles = construct(particles_group.attrs['type'],
                                  particles_group.attrs['species'],
                                  pos=pos[:, start_local:end_local],
                                  vel=vel[:, start_local:end_local],
                                  mass=particles_group.attrs['mass'],
                                  N=N,
                                  )
    # Compute domain layout
    nr_domain_cuts = int(round(nprocs**(1.0/3.0)))
    domain_size = boxsize/nr_domain_cuts
    domain_layout = arange(nprocs, dtype='int32').reshape([nr_domain_cuts]*3)
    # Particle indices to send to each process
    indices_send = empty((nprocs, particles.N), dtype='uintp')
    # The number of particles to send to each process, and the max of these
    N_send = zeros(nprocs, dtype='uintp')
    N_send_max = 0
    # Find out where to send which particle
    posx = particles.posx
    posy = particles.posy
    posz = particles.posz
    for i in range(particles.N_local):
        owner = domain_layout[int(posx[i]//domain_size),
                              int(posy[i]//domain_size),
                              int(posz[i]//domain_size),
                              ]
        indices_send[owner, N_send[owner]] = i
        N_send[owner] += 1
        if N_send[owner] > N_send_max:
            N_send_max = N_send[owner]
    # Allocate buffers to their maximum needed sizes
    sendbuf = empty(N_send_max)
    N_recv_max = allreduce(N_send_max, op=MPI.MAX)
    recvbuf = empty(N_recv_max)
    # Exchange particles between processes
    for j in range(1, nprocs):
        # Process ranks to send/recieve to/from
        ID_send = (rank + j) % nprocs
        ID_recv = (rank - j) % nprocs
        N_send_j = N_send[ID_send]
        indices_send_j = indices_send[ID_send, :]
        # Fill send buffer
        for k in range(N_send_j):
            sendbuf[k] = posx[indices_send_j[k]]
        # First send/recieve the size of the exchanged data
        N_recv_j = sendrecv(N_send_j, dest=ID_send, source=ID_recv)
        # Now send and recieve particle data
        Sendrecv(sendbuf[:N_send_j], dest=ID_send, recvbuf=recvbuf, source=ID_recv)
        #recvbuf[:N_recv_j]



    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # THE PARTICLES VARIABLE SHOULD BE A TUPLE OF PARTICLES OR SOMETHING
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return particles

