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
    with h5py.File(filename, mode='r', driver='mpio', comm=comm) as hdf5_file:
        # Load all particles
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

    #################################################################################################################################
    # SEPERER NEDENSTÅENDE UD I ET MODUL FOR SIG (COMMUNICATION ELLER NOGET?). DOMAIN LAYOUT-BEREGNINGEN BØR NOK SKE VED IMPORT-TID.
    ################################################################################################################################
    # No need to consider exchange of particles when running on one process 
    if nprocs == 1:
        return particles
    # Compute domain layout
    nr_domain_cuts = int(round(nprocs**(1.0/3.0)))
    domain_size = boxsize/nr_domain_cuts
    domain_layout = arange(nprocs, dtype='int32').reshape([nr_domain_cuts]*3)
    # Particle indices to send to each process
    N_local = particles.N_local
    indices_send = empty((nprocs, N_local), dtype='uintp')  # Overkill to make room for N_local particles to be send!
    # The number of particles to send to each process, and the max of these
    N_send = zeros(nprocs, dtype='uintp')
    N_send_max = 0
    N_send_tot = 0
    # Find out where to send which particle
    posx = particles.posx
    posy = particles.posy
    posz = particles.posz
    for i in range(N_local):
        posx_domain = int(posx[i]//domain_size)
        posy_domain = int(posy[i]//domain_size)
        posz_domain = int(posz[i]//domain_size)
        owner = domain_layout[posx_domain, posy_domain, posz_domain]
        indices_send[owner, N_send[owner]] = i
        # Update N_send, N_send_tot and N_send_max if particle should be sent
        if owner != rank:
            N_send[owner] += 1
            N_send_tot += 1
            if N_send[owner] > N_send_max:
                N_send_max = N_send[owner]
    # Print out message
    N_send_tot_global = reduce(N_send_tot, op=MPI.SUM)
    if master:
        print('Exchanging', N_send_tot_global, 'particles')
    # Allocate send buffer to its maximum needed size
    sendbuf = empty(N_send_max)
    # Find out how many particles will be recieved
    N_recv = empty(nprocs, dtype='uintp')
    N_recv_max = 0
    N_recv_tot = 0
    for j in range(1, nprocs):
        # Process ranks to send/recieve to/from
        ID_send = (rank + j) % nprocs
        ID_recv = (rank - j) % nprocs
        # Send and recieve nr of particles to be exchanged
        N_recv[ID_recv] = sendrecv(N_send[ID_send], dest=ID_send, source=ID_recv)
        N_recv_tot += N_recv[ID_recv]
        if N_recv[ID_recv] > N_recv_max:
            N_recv_max = N_recv[ID_recv]
    # Enlarge the Particles data attributes, if needed
    N_allocated = particles.N_allocated
    N_needed = N_local + N_recv_tot
    if N_allocated < N_needed:
        particles.resize(N_needed)
        N_allocated = N_needed
        posx = particles.posx
        posy = particles.posy
        posz = particles.posz
    # Also extract velocites and memory views
    velx = particles.velx
    vely = particles.vely
    velz = particles.velz
    posx_mw = particles.posx_mw
    posy_mw = particles.posy_mw
    posz_mw = particles.posz_mw
    velx_mw = particles.velx_mw
    vely_mw = particles.vely_mw
    velz_mw = particles.velz_mw
    # Exchange particles between processes
    flag_hold = -1
    N_recv_cum = 0
    indices_holds = empty(N_send_tot, dtype='uintp')
    indices_holds_count = 0
    for j in range(1, nprocs):
        # Process ranks to send/recieve to/from
        ID_send = (rank + j) % nprocs
        ID_recv = (rank - j) % nprocs
        # Extract number of particles to send/recieve and their (send-)indices
        N_send_j = N_send[ID_send]
        N_recv_j = N_recv[ID_recv]
        indices_send_j = indices_send[ID_send, :N_send_j]
        # Fill send buffer and send/recieve posx
        index_recv_j = N_local + N_recv_cum
        for i in range(N_send_j):
            index_send = indices_send_j[i]
            sendbuf[i] = posx[index_send]
            # Flag missing particles (holds) by setting posx equal to flag_hold
            posx[index_send] = flag_hold
            # Save index of the now missing particle (hold)
            indices_holds[indices_holds_count] = index_send
            indices_holds_count += 1
        Sendrecv(sendbuf[:N_send_j], dest=ID_send, recvbuf=posx_mw[index_recv_j:], source=ID_recv)
        # Fill send buffer and send/recieve posy
        for i in range(N_send_j):
            sendbuf[i] = posy[indices_send_j[i]]
        Sendrecv(sendbuf[:N_send_j], dest=ID_send, recvbuf=posy_mw[index_recv_j:], source=ID_recv)
        # Fill send buffer and send/recieve posz
        for i in range(N_send_j):
            sendbuf[i] = posz[indices_send_j[i]]
        Sendrecv(sendbuf[:N_send_j], dest=ID_send, recvbuf=posz_mw[index_recv_j:], source=ID_recv)
        # Fill send buffer and send/recieve velx
        for i in range(N_send_j):
            sendbuf[i] = velx[indices_send_j[i]]
        Sendrecv(sendbuf[:N_send_j], dest=ID_send, recvbuf=velx_mw[index_recv_j:], source=ID_recv)
        # Fill send buffer and send/recieve vely
        for i in range(N_send_j):
            sendbuf[i] = vely[indices_send_j[i]]
        Sendrecv(sendbuf[:N_send_j], dest=ID_send, recvbuf=vely_mw[index_recv_j:], source=ID_recv)
        # Fill send buffer and send/recieve velz
        for i in range(N_send_j):
            sendbuf[i] = velz[indices_send_j[i]]
        Sendrecv(sendbuf[:N_send_j], dest=ID_send, recvbuf=velz_mw[index_recv_j:], source=ID_recv)
        # Update the cummulative counter
        N_recv_cum += N_recv_j
    # Update N_local
    N_local = N_needed - N_send_tot
    particles.N_local = N_local
    # Rearrange particle data elements furthest towards the end to fill holds
    index_move = N_allocated - 1
    for h in range(N_send_tot):
        index_hold = indices_holds[h]
        if index_hold < N_local:
            # Hold which should be filled is located
            for index_move in range(index_move, -1, -1):
                if posx[index_move] != flag_hold:
                    # The particle furthest from index 0 is located
                    break
            # Move particle data
            posx[index_hold] = posx[index_move]
            posy[index_hold] = posy[index_move]
            posz[index_hold] = posz[index_move]
            velx[index_hold] = velx[index_move]
            vely[index_hold] = vely[index_move]
            velz[index_hold] = velz[index_move]
            # Update index of particle to move
            index_move -= 1
            # Break out if all remaining holes lie outside of the
            # region containing actual particles
            if index_move < N_local:
                break


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # THE PARTICLES VARIABLE SHOULD BE A TUPLE OF PARTICLES OR SOMETHING
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return particles

