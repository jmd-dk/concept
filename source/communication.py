# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    pass
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    """

# This function examines every particle and communicates them to the
# process governing the domain in which the particle is located
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               # Locals
               ID_recv='int',
               ID_send='int',
               N_allocated='size_t',
               N_local='size_t',
               N_needed='size_t',
               N_recv='size_t[::1]',
               N_recv_cum='size_t',
               N_recv_j='size_t',
               N_recv_max='size_t',
               N_recv_tot='size_t',  # Used to be ptrdiff_t
               N_send='size_t[::1]',
               N_send_j='size_t',
               N_send_max='size_t',
               N_send_tot='size_t',
               flag_hold='double',
               i='size_t',
               index_hold='size_t',
               index_move='ptrdiff_t',
               index_recv_j='size_t',
               index_send='size_t',
               indices_holds='size_t[::1]',
               indices_holds_count='size_t',
               indices_send='size_t[:, ::1]',
               indices_send_j='size_t[::1]',
               j='int',
               owner='int',
               posx='double*',
               posx_domain='int',
               posx_mw='double[::1]',
               posy='double*',
               posy_domain='int',
               posy_mw='double[::1]',
               posz='double*',
               posz_domain='int',
               posz_mw='double[::1]',
               sendbuf='double[::1]',
               velx='double*',
               velx_mw='double[::1]',
               vely='double*',
               vely_mw='double[::1]',
               velz='double*',
               velz_mw='double[::1]',
               )
def exchange_all(particles):
    # No need to consider exchange of particles if running serial
    if nprocs == 1:
        return
    # Initialize some variables
    flag_hold = -1
    N_send = zeros(nprocs, dtype='uintp')
    N_recv = empty(nprocs, dtype='uintp')
    if not cython.compiled:
        N_recv = asarray(N_recv, dtype='int32')
    N_send_max = 0
    N_recv_max = 0
    N_send_tot = 0
    N_recv_tot = 0
    N_recv_cum = 0
    N_local = particles.N_local
    N_allocated = particles.N_allocated
    indices_send = empty((nprocs, N_local), dtype='uintp')  # Overkill to make room for N_local particles to be send!
    indices_holds_count = 0
    posx = particles.posx
    posy = particles.posy
    posz = particles.posz
    # Find out where to send which particle
    for i in range(N_local):
        posx_domain = int(posx[i]//domain_size_x)
        posy_domain = int(posy[i]//domain_size_y)
        posz_domain = int(posz[i]//domain_size_z)
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
    N_needed = N_local + N_recv_tot
    if N_allocated < N_needed:
        particles.resize(N_needed)
        N_allocated = N_needed
        # Reextract position pointers
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
    indices_holds = empty(N_send_tot, dtype='uintp')
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
    for i in range(N_send_tot):
        index_hold = indices_holds[i]
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


# Function for prime checking. Used by the cutout_domains function.
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               n='int',
               # Locals
               f='int',
               len_primeset='int',
               r='int',
               )
@cython.returns('list')
def isprime(n):
    if n == 2 or n == 3:
        return True
    if n < 2 or (n%2) == 0:
        return False
    if n < 9:
        return True
    if (n%3) == 0:
        return False
    r = int(sqrt(n))
    f = 5
    while f <= r:
        if (n%f) == 0 or (n%(f + 2)) == 0:
            return False
        f += 6
    return True


# Function for cutting out domains as rectangular boxes in the best possible
# way. When all dimensions cannot be divided equally, the x-dimension is
# subdivided the most, then the y-dimension and lastly the z-dimension.
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               n='int',
               basecall='bint',
               # Locals
               N_primes='int',
               i='int',
               primeset='list',
               )
@cython.returns('list')
def cutout_domains(n, basecall=True):
    primeset  = []
    while n > 1:
        for i in range(2, int(n + 1)):
            if (n%i) == 0:
                if isprime(i):
                    primeset.append(i)
                else:
                    primeset += cutout_domains(i, basecall=False)
                n /= i
    # The returned list should always consist of 3 values
    if basecall:
        N_primes = len(primeset)
        if N_primes < 4:
            return sorted(primeset + [1]*(3 - N_primes), reverse=True)
        else:
            len_primeset = len(primeset)
            while len_primeset > 3:
                primeset = sorted(primeset, reverse=True)
                primeset[len_primeset - 2] *= primeset[len_primeset - 1]
                primeset.pop()
                len_primeset -= 1
            return sorted(primeset, reverse=True)
    return primeset


# Cutout domains at import time
cython.declare(domain_cuts='list',
               domain_layout='int[:, :, ::1]',
               domain_size_x='double',
               domain_size_y='double',
               domain_size_z='double',
               )
domain_cuts = cutout_domains(nprocs)
domain_size_x = boxsize/domain_cuts[0]
domain_size_y = boxsize/domain_cuts[1]
domain_size_z = boxsize/domain_cuts[2]
domain_layout = arange(nprocs, dtype='int32').reshape(domain_cuts)
