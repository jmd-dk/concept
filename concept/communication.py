# Copyright (C) 2015 Jeppe Mosgard Dakin
#
# This file is part of CONCEPT, the cosmological N-body code in Python
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


# Import everything from the commons module. In the .pyx file,
# this line willbe replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    pass
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    """


# Function for communicating sizes of receive buffers
@cython.header(# Arguments
               N_send='size_t[::1]',
               # Locals
               ID_recv='int',
               ID_send='int',
               N_recv='size_t[::1]',
               N_recv_max='size_t',
               j='int',
               k='int',
               same_N_send='bint',
               max_bfore_rank='size_t',
               max_after_rank='size_t',
               returns='size_t[::1]',
               )
def find_N_recv(N_send):
    """Given the size of arrays to send, N_send, which itself has a
    length of either 1 (same data size to send to every process) or
    n_procs (individual data sizes to send to the processes), this
    function communicates this to all processes, so that everyone knows
    how much to receive from every process. The entrance number rank is
    unused (the process do not send to itself). The maximum number to
    receive is useful when allocating receive buffers, so this number is
    stored in this otherwize unused entrance.
    """
    N_recv = empty(nprocs, dtype='uintp')
    # Check whether N_send is the same for each process to send to
    same_N_send = (N_send.size == 1)
    # If N_send is the same for each process, an Allgather will suffice
    if same_N_send:
        Allgather(N_send, N_recv)
        # Store the max N_recv_in the unused entrance in N_recv
        max_bfore_rank = 0 if rank == 0 else max(N_recv[:rank])
        max_after_rank = 0 if rank == nprocs - 1 else max(N_recv[(rank + 1):])
        N_recv[rank] = (max_bfore_rank if max_bfore_rank > max_after_rank
                        else max_after_rank)
        return N_recv
    # Find out how many particles will be received from each process
    N_recv_max = 0
    for j in range(1, nprocs):
        # Process ranks to send/receive to/from
        ID_send = mod(rank + j, nprocs)
        ID_recv = mod(rank - j, nprocs)
        # Send and receive nr of particles to be exchanged
        N_recv[ID_recv] = sendrecv(N_send[ID_send],
                                   dest=ID_send, source=ID_recv)
        if N_recv[ID_recv] > N_recv_max:
            N_recv_max = N_recv[ID_recv]
    # Store N_recv_max in the unused entrance in N_recv
    N_recv[rank] = N_recv_max
    return N_recv

# This function examines every particle and communicates them to the
# process governing the domain in which the particle is located
@cython.header(# Arguments
               particles='Particles',
               reset_buffers='bint',
               # Locals
               ID_recv='int',
               ID_send='int',
               N_local='size_t',
               N_needed='size_t',
               N_recv='size_t[::1]',
               N_recv_j='size_t',
               N_recv_max='size_t',
               N_recv_tot='size_t',
               N_send_j='size_t',
               N_send_max='size_t',
               N_send_owner='size_t',
               N_send_tot='size_t',
               N_send_tot_global='size_t',
               holes_filled='int',
               i='size_t',
               index_recv_j='size_t',
               indices_send_j='size_t*',
               indices_send_owner='size_t*',
               j='int',
               k='size_t',
               k_start='size_t',
               momx='double*',
               momx_mv='double[::1]',
               momy='double*',
               momy_mv='double[::1]',
               momz='double*',
               momz_mv='double[::1]',
               owner='int',
               posx='double*',
               posx_mv='double[::1]',
               posy='double*',
               posy_mv='double[::1]',
               posz='double*',
               posz_mv='double[::1]',
               Δmemory='size_t',
               )
def exchange(particles, reset_buffers=False):
    """This function will do an exchange of particles between processes,
    so that every particle resides on the process in charge of the
    domian where the particle is located. The variable indices_send
    holds arrays of indices of particles to send to the different
    processes, while particle data is copied to sendbuf before it is
    send. These two variables will grow in size if needed. Call with
    reset_buffers=True to reset these variables to their most basic
    forms, freeing up memory. 
    """
    global N_send, indices_send, indices_send_sizes, sendbuf, sendbuf_mv
    # No need to consider exchange of particles if running serial
    if nprocs == 1:
        return
    # Extract some variables from particles
    N_local = particles.N_local
    posx = particles.posx
    posy = particles.posy
    posz = particles.posz
    # The index buffers indices_send[:] increase in size by this amount
    Δmemory = 2 + cast(0.01*N_local/nprocs, 'size_t')
    # Reset the number of particles to be sent
    for j in range(nprocs):
        N_send[j] = 0
    # Find out where to send which particle
    for i in range(N_local):
        # Rank of the process that local particle i belongs to
        owner = domain(posx[i], posy[i], posz[i])
        if owner != rank:
            # Particle owned by nonlocal process owner.
            # Append the owner's index buffer with the particle index.
            indices_send[owner][N_send[owner]] = i
            # Increase the number of particle to send to this process.            
            N_send[owner] += 1
            # Enlarge the index buffer indices_send[owner] if needed
            if N_send[owner] == indices_send_sizes[owner]:
                indices_send_sizes[owner] += Δmemory
                indices_send[owner] = realloc(indices_send[owner],
                                              indices_send_sizes[owner]
                                              *sizeof('size_t'))
    # No need to continue if no particles should be exchanged
    N_send_tot = sum(N_send)
    N_send_tot_global = allreduce(N_send_tot, op=MPI.SUM)
    if N_send_tot_global == 0:
        return
    # Print out exchange message
    if N_send_tot_global == 1:
        masterprint('Exchanging 1 particle ...')
    elif N_send_tot_global > 1:
        # The int casting is necessary in pure Python
        masterprint('Exchanging', N_send_tot_global, 'particles ...')
    # Enlarge sendbuf, if necessary
    N_send_max = max(N_send)
    if N_send_max > sendbuf_mv.size:
        sendbuf = realloc(sendbuf, N_send_max*sizeof('double'))
        sendbuf_mv = cast(sendbuf, 'double[:N_send_max]')
    # Find out how many particles to receive
    N_recv = find_N_recv(N_send)
    # Pure Python has a hard time understanding uintp as an integer
    if not cython.compiled:
        N_recv = asarray(N_recv, dtype='int64')
    # The maximum number of particles to
    # receive is stored in entrance rank.
    N_recv_max = N_recv[rank]
    N_recv_tot = sum(N_recv) - N_recv_max
    # Enlarge the Particles data attributes, if needed. This may not be
    # strcitly necessary as more particles may be send than received.
    N_needed = N_local + N_recv_tot
    if particles.N_allocated < N_needed:
        particles.resize(N_needed)
        # Reextract position pointers
        posx = particles.posx
        posy = particles.posy
        posz = particles.posz
    # Extract momenta and memory views of the possibly resized data
    momx = particles.momx
    momy = particles.momy
    momz = particles.momz
    posx_mv = particles.posx_mv
    posy_mv = particles.posy_mv
    posz_mv = particles.posz_mv
    momx_mv = particles.momx_mv
    momy_mv = particles.momy_mv
    momz_mv = particles.momz_mv
    # Exchange particles between processes
    index_recv_j = N_local
    for j in range(1, nprocs):
        # Process ranks to send/receive to/from
        ID_send = mod(rank + j, nprocs)
        ID_recv = mod(rank - j, nprocs)
        # Number of particles to send/receive
        N_send_j = N_send[ID_send]
        N_recv_j = N_recv[ID_recv]
        # The indices of particles to send and the index from
        # which received particles are allowed to be appended.
        indices_send_j = indices_send[ID_send]
        # Send/receive posx
        for i in range(N_send_j):
            sendbuf[i] = posx[indices_send_j[i]]
        Sendrecv(sendbuf_mv[:N_send_j],
                 dest=ID_send,
                 recvbuf=posx_mv[index_recv_j:],
                 source=ID_recv)
        # Send/receive posy
        for i in range(N_send_j):
            sendbuf[i] = posy[indices_send_j[i]]
        Sendrecv(sendbuf_mv[:N_send_j],
                 dest=ID_send,
                 recvbuf=posy_mv[index_recv_j:],
                 source=ID_recv)
        # Send/receive posz
        for i in range(N_send_j):
            sendbuf[i] = posz[indices_send_j[i]]
        Sendrecv(sendbuf_mv[:N_send_j],
                 dest=ID_send,
                 recvbuf=posz_mv[index_recv_j:],
                 source=ID_recv)
        # Send/receive momx
        for i in range(N_send_j):
            sendbuf[i] = momx[indices_send_j[i]]
        Sendrecv(sendbuf_mv[:N_send_j],
                 dest=ID_send,
                 recvbuf=momx_mv[index_recv_j:],
                 source=ID_recv)
        # Send/receive momy
        for i in range(N_send_j):
            sendbuf[i] = momy[indices_send_j[i]]
        Sendrecv(sendbuf_mv[:N_send_j],
                 dest=ID_send,
                 recvbuf=momy_mv[index_recv_j:],
                 source=ID_recv)
        # Send/receive momz
        for i in range(N_send_j):
            sendbuf[i] = momz[indices_send_j[i]]
        Sendrecv(sendbuf_mv[:N_send_j],
                 dest=ID_send,
                 recvbuf=momz_mv[index_recv_j:],
                 source=ID_recv)
        # Update the start index for received data
        index_recv_j += N_recv_j
        # Mark the holes in the data by setting posx[hole] = -1
        for k in range(N_send_j):
            posx[indices_send_j[k]] = -1
    # Move particle data to fill holes
    k_start = 0
    holes_filled = 0
    if N_send_tot > 0:
        # Loop backward over particles
        for i in range(index_recv_j - 1, -1, -1):
            # Index i should be a particle
            if posx[i] == -1:
                continue
            # Loop forward over holes
            for k in range(k_start, index_recv_j):
                # Index k should be a hole
                if posx[k] != -1:
                    continue
                # Particle i and hole k found.
                # Fill the hole with the particle
                posx[k] = posx[i]
                posy[k] = posy[i]
                posz[k] = posz[i]
                momx[k] = momx[i]
                momy[k] = momy[i]
                momz[k] = momz[i]
                k_start = k + 1
                holes_filled += 1
                break
            # All holes have been filled
            if holes_filled == N_send_tot:
                break
    # Update N_local
    particles.N_local = N_needed - N_send_tot
    # Pure Python has a hard time understanding uintp as an integer
    if not cython.compiled:
        particles.N_local = asarray(particles.N_local, dtype='int64')
    # If reset_buffers == True, reset the global indices_send and
    # sendbuf to their basic forms. This buffer will then be rebuild in
    # future calls.
    if reset_buffers:
        for j in range(nprocs):
            indices_send[j] = realloc(indices_send[j], 1*sizeof('size_t'))
            indices_send_sizes[j] = 1
            sendbuf = realloc(sendbuf, 1*sizeof('double'))
            sendbuf_mv = cast(sendbuf, 'double[:1]')
    # Finalize exchange message
    masterprint('done')
    

# Function for cutting out domains as rectangular boxes in the best
# possible way. When all dimensions cannot be divided equally, the
# x-dimension is subdivided the most, then the y-dimension and lastly
# the z-dimension.
@cython.header(# Arguments
               n='int',
               basecall='bint',
               # Locals
               N_primes='int',
               f='int',
               i='int',
               r='int',
               len_primeset='int',
               primeset='list',
               returns='list',
               )
def cutout_domains(n, basecall=True):
    """This function works by computing a prime factorization of n
    and then multiplying the smallest factors until 3 remain.
    """
    # Factorize n
    primeset = []
    while n > 1:
        for i in range(2, int(n + 1)):
            if n % i == 0:
                # Check whether i is prime
                if i == 2 or i == 3:
                    i_is_prime = True
                elif i < 2 or i % 2 == 0:
                    i_is_prime = False
                elif i < 9:
                    i_is_prime = True
                elif i % 3 == 0:
                    i_is_prime = False
                else:
                    r = int(sqrt(i))
                    f = 5
                    while f <= r:
                        if i % f == 0 or i % (f + 2) == 0:
                            i_is_prime = False
                            break
                        f += 6
                    else:
                        i_is_prime = True
                # If i is prime it is a prime factor of n. If not,
                # factorize i to get its prime factors
                if i_is_prime:
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

# This function takes coordinates as arguments and returns the rank of
# the process that governs the domain in which the coordinates reside.
@cython.header(# Arguments
               x='double',
               y='double',
               z='double',
               # Locals
               x_index='int',
               y_index='int',
               z_index='int',
               returns='int',
               )
def domain(x, y, z):
    x_index = int(x/domain_size_x)
    y_index = int(y/domain_size_y)
    z_index = int(z/domain_size_z)
    return domain_layout[x_index, y_index, z_index]


# This function computes the ranks of the processes governing the 27
# neighboring domains.
@cython.header(# Locals
               domain_cuts='int[::1]',
               domain_layout='int[:, :, ::1]',
               rank_right='int',
               rank_left='int',
               rank_forward='int',
               rank_backward='int',
               rank_up='int',
               rank_down='int',
               rank_rightforward='int',
               rank_rightbackward='int',
               rank_rightup='int',
               rank_rightdown='int',
               rank_leftforward='int',
               rank_leftbackward='int',
               rank_leftup='int',
               rank_leftdown='int',
               rank_forwardup='int',
               rank_forwarddown='int',
               rank_backwardup='int',
               rank_backwarddown='int',
               rank_rightforwardup='int',
               rank_rightforwarddown='int',
               rank_rightbackwardup='int',
               rank_rightbackwarddown='int',
               rank_leftforwardup='int',
               rank_leftforwarddown='int',
               rank_leftbackwardup='int',
               rank_leftbackwarddown='int',
               returns='dict',
               )
def neighboring_ranks():
    # Number of domains in all three dimensions
    domain_cuts = array(cutout_domains(nprocs), dtype='int32')
    # The 3D layout of the division of the box
    domain_layout = arange(nprocs, dtype='int32').reshape(domain_cuts)
    # Get the ranks of the 6 face-to-face neighboring processes
    rank_right = domain_layout[mod(domain_local[0] + 1, domain_cuts[0]),
                               domain_local[1],
                               domain_local[2]]
    rank_left = domain_layout[mod(domain_local[0] - 1, domain_cuts[0]),
                               domain_local[1],
                               domain_local[2]]
    rank_forward = domain_layout[domain_local[0],
                                 mod(domain_local[1] + 1, domain_cuts[1]),
                                 domain_local[2]]
    rank_backward = domain_layout[domain_local[0],
                                  mod(domain_local[1] - 1, domain_cuts[1]),
                                  domain_local[2]]
    rank_up = domain_layout[domain_local[0],
                            domain_local[1],
                            mod(domain_local[2] + 1, domain_cuts[2])]
    rank_down = domain_layout[domain_local[0],
                              domain_local[1],
                              mod(domain_local[2] - 1, domain_cuts[2])]
    # Get the ranks of the 12 edge-to-edge neighboring processes
    rank_rightforward = domain_layout[mod(domain_local[0] + 1, domain_cuts[0]),
                                      mod(domain_local[1] + 1, domain_cuts[1]),
                                      domain_local[2]]
    rank_rightbackward = domain_layout[mod(domain_local[0] + 1,
                                           domain_cuts[0]),
                                       mod(domain_local[1] - 1,
                                           domain_cuts[1]),
                                       domain_local[2]]
    rank_rightup = domain_layout[mod(domain_local[0] + 1, domain_cuts[0]),
                                 domain_local[1],
                                 mod(domain_local[2] + 1, domain_cuts[2])]
    rank_rightdown = domain_layout[mod(domain_local[0] + 1, domain_cuts[0]),
                                   domain_local[1],
                                   mod(domain_local[2] - 1, domain_cuts[2])]
    rank_leftforward = domain_layout[mod(domain_local[0] - 1, domain_cuts[0]),
                                     mod(domain_local[1] + 1, domain_cuts[1]),
                                     domain_local[2]]
    rank_leftbackward = domain_layout[mod(domain_local[0] - 1, domain_cuts[0]),
                                      mod(domain_local[1] - 1, domain_cuts[1]),
                                      domain_local[2]]
    rank_leftup = domain_layout[mod(domain_local[0] - 1, domain_cuts[0]),
                                domain_local[1],
                                mod(domain_local[2] + 1, domain_cuts[2])]
    rank_leftdown = domain_layout[mod(domain_local[0] - 1, domain_cuts[0]),
                                  domain_local[1],
                                  mod(domain_local[2] - 1, domain_cuts[2])]
    rank_forwardup = domain_layout[domain_local[0],
                                   mod(domain_local[1] + 1, domain_cuts[1]),
                                   mod(domain_local[2] + 1, domain_cuts[2])]
    rank_forwarddown = domain_layout[domain_local[0],
                                     mod(domain_local[1] + 1, domain_cuts[1]),
                                     mod(domain_local[2] - 1, domain_cuts[2])]
    rank_backwardup = domain_layout[domain_local[0],
                                      mod(domain_local[1] - 1, domain_cuts[1]),
                                      mod(domain_local[2] + 1, domain_cuts[2])]
    rank_backwarddown = domain_layout[domain_local[0],
                                      mod(domain_local[1] - 1, domain_cuts[1]),
                                      mod(domain_local[2] - 1, domain_cuts[2])]
    # Get the ranks of the 8 point-to-point neighboring processes
    rank_rightforwardup = domain_layout[mod(domain_local[0] + 1,
                                            domain_cuts[0]),
                                        mod(domain_local[1] + 1,
                                            domain_cuts[1]),
                                        mod(domain_local[2] + 1,
                                            domain_cuts[2])]
    rank_rightforwarddown = domain_layout[mod(domain_local[0] + 1,
                                              domain_cuts[0]),
                                          mod(domain_local[1] + 1,
                                              domain_cuts[1]),
                                          mod(domain_local[2] - 1,
                                              domain_cuts[2])]
    rank_rightbackwardup = domain_layout[mod(domain_local[0] + 1,
                                             domain_cuts[0]),
                                         mod(domain_local[1] - 1,
                                             domain_cuts[1]),
                                         mod(domain_local[2] + 1,
                                             domain_cuts[2])]
    rank_rightbackwarddown = domain_layout[mod(domain_local[0] + 1,
                                               domain_cuts[0]),
                                           mod(domain_local[1] - 1,
                                               domain_cuts[1]),
                                           mod(domain_local[2] - 1,
                                               domain_cuts[2])]
    rank_leftforwardup = domain_layout[mod(domain_local[0] - 1,
                                           domain_cuts[0]),
                                       mod(domain_local[1] + 1,
                                           domain_cuts[1]),
                                       mod(domain_local[2] + 1,
                                           domain_cuts[2])]
    rank_leftforwarddown = domain_layout[mod(domain_local[0] - 1,
                                             domain_cuts[0]),
                                         mod(domain_local[1] + 1,
                                             domain_cuts[1]),
                                         mod(domain_local[2] - 1,
                                             domain_cuts[2])]
    rank_leftbackwardup = domain_layout[mod(domain_local[0] - 1,
                                            domain_cuts[0]),
                                        mod(domain_local[1] - 1,
                                            domain_cuts[1]),
                                        mod(domain_local[2] + 1,
                                            domain_cuts[2])]
    rank_leftbackwarddown = domain_layout[mod(domain_local[0] - 1,
                                              domain_cuts[0]),
                                          mod(domain_local[1] - 1,
                                              domain_cuts[1]),
                                          mod(domain_local[2] - 1,
                                              domain_cuts[2])]
    # Return dict
    return {'right': rank_right,
            'left': rank_left,
            'forward': rank_forward,
            'backward': rank_backward,
            'up': rank_up,
            'down': rank_down,
            'rightforward': rank_rightforward,
            'rightbackward': rank_rightbackward,
            'rightup': rank_rightup,
            'rightdown': rank_rightdown,
            'leftforward': rank_leftforward,
            'leftbackward': rank_leftbackward,
            'leftup': rank_leftup,
            'leftdown': rank_leftdown,
            'forwardup': rank_forwardup,
            'forwarddown': rank_forwarddown,
            'backwardup': rank_backwardup,
            'backwarddown': rank_backwarddown,
            'rightforwardup': rank_rightforwardup,
            'rightforwarddown': rank_rightforwarddown,
            'rightbackwardup': rank_rightbackwardup,
            'rightbackwarddown': rank_rightbackwarddown,
            'leftforwardup': rank_leftforwardup,
            'leftforwarddown': rank_leftforwarddown,
            'leftbackwardup': rank_leftbackwardup,
            'leftbackwarddown': rank_leftbackwarddown,
            }



# Cutout domains at import time
cython.declare(domain_cuts='list',
               domain_layout='int[:, :, ::1]',
               domain_local='int[::1]',
               domain_size_x='double',
               domain_size_y='double',
               domain_size_z='double',
               i='int',
               j='int',
               k='int',
               neighbor_domains='int[::1]',
               x_index='int',
               y_index='int',
               z_index='int',
               )
# Number of domains of the box in all three dimensions
domain_cuts = cutout_domains(nprocs)
# The 3D layout of the division of the box
domain_layout = arange(nprocs, dtype='int32').reshape(domain_cuts)
# The indices in domain_layout of the local domain
domain_local = array(np.unravel_index(rank, domain_cuts), dtype='int32')
# The size of the domain, which are the same for all of them
domain_size_x = boxsize/domain_cuts[0]
domain_size_y = boxsize/domain_cuts[1]
domain_size_z = boxsize/domain_cuts[2]

# Initialize the variables used in the exchange function at import time
cython.declare(N_send='size_t[::1]',
               indices_send='size_t**',
               indices_send_sizes='size_t[::1]',
               sendbuf='double*',
               sendbuf_mv='double[::1]',
               )
# This variable stores the number of particles to send to each prcess
N_send = zeros(nprocs, dtype='uintp')
# This size_t** variable stores the indices of particles to be send to
# other processes. That is, indices_send[other_rank][i] is the local
# index of some particle which should be send to other_rank.
indices_send = malloc(nprocs*sizeof('size_t*'))
for j in range(nprocs):
    indices_send[j] = malloc(1*sizeof('size_t'))
# The size of the allocated indices_send[:] memory
indices_send_sizes = ones(nprocs, dtype='uintp')
# The send buffer for the particle data
sendbuf = malloc(1*sizeof('double'))
sendbuf_mv = cast(sendbuf, 'double[:1]')
