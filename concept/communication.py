# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2017 Jeppe Mosgaard Dakin.
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
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *



# Function for fairly partitioning data among the processes
@cython.header(# Arguments
               size='Py_ssize_t',
               # Locals
               size_local='Py_ssize_t',
               start_local='Py_ssize_t',
               rank_transition='Py_ssize_t',
               returns='tuple',
               )
def partition(size):
    """This function takes in the size (nr. of elements) of an array
    and partitions it fairly among the processes.
    Both the starting index and size of the local part are returned.
    If the given size cannot be divided evenly, one additional element
    will be given to the higher ranks.
    """
    # Size and starting index of local part of the data
    size_local = size//nprocs
    start_local = rank*size_local
    # Lowest rank which receives one extra data point
    rank_transition = nprocs + size_local*nprocs - size
    # Correct local size and starting index for processes receiving
    # the extra data point.
    if rank >= rank_transition:
        size_local += 1
        start_local += rank - rank_transition
    return start_local, size_local

# Function for communicating sizes of receive buffers
@cython.header(# Arguments
               N_send='Py_ssize_t[::1]',
               # Locals
               ID_recv='int',
               ID_send='int',
               N_recv='Py_ssize_t[::1]',
               N_recv_max='Py_ssize_t',
               j='int',
               same_N_send='bint',
               max_bfore_rank='Py_ssize_t',
               max_after_rank='Py_ssize_t',
               returns='Py_ssize_t[::1]',
               )
def find_N_recv(N_send):
    """Given the size of arrays to send, N_send, which itself has a
    length of either 1 (same data size to send to every process) or
    n_procs (individual data sizes to send to the processes), this
    function communicates this to all processes, so that everyone knows
    how much to receive from every process. The entrance number rank is
    unused (the process do not send to itself). The maximum number to
    receive is useful when allocating receive buffers, so this number is
    stored in this otherwise unused entrance.
    """
    N_recv = empty(nprocs, dtype=C2np['Py_ssize_t'])
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
# process governing the domain in which the particle is located.
@cython.header(# Arguments
               component='Component',
               reset_buffers='bint',
               # Locals
               ID_recv='int',
               ID_send='int',
               N_local='Py_ssize_t',
               N_needed='Py_ssize_t',
               N_recv='Py_ssize_t[::1]',
               N_recv_j='Py_ssize_t',
               N_recv_max='Py_ssize_t',
               N_recv_tot='Py_ssize_t',
               N_send_j='Py_ssize_t',
               N_send_max='Py_ssize_t',
               N_send_owner='Py_ssize_t',
               N_send_tot='Py_ssize_t',
               N_send_tot_global='Py_ssize_t',
               holes_filled='Py_ssize_t',
               i='Py_ssize_t',
               index_recv_j='Py_ssize_t',
               indices_send_j='Py_ssize_t*',
               indices_send_owner='Py_ssize_t*',
               j='int',
               k='Py_ssize_t',
               k_start='Py_ssize_t',
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
               sendbuf_size_original='Py_ssize_t',
               Δmemory='Py_ssize_t',
               )
def exchange(component, reset_buffers=False):
    """This function will do an exchange of particles between processes,
    so that every particle resides on the process in charge of the
    domain where the particle is located. The variable indices_send
    holds arrays of indices of particles to send to the different
    processes, while particle data is copied to sendbuf before it is
    send. These two variables will grow in size if needed. Call with
    reset_buffers=True to reset these variables to their most basic
    forms, freeing up memory. 
    """
    global sendbuf, sendbuf_mv
    # No need to consider exchange of particles if running serially
    if nprocs == 1:
        return
    # Only particles are exchangeable
    if component.representation != 'particles':
        return
    # The original size of the send buffer
    sendbuf_size_original = sendbuf_mv.shape[0]
    # Extract some variables from component
    N_local = component.N_local
    posx = component.posx
    posy = component.posy
    posz = component.posz
    # The index buffers indices_send[:] increase in size by this amount
    Δmemory = 2 + cast(0.01*N_local/nprocs, 'Py_ssize_t')
    # Reset the number of particles to be sent
    for j in range(nprocs):
        N_send[j] = 0
    # Find out where to send which particle
    for i in range(N_local):
        # Rank of the process that local particle i belongs to
        owner = which_domain(posx[i], posy[i], posz[i])
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
                                              indices_send_sizes[owner]*sizeof('Py_ssize_t'))
    # No need to continue if no particles should be exchanged
    N_send_tot = sum(N_send)
    N_send_tot_global = allreduce(N_send_tot, op=MPI.SUM)
    if N_send_tot_global == 0:
        return
    # Print out exchange message
    if N_send_tot_global == 1:
        masterprint('Exchanging 1 {} ...'.format(component.name[:(len(component.name) - 1)]))
    elif N_send_tot_global > 1:
        masterprint('Exchanging {} {} ...'.format(N_send_tot_global, component.name))
    # Enlarge sendbuf, if necessary
    N_send_max = max(N_send)
    if N_send_max > sendbuf_mv.shape[0]:
        sendbuf = realloc(sendbuf, N_send_max*sizeof('double'))
        sendbuf_mv = cast(sendbuf, 'double[:N_send_max]')
    # Find out how many particles to receive
    N_recv = find_N_recv(N_send)
    # The maximum number of particles to
    # receive is stored in entrance rank.
    N_recv_max = N_recv[rank]
    N_recv_tot = sum(N_recv) - N_recv_max
    # Enlarge the component data attributes, if needed. This may not be
    # strcitly necessary as more particles may be send than received.
    N_needed = N_local + N_recv_tot
    if component.N_allocated < N_needed:
        component.resize(N_needed)
        # Reextract position pointers
        posx = component.posx
        posy = component.posy
        posz = component.posz
    # Extract momenta pointers and all memory views
    # of the possibly resized data.
    momx = component.momx
    momy = component.momy
    momz = component.momz
    posx_mv = component.posx_mv
    posy_mv = component.posy_mv
    posz_mv = component.posz_mv
    momx_mv = component.momx_mv
    momy_mv = component.momy_mv
    momz_mv = component.momz_mv
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
                # Fill the hole with the particle.
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
    component.N_local = N_needed - N_send_tot
    # If reset_buffers == True, reset the global indices_send and
    # sendbuf to their basic forms. This buffer will then be rebuild in
    # future calls.
    if reset_buffers:
        sendbuf = realloc(sendbuf, sendbuf_size_original*sizeof('double'))
        sendbuf_mv = cast(sendbuf, 'double[:sendbuf_size_original]')
        for j in range(nprocs):
            indices_send[j] = realloc(indices_send[j], 1*sizeof('Py_ssize_t'))
            indices_send_sizes[j] = 1
    # Finalize exchange message
    masterprint('done')

# Function for communicating boundary values of a
# domain grid between processes.
@cython.header(# Arguments
               domain_grid='double[:, :, ::1]',
               mode='str',
               # Locals
               i='int',
               index_recv_end_i='Py_ssize_t',
               index_recv_end_j='Py_ssize_t',
               index_recv_end_k='Py_ssize_t',
               index_recv_start_i='Py_ssize_t',
               index_recv_start_j='Py_ssize_t',
               index_recv_start_k='Py_ssize_t',
               index_send_end_i='Py_ssize_t',
               index_send_end_j='Py_ssize_t',
               index_send_end_k='Py_ssize_t',
               index_send_start_i='Py_ssize_t',
               index_send_start_j='Py_ssize_t',
               index_send_start_k='Py_ssize_t',
               j='int',
               k='int',
               operation='str',
               reverse='bint',
               )
def communicate_domain(domain_grid, mode=''):
    """This function can operate in two different modes,
    'add contributions' and 'populate'. The comments in the function
    body describe the case of mode == 'add contributions'.

    Mode 'add contributions':
    This corresponds to local boundaries += nonlocal pseudos and ghosts.
    The pseudo and ghost elements get send to the corresponding
    neighbouring process, where these values are added to the existing
    local values.
    
    Mode 'populate':
    This corresponds to local pseudos and ghosts = nonlocal boundaries.
    The values of the local boundary elements get send to the
    corresponding neighbouring process, where these values replace the
    existing values of the pseudo and ghost elements.
    
    A domain_grid consists of 27 parts; the local bulk,
    6 faces (2 kinds), 12 edges (4 kinds) and 8 corners (8 kinds),
    each with the following dimensions:
    shape[0]*shape[1]*shape[2]  # Bulk
    2*shape[1]*shape[2]         # Lower face
    shape[0]*2*shape[2]         # Lower face
    shape[0]*shape[1]*2         # Lower face
    3*shape[1]*shape[2]         # Upper face
    shape[0]*3*shape[2]         # Upper face
    shape[0]*shape[1]*3         # Upper face
    2*2*shape[2]                # Lower, lower edge
    2*shape[1]*2                # Lower, lower edge
    shape[0]*2*2                # Lower, lower edge
    2*3*shape[2]                # Lower, upper edge
    2*shape[1]*3                # Lower, upper edge
    shape[0]*2*3                # Lower, upper edge
    3*2*shape[2]                # Upper, lower edge
    3*shape[1]*2                # Upper, lower edge
    shape[0]*3*2                # Upper, lower edge
    3*3*shape[2]                # Upper, upper edge
    3*shape[1]*3                # Upper, upper edge
    shape[0]*3*3                # Upper, upper edge
    2*2*2                       # Lower, lower, lower corner
    2*2*3                       # Lower, lower, upper corner
    2*3*2                       # Lower, upper, lower corner
    3*2*2                       # Upper, lower, lower corner
    2*3*3                       # Lower, upper, upper corner
    3*2*3                       # Upper, lower, upper corner
    3*3*2                       # Upper, upper, lower corner
    3*3*3                       # Upper, upper, upper corner
    In the above, shape is the shape of the local grid.
    That is, the total grid has
    (shape[0] + 5)*(shape[2] + 5)*(shape[2] + 5) elements.
    """
    # Dependent on the mode, set the operation to be performed on the
    # received data, and the direction of communication.
    if mode == 'add contributions':
        operation = '+='
        reverse = False
    elif mode == 'populate':
        operation = '='
        reverse = True
    elif not mode:
        abort('communicate_domain called with no mode.\n'
              'Call with mode=\'add contributions\' or mode=\'populate\'.')
    else:
        abort('Mode "{}" not implemented'.format(mode))
    for i in range(-1, 2):
        if i == -1:
            # Send left, receive right
            index_send_start_i = 0
            index_send_end_i   = 2
            index_recv_start_i = ℤ[domain_grid.shape[0]] - 5
            index_recv_end_i   = ℤ[domain_grid.shape[0] - 3]
        elif i == 0:
            # Do not send to or receive from this direction.
            # Include the entire i-dimension of the local bulk.
            index_send_start_i = 2
            index_send_end_i   = ℤ[domain_grid.shape[0] - 3]
            index_recv_start_i = 2
            index_recv_end_i   = ℤ[domain_grid.shape[0] - 3]
        else:  # i == -1
            # Send right, receive left
            index_send_start_i = ℤ[domain_grid.shape[0] - 3]
            index_send_end_i   = ℤ[domain_grid.shape[0]]
            index_recv_start_i = 2
            index_recv_end_i   = 5
        for j in range(-1, 2):
            if j == -1:
                # Send backward, receive forward
                index_send_start_j = 0
                index_send_end_j   = 2
                index_recv_start_j = ℤ[domain_grid.shape[1] - 5]
                index_recv_end_j   = ℤ[domain_grid.shape[1] - 3]
            elif j == 0:
                # Do not send to or receive from this direction.
                # Include the entire j-dimension of the local bulk.
                index_send_start_j = 2
                index_send_end_j   = ℤ[domain_grid.shape[1] - 3]
                index_recv_start_j = 2
                index_recv_end_j   = ℤ[domain_grid.shape[1] - 3]
            else:  # j == -1
                # Send forward, receive backward
                index_send_start_j = ℤ[domain_grid.shape[1] - 3]
                index_send_end_j   = ℤ[domain_grid.shape[1]]
                index_recv_start_j = 2
                index_recv_end_j   = 5
            for k in range(-1, 2):
                # Do not communicate the local bulk
                if i == j == k == 0:
                    continue
                if k == -1:
                    # Send downward, receive upward
                    index_send_start_k = 0
                    index_send_end_k   = 2
                    index_recv_start_k = ℤ[domain_grid.shape[2] - 5]
                    index_recv_end_k   = ℤ[domain_grid.shape[2] - 3]
                elif k == 0:
                    # Do not send to or receive from this direction.
                    # Include the entire k-dimension of the local bulk.
                    index_send_start_k = 2
                    index_send_end_k   = ℤ[domain_grid.shape[2] - 3]
                    index_recv_start_k = 2
                    index_recv_end_k   = ℤ[domain_grid.shape[2] - 3]
                else:  # k == -1
                    # Send upward, receive downward
                    index_send_start_k = ℤ[domain_grid.shape[2] - 3]
                    index_send_end_k   = ℤ[domain_grid.shape[2]]
                    index_recv_start_k = 2
                    index_recv_end_k   = 5
                # Communicate this part
                smart_mpi(domain_grid[index_send_start_i:index_send_end_i,
                                      index_send_start_j:index_send_end_j,
                                      index_send_start_k:index_send_end_k],
                          domain_grid[index_recv_start_i:index_recv_end_i,
                                      index_recv_start_j:index_recv_end_j,
                                      index_recv_start_k:index_recv_end_k],
                          dest  =rank_neighboring_domain(+i, +j, +k),
                          source=rank_neighboring_domain(-i, -j, -k),
                          reverse=reverse,
                          mpifun='Sendrecv',
                          operation=operation)

# Function for cutting out domains as rectangular boxes in the best
# possible way. The return value is an array of 3 elements; the number
# of subdivisions of the box for each dimension. When all dimensions
# cannot be equally divided, the x-dimension is subdivided the most,
# then the y-dimension and lastly the z-dimension.
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
               returns='int[::1]',
               )
def cutout_domains(n, basecall=True):
    """This function works by computing a prime factorization of n
    and then multiplying the smallest factors until 3 remain.
    """
    # Factorize n
    prime_factors = []
    while n > 1:
        for i in range(2, n + 1):
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
                # factorize i to get its prime factors.
                if i_is_prime:
                    prime_factors.append(i)
                else:
                    prime_factors += list(cutout_domains(i, basecall=False))
                n //= i
    # The returned list should always consist of 3 values
    if basecall:
        N_primes = len(prime_factors)
        if N_primes < 4:
            return np.array(sorted(prime_factors + [1]*(3 - N_primes), reverse=True),
                            dtype=C2np['int'])
        else:
            len_prime_factors = len(prime_factors)
            while len_prime_factors > 3:
                prime_factors = sorted(prime_factors, reverse=True)
                prime_factors[len_prime_factors - 2] *= prime_factors[len_prime_factors - 1]
                prime_factors.pop()
                len_prime_factors -= 1
            return np.array(sorted(prime_factors, reverse=True), dtype=C2np['int'])
    return np.array(prime_factors, dtype=C2np['int'])

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
def which_domain(x, y, z):
    x_index = int(x/domain_size_x)
    y_index = int(y/domain_size_y)
    z_index = int(z/domain_size_z)
    return domain_layout[x_index, y_index, z_index]

# This function computes the ranks of the processes governing the
# domain which is located i domains to the right, j domains forward and
# k domains up, relative to the local domain.
@cython.header(# Arguments
               i='int',
               j='int',
               k='int',
               # Locals
               returns='int',
               )
def rank_neighboring_domain(i, j, k):
    return domain_layout[mod(domain_layout_local_indices[0] + i, domain_subdivisions[0]),
                         mod(domain_layout_local_indices[1] + j, domain_subdivisions[1]),
                         mod(domain_layout_local_indices[2] + k, domain_subdivisions[2]),
                         ]

# Very general function for different MPI communications
@cython.pheader(# Arguments
                block_send='object',  # Memoryview of dimension 1, 2 or 3
                block_recv='object',  # Memoryview of dimension 1, 2 or 3
                dest='int',
                source='int',
                reverse='bint',
                mpifun='str',
                operation='str',
                # Local
                arr_recv='object',  # NumPy aray
                arr_send='object',  # NumPy aray
                contiguous_recv='bint',
                contiguous_send='bint',
                data_recv='object',  # NumPy aray
                data_send='object',  # NumPy aray
                dims_recv='Py_ssize_t',
                dims_send='Py_ssize_t',
                i='Py_ssize_t',
                index='Py_ssize_t',
                j='Py_ssize_t',
                k='Py_ssize_t',
                mv_1D='double[:]',
                mv_2D='double[:, :]',
                mv_3D='double[:, :, :]',
                recving='bint',
                sending='bint',
                size_recv='Py_ssize_t',
                size_send='Py_ssize_t',
                sizes_recv='Py_ssize_t[::1]',
                reverse_mpifun_mapping='dict',
                returns='object',  # NumPy array or mpi4py.MPI.Request
                )
def smart_mpi(block_send=(), block_recv=(), dest=-1, source=-1,
              reverse=False, mpifun='', operation='='):
    """This function will do MPI communication. It will send the data in
    the array/memoryview block_send to the process of rank dest
    and receive data into array/memoryview block_recv from rank source.
    The arrays can be of any shape (currently bigger than 0 and less
    than 4) and size and may be different for the two.
    If block_recv is larger than the received data, the extra elements
    in the end will be filled with junk if the dimension of block_recv
    is larger than 1. Though for the sake of performance, always pass
    a fitting block_recv.
    The MPI function to use is specified in the mpifun argument
    (e.g. mpifun='sendrecv' or mpifun='send'). Uppercase communication
    (array communication) is always used, regardless of the case of the
    value of mpifun.
    All arguments are optional, so that it is not needed to specify e.g.
    block_recv when doing a Send. For Cython to be able to compile this,
    a cython.pheader decorator is used (corresponding to cython.ccall
    or cpdef). Also, if a call to smart_mpi results in a receive but not
    a send, block_recv can be passed as the first argument
    instead of block_send (which is not used in this case).
    It is allowed not to pass in a block_recv, even when a message
    should be received. In that case, the recvbuf buffer will be used
    and returned.
    The received data can either be copied into block_recv (overwriting
    existing data) or it can be added to the existing data. Change
    this behaviour through the operation argument (operation='=' or
    operation='+=').
    If the passed blocks are contiguous, they will be used directly
    in the communication (though in the case of block_recv, only when
    operation='='). If not, contiguous buffers will be used. The
    buffers used are the variables sendbuf/sendbuf_mv and
    recvbuf/recvbuf_mv. These will be enlarged if necessary.
    Since the buffers contain doubles, the passed arrays must also
    contain doubles if the buffers are to be used. If communication can
    take place directly without the use of the buffers, the passed
    arrays may contain any type (though the type of the send and recv
    block should always be identical).
    What is returned depends on the choice of mpifun. Whenever a message
    should be received, the passed block_recv is returned (as block_recv
    is populated with values in-place, this is rarely used). When a
    non-blocking send-only is used, the MPI request is returned. In a
    blocking send-only is used, None is returned.
    If reverse == True, the communication is reversed, meaning that
    sending block_send to dist and receiving into block_recv from source
    turns into sending block_recv to source and receiving into
    block_send from dist.
    """
    global sendbuf, sendbuf_mv, recvbuf, recvbuf_mv
    # Sanity check on operation argument
    if operation not in ('=', '+=') and master:
        abort('Operation "{}" is not implemented'.format(operation))
    # Determine whether we are sending and/or receiving
    mpifun = mpifun.lower()
    sending = False
    recving = False
    if 'all' in mpifun:
        sending = True
        recving = True
    else:
        if 'send' in mpifun:
            sending = True
            if dest == -1:
                abort('Cannot send when no destination is given')
        if 'recv' in mpifun:
            recving = True
            if source == -1:
                abort('Cannot receive when no source is given')
    if not sending and not recving:
        if mpifun:
            abort('MPI function "{}" not understood'.format(mpifun))
        else:
            abort('Which MPI function to use is not specified')
    # If requested, reverse the communication direction
    if reverse:
        # Swap the send and receive blocks
        block_send, block_recv = block_recv, block_send
        # Swap the source and destination
        dest, source = source, dest
        # Reverse the MPI function
        reverse_mpifun_mapping = {'recv'    : 'send',
                                  'send'    : 'recv',
                                  'sendrecv': 'sendrecv',
                                   }
        if mpifun not in reverse_mpifun_mapping:
            abort('MPI function "{}" cannot be reversed'.format(mpifun))
        mpifun = reverse_mpifun_mapping[mpifun]
    # If only receiving, block_recv should be
    # accessible as the first argument.
    if not sending and recving and len(block_send) and not len(block_recv):
        block_send, block_recv = block_recv, block_send
    # NumPy arrays over the data
    arr_send = asarray(block_send)
    arr_recv = asarray(block_recv)
    # If the input blocks contain different types (and one of them
    # contain doubles), convert them both to doubles.
    # This is not done in-place, meaning that the passed recv_block will
    # not be changed! The returned block should be used in stead.
    if sending and recving:
        if   (    arr_send.dtype == np.dtype(C2np['double'])
              and arr_recv.dtype != np.dtype(C2np['double'])):
              arr_recv = arr_recv.astype(C2np['double'])
        elif (    arr_send.dtype != np.dtype(C2np['double'])
              and arr_recv.dtype == np.dtype(C2np['double'])):
              arr_send = arr_send.astype(C2np['double'])
    # Are the passed arrays contiguous?
    contiguous_send = arr_send.flags.c_contiguous
    contiguous_recv = arr_recv.flags.c_contiguous
    # Get the dimensionality and sizes of the passed arrays
    dims_send = len(arr_send.shape)
    dims_recv = len(arr_recv.shape)
    size_send = arr_send.size
    # Figure out the size of the data to be received
    size_recv = 0
    if sending and recving:
        if mpifun == 'allgather':
            # A block of size_send is to be received from each process
            size_recv = nprocs*size_send
        elif mpifun == 'allgatherv':
            # The blocks to be received from each process may have
            # different sizes. Communicate these sizes.
            sizes_recv = empty(nprocs, dtype=C2np['Py_ssize_t'])
            Allgather(asarray(size_send, dtype=C2np['Py_ssize_t']), sizes_recv)
        else:
            # Communicate the size of the data to be exchanged
            size_recv = sendrecv(size_send, dest=dest, source=source)
    elif recving:
        # The exact size of the data to receive is not known,
        # but it cannot be larger than the size of the receiver block.
        size_recv = arr_recv.size
    # The recv_block cannot be a scalar NumPy array.
    # Do an in-place reshape to 1D-array of size 1.
    if dims_recv == 0:
        arr_recv.resize(1, refcheck=False)
        dims_recv = 1
    # Based on the contiguousity of the input arrays, assign the names
    # data_send and data_recv to the contiguous blocks of data, which
    # are to be passed into the MPI functions.
    if contiguous_send:
        data_send = arr_send
    else:
        # Expand the send bufffer if necessary
        if size_send > sendbuf_mv.shape[0]:
            sendbuf = realloc(sendbuf, size_send*sizeof('double'))
            sendbuf_mv = cast(sendbuf, 'double[:size_send]')
        data_send = sendbuf_mv[:size_send]
    if contiguous_recv and operation == '=':
        # Only if operation == '=' can we receive
        # directly into the input array.
        data_recv = arr_recv
    else:
        # Expand the receive bufffer if necessary
        if size_recv > recvbuf_mv.shape[0]:
            recvbuf = realloc(recvbuf, size_recv*sizeof('double'))
            recvbuf_mv = cast(recvbuf, 'double[:size_recv]')
        data_recv = recvbuf_mv[:size_recv]
    # When no block_recv is passed, use the recvbuf buffer
    if arr_recv.size == 0:
        # Expand the receive bufffer if necessary
        if size_recv > recvbuf_mv.shape[0]:
            recvbuf = realloc(recvbuf, size_recv*sizeof('double'))
            recvbuf_mv = cast(recvbuf, 'double[:size_recv]')
        data_recv = recvbuf_mv[:size_recv]
        arr_recv = asarray(data_recv)
    # Fill send buffer if this is to be used
    if sending and not contiguous_send:
        index = 0
        if dims_send == 1:
            mv_1D = arr_send
            for i in range(mv_1D.shape[0]):
                sendbuf[i] = mv_1D[i]
        elif dims_send == 2:
            mv_2D = arr_send
            for i in range(mv_2D.shape[0]):
                for j in range(mv_2D.shape[1]):
                    sendbuf[index] = mv_2D[i, j]
                    index += 1
        elif dims_send == 3:
            mv_3D = arr_send
            for i in range(mv_3D.shape[0]):
                for j in range(mv_3D.shape[1]):
                    for k in range(mv_3D.shape[2]):
                        sendbuf[index] = mv_3D[i, j, k]
                        index += 1
    # Do the communication
    if mpifun == 'allgather':
        Allgather(data_send, data_recv)
    elif mpifun == 'allgatherv':
        Allgatherv(data_send, (data_recv, sizes_recv))
    elif mpifun == 'isend':
        return Isend(data_send, dest=dest)
    elif mpifun == 'recv':
        Recv(data_recv, source=source)
    elif mpifun == 'send':
        Send(data_send, dest=dest)
    elif mpifun == 'sendrecv':
        Sendrecv(data_send, recvbuf=data_recv, dest=dest, source=source)
    else:
        abort('MPI function "{}" is not implemented'.format(mpifun))
    # Copy or add the received data from the buffer
    # to the passed block_recv (arr_recv), if needed.
    if not recving:
        return
    index = 0
    if operation == '=' and not contiguous_recv:
        if dims_recv == 1:
            mv_1D = arr_recv
            for i in range(size_recv):
                mv_1D[i] = recvbuf[i]
        elif dims_recv == 2:
            mv_2D = arr_recv
            for i in range(mv_2D.shape[0]):
                for j in range(mv_2D.shape[1]):
                    mv_2D[i, j] = recvbuf[index]
                    index += 1
        elif dims_recv == 3:
            mv_3D = arr_recv
            for i in range(mv_3D.shape[0]):
                for j in range(mv_3D.shape[1]):
                    for k in range(mv_3D.shape[2]):
                        mv_3D[i, j, k] = recvbuf[index]
                        index += 1
        else:
            abort('Cannot handle block_recv of dimension {}'.format(dims_recv))
    elif operation == '+=':
        if dims_recv == 1:
            mv_1D = arr_recv
            for i in range(size_recv):
                mv_1D[i] += recvbuf[i]
        elif dims_recv == 2:
            mv_2D = arr_recv
            for i in range(mv_2D.shape[0]):
                for j in range(mv_2D.shape[1]):
                    mv_2D[i, j] += recvbuf[index]
                    index += 1
        elif dims_recv == 3:
            mv_3D = arr_recv
            for i in range(mv_3D.shape[0]):
                for j in range(mv_3D.shape[1]):
                    for k in range(mv_3D.shape[2]):
                        mv_3D[i, j, k] += recvbuf[index]
                        index += 1
        else:
            abort('Cannot handle block_recv of dimension {}'.format(dims_recv))
    # Return the now populated block_recv (arr_recv)
    return arr_recv

# Cutout domains at import time
cython.declare(domain_subdivisions='int[::1]',
               domain_layout='int[:, :, ::1]',
               domain_layout_local_indices='int[::1]',
               domain_size_x='double',
               domain_size_y='double',
               domain_size_z='double',
               domain_volume='double',
               domain_start_x='double',
               domain_start_y='double',
               domain_start_z='double',
               domain_end_x='double',
               domain_end_y='double',
               domain_end_z='double',
               j='int',
               x_index='int',
               y_index='int',
               z_index='int',
               )
# Number of subdivisions (domains) of the box
# in each of the three dimensions.
domain_subdivisions = cutout_domains(nprocs)
# The global 3D layout of the division of the box
domain_layout = arange(nprocs, dtype=C2np['int']).reshape(domain_subdivisions)
# The indices in domain_layout of the local domain
domain_layout_local_indices = asarray(np.unravel_index(rank, domain_subdivisions), dtype=C2np['int'])
# The size of the domain, which are the same for all of them
domain_size_x = boxsize/domain_subdivisions[0]
domain_size_y = boxsize/domain_subdivisions[1]
domain_size_z = boxsize/domain_subdivisions[2]
domain_volume = domain_size_x*domain_size_y*domain_size_z
# The start and end coordinates of the local domain
domain_start_x = domain_layout_local_indices[0]*domain_size_x
domain_start_y = domain_layout_local_indices[1]*domain_size_y
domain_start_z = domain_layout_local_indices[2]*domain_size_z
domain_end_x = domain_start_x + domain_size_x
domain_end_y = domain_start_x + domain_size_x
domain_end_z = domain_start_x + domain_size_x

# Initialize the send and receive buffers
cython.declare(recvbuf='double*',
               recvbuf_mv='double[::1]',
               sendbuf='double*',
               sendbuf_mv='double[::1]',
               )
sendbuf = malloc(1*sizeof('double'))
sendbuf_mv = cast(sendbuf, 'double[:1]')
recvbuf = malloc(1*sizeof('double'))
recvbuf_mv = cast(recvbuf, 'double[:1]')

# Initialize variables used in the exchange function
cython.declare(N_send='Py_ssize_t[::1]',
               indices_send='Py_ssize_t**',
               indices_send_sizes='Py_ssize_t[::1]',
               )
# This variable stores the number of particles to send to each prcess
N_send = empty(nprocs, dtype=C2np['Py_ssize_t'])
# This Py_ssize_t** variable stores the indices of particles to be send
# to other processes. That is, indices_send[other_rank][i] is the local
# index of some particle which should be send to other_rank.
indices_send = malloc(nprocs*sizeof('Py_ssize_t*'))
for j in range(nprocs):
    indices_send[j] = malloc(1*sizeof('Py_ssize_t'))
# The size of the allocated indices_send[:] memory
indices_send_sizes = ones(nprocs, dtype=C2np['Py_ssize_t'])

