from species cimport Particles
ctypedef double* (*func_ddd_ddd_pxd)(double, double, double)

cdef:
    tuple partition(tuple array_shape)
    double[:, :, :, ::1] tabulate_vectorfield(int gridsize, func_ddd_ddd_pxd func, double factor, str filename)
    double CIC_grid2coordinates_scalar(double[:,:,::1] grid, double x, double y, double z)
    double* CIC_grid2coordinates_vector(double[:,:,:,::1] grid, double x, double y, double z)
    CIC_particles2grid(Particles particles, double[:,:,::1] grid)
    communicate_boundaries(double[:,:,::1] grid, int mode=*)
    communicate_ghosts(double[:,:,::1] grid)
    domain2PM(double[:,:,::1] domain_grid, double[:,:,::1] PM_grid)
    PM2domain(double[:,:,::1] domain_grid, double[:,:,::1] PM_grid)
