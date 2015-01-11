ctypedef double* (*vector_func)(double, double, double)
cdef double[:, :, :, ::1] tabulate_vectorfield(int gridsize, vector_func func, double factor, str filename)

cdef double* CIC_grid2coordinates_vector(double[:, :, :, ::1] grid, double x, double y, double z)

from species cimport Particles
cdef  CIC_coordinates2grid(double[:, :, ::1] grid, Particles particle)
