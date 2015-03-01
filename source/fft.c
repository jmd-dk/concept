#include <fftw3-mpi.h>
#include <stdbool.h>
#include <stdio.h>

/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
gridstart_local_x AND ptrdiff_t gridstart_local_y ARE INCLUDED IN THE RETURN STRUCT, BUT NOT USED (RIGHT?)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/


/* This file defines the functions fftw_setup and fftw_clean, which
together with fftw_execute (included in fftw3-mpi.h) constitutes the
necessary functions for using FFTW to do parallel, real, 3D in-place
transforms through Cython.
*/


/* Note on indexing.
- In real space, before any transformations:
  grid[i, j, k] --> grid[((i - gridstart_local_x)*gridsize_y + j)*gridsize_padding + k], provided we execute this on the correct process, where (gridstart_local_x <= i && i < gridstart_local_x + gridsize_local_x)
  That is, each process can acces every element in its slab by:
  for (i = 0; i < gridsize_local_x; ++i)
  {
    for (j = 0; j < gridsize_y; ++j)
    {
      for (k = 0; k < gridsize_z; ++k)
      {
        double element = grid[(i*gridsize_y + j)*gridsize_padding + k]; // This is the [i + gridstart_local_x, j, k]'th element
      }
    }
  }
- After a forward, inplace r2c transformation:
  grid[i, j, k] --> grid[((j - gridstart_local_y)*gridsize_x + i)*gridsize_padding + k], provided we execute this on the correct process, where (gridstart_local_y <= j && j < gridstart_local_y + gridsize_local_y)
  That is, each process can acces every element in its slab by:
  for (i = 0; i < gridsize_x; ++i)
  {
    for (j = 0; j < gridsize_local_y; ++j)
    {
      for (k = 0; k < gridsize_z; ++k)
      {
        double element = grid[(j*gridsize_x + i)*gridsize_padding + k]; // This is the [i, j + gridstart_local_y, k]'th element
      }
    }
  }
*/


// Struct for return type of the fftw_setup function
struct fftw_return_struct {
  ptrdiff_t gridsize_local_x;
  ptrdiff_t gridsize_local_y;
  ptrdiff_t gridstart_local_x;
  ptrdiff_t gridstart_local_y;
  double* grid;
  fftw_plan plan_forward;
  fftw_plan plan_backward;
};
// This function initializez fftw_mpi, allocates a grid, desides the local lengths and starting indices and creates forwards and backwards plans
struct fftw_return_struct fftw_setup(ptrdiff_t gridsize_x, ptrdiff_t gridsize_y, ptrdiff_t gridsize_z)
{
  // Size of last dimension with padding
  ptrdiff_t gridsize_padding = 2*(gridsize_z/2 + 1);

  // Initialize parallel fftw (note that MPI_Init should not be called, as mpi is already running via mpi4py)
  fftw_mpi_init();

  // Process identification
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  int root = 0;
  bool master = rank == root;

  // Declaration and allocation of the (local part of the) grid. This also initializes gridsize_local_(x/y) and gridstart_local_(x/y).
  ptrdiff_t gridsize_local_x, gridsize_local_y, gridstart_local_x, gridstart_local_y;
  double* grid = fftw_alloc_real(fftw_mpi_local_size_3d_transposed(gridsize_x, gridsize_y, gridsize_padding, MPI_COMM_WORLD, &gridsize_local_x, &gridstart_local_x,
                                                                                                                             &gridsize_local_y, &gridstart_local_y));
  // The master process reads in previous wisdom and broadcasts it
  char wisdom_file_buffer[100];
  sprintf(wisdom_file_buffer, ".fftw_wisdom_gridsize=%td_nprocs=%i", gridsize_x, nprocs);
  const char* wisdom_file = &wisdom_file_buffer[0];
  int previous_wisdom;
  if (master)
  {
    previous_wisdom = fftw_import_wisdom_from_filename(wisdom_file);
    if (! previous_wisdom)
    {
      // Grammar is important!
      if (nprocs == 1)
      {
        printf("Acquiring FFTW wisdom for grid of linear size %td on %i process\n", gridsize_x, nprocs);
      }
      else
      {
        printf("Acquiring FFTW wisdom for grid of linear size %td on %i processes\n", gridsize_x, nprocs);
      }
    }
  }
  fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);

  // Create the two plans
  fftw_plan plan_forward  = fftw_mpi_plan_dft_r2c_3d(gridsize_x, gridsize_y, gridsize_z,  // In order of patience: FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE
                                                     grid, (fftw_complex*) grid, MPI_COMM_WORLD, FFTW_PATIENT | FFTW_MPI_TRANSPOSED_OUT);
  fftw_plan plan_backward = fftw_mpi_plan_dft_c2r_3d(gridsize_x, gridsize_y, gridsize_z,
                                                     (fftw_complex*) grid, grid, MPI_COMM_WORLD, FFTW_PATIENT | FFTW_MPI_TRANSPOSED_IN);

  // If new wisdom is acquired, the master process saves it to disk
  fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
  if (master && ! previous_wisdom)
  {
    fftw_export_wisdom_to_filename(wisdom_file);
  }

  // Return a struct with variables
  struct fftw_return_struct fftw_struct = {gridsize_local_x, gridsize_local_y, gridstart_local_x, gridstart_local_y, grid, plan_forward, plan_backward};
  return fftw_struct;
}


// Call this function when all FFT work is done
void fftw_clean(double* grid, fftw_plan plan_forward, fftw_plan plan_backward)
{
  fftw_free(grid);
  fftw_destroy_plan(plan_forward);
  fftw_destroy_plan(plan_backward);
  fftw_mpi_cleanup();
}

