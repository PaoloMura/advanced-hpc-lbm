/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
// #define DEBUG           true

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;        /* density per link */
  float accel;          /* density redistribution */
  float omega;          /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float *speeds0;
  float *speeds1;
  float *speeds2;
  float *speeds3;
  float *speeds4;
  float *speeds5;
  float *speeds6;
  float *speeds7;
  float *speeds8;
} t_speed;

/* struct to hold the MPI values */
typedef struct
{
  int rank;          /* the rank of this process */
  int above;         /* the rank above this process */
  int below;         /* the rank below this process */
  int nprocs;        /* the total number of processes */
  int local_rows;    /* the number of rows allocated to this process */
  int start_row;     /* the start row of this process (excluding halo region) */
  int end_row;       /* the end row of this process (excluding halo region) */
  MPI_Status* status /* status flag used in MPI_Sendrecv calls */
} t_mpi;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_mpi* mpi_params);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles, 
               const int finalRound, const float w_1, const float w_2,
               const float w0, const float w1, const float w2, const t_mpi mpi_params);
int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles, const float w1, const float w2, const t_mpi mpi_params);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* restrict cells, int* restrict obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* restrict cells, int* restrict obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/* Return the number of rows allocated to the given rank */
int get_rows_for_rank(int rank, int nprocs, int total_rows);

/* Return the start row of the given rank (not including halo region) */
int get_start_row(int rank, int nprocs, int total_rows);

/* Return the total number of unocupied cells in this process's grid section */
int get_tot_cells(const t_param params, const t_mpi mpi_params, const int* restrict obstacles)

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;     /* name of the input parameter file */
  char*    obstaclefile = NULL;  /* name of a the input obstacle file */
  t_param  params;               /* struct to hold parameter values */
  t_speed  cells;                /* grid containing fluid densities */
  t_speed  tmp_cells;            /* scratch space */
  t_speed  tmp_tmp_cells;        /* temporary value used for swapping pointers */
  int*     obstacles = NULL;     /* grid indicating which cells are blocked */
  float*   av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct   timeval timstr;                                                             /* structure to hold elapsed time */
  double   tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
  t_mpi    mpi_params;           /* struct to hold MPI values */
  int tot_cells;                 /* sum of unoccupied cells across all ranks */
  float tot_cells_inv            /* inverse of tot_cells */
  float tot_vels;                /* sum of velocities across all ranks */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Setup MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &(mpi_params.nprocs));
  MPI_Comm_rank(MPI_COMM_WORLD, &(mpi_params.rank));

  mpi_params.above = mpi_params.rank + 1;
  if (mpi_params.above == mpi_params.nprocs) mpi_params.above = 0;

  mpi_params.below = mpi_params.rank - 1;
  if (mpi_params.below == -1) mpi_params.below = mpi_params.nprocs - 1;

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &mpi_params);

  /* compute weighting factors */
  const float w_1 = params.density * params.accel / 9.f;
  const float w_2 = params.density * params.accel / 36.f;

  /* constants for the collision step */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* find the total number of unoccupied cells in this process's allocated region */
  const int local_tot_cells = get_tot_cells(params, mpi_params, obstacles);

  /* find the total number of cells across the whole grid */
  MPI_Reduce(&local_tot_cells, &tot_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (mpi_params.rank == 0) tot_cells_inv = 1.f / (float) tot_cells;

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  /* perform a single accelerate flow step before iterating */
  accelerate_flow(params, &cells, obstacles, w_1, w_2, mpi_params);
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    int finalRound = (tt == params.maxIters - 1);
    float tot_vel = timestep(params, &cells, &tmp_cells, obstacles, finalRound, w_1, w_2, w0, w1, w2, mpi_params);
    
    /* calculate the average velocities, aggregating results in rank 0 */
    MPI_Reduce(&tot_vel, &tot_vels, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mpi_params.rank == 0) av_vels[tt] = tot_vel * tot_cells_inv;

    /* need to swap the grid pointers */
    tmp_tmp_cells = cells;
    cells = tmp_cells;
    tmp_cells = tmp_tmp_cells;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, &cells));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  /* TODO: refactor this section s.t. you aggregate results across the ranks
  **       then only print output from rank 0 */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, &cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, &cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  MPI_Finalize();

  return EXIT_SUCCESS;
}

float timestep(const t_param params, 
               t_speed* restrict cells, 
               t_speed* restrict tmp_cells,
               int* restrict obstacles, 
               const int finalRound,
               const float w_1,
               const float w_2,
               const float w0,
               const float w1,
               const float w2,
               const t_mpi mpi_params) {

  /* variables for accelerating velocity */
  float  tot_u = 0.f;    /* accumulated magnitudes of velocity for each cell */

  /* tell the compiler alignment information */
  __assume_aligned(cells->speeds0, 64);
  __assume_aligned(cells->speeds1, 64);
  __assume_aligned(cells->speeds2, 64);
  __assume_aligned(cells->speeds3, 64);
  __assume_aligned(cells->speeds4, 64);
  __assume_aligned(cells->speeds5, 64);
  __assume_aligned(cells->speeds6, 64);
  __assume_aligned(cells->speeds7, 64);
  __assume_aligned(cells->speeds8, 64);

  __assume_aligned(tmp_cells->speeds0, 64);
  __assume_aligned(tmp_cells->speeds1, 64);
  __assume_aligned(tmp_cells->speeds2, 64);
  __assume_aligned(tmp_cells->speeds3, 64);
  __assume_aligned(tmp_cells->speeds4, 64);
  __assume_aligned(tmp_cells->speeds5, 64);
  __assume_aligned(tmp_cells->speeds6, 64);
  __assume_aligned(tmp_cells->speeds7, 64);
  __assume_aligned(tmp_cells->speeds8, 64);

  __assume_aligned(obstacles, 64);

  __assume((params.nx)%2==0);
  __assume((params.ny)%2==0);
  
  /* loop over all cells in this process's jurisdiction */
  for (int jj = 1; jj <= mpi_params.local_rows; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* PROPAGATE STEP*/

      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int y_n = jj + 1;
      const int x_e = (ii < params.nx - 1) ? (ii + 1) : 0;
      const int y_s = jj - 1;
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      const float densities0 = cells->speeds0[ii + jj*params.nx]; /* central cell, no movement */
      const float densities1 = cells->speeds1[x_w + jj*params.nx]; /* east */
      const float densities2 = cells->speeds2[ii + y_s*params.nx]; /* north */
      const float densities3 = cells->speeds3[x_e + jj*params.nx]; /* west */
      const float densities4 = cells->speeds4[ii + y_n*params.nx]; /* south */
      const float densities5 = cells->speeds5[x_w + y_s*params.nx]; /* north-east */
      const float densities6 = cells->speeds6[x_e + y_s*params.nx]; /* north-west */
      const float densities7 = cells->speeds7[x_e + y_n*params.nx]; /* south-west */
      const float densities8 = cells->speeds8[x_w + y_n*params.nx]; /* south-east */

      /* REBOUND STEP */

      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->speeds1[ii + jj*params.nx] = densities3;
        tmp_cells->speeds2[ii + jj*params.nx] = densities4;
        tmp_cells->speeds3[ii + jj*params.nx] = densities1;
        tmp_cells->speeds4[ii + jj*params.nx] = densities2;
        tmp_cells->speeds5[ii + jj*params.nx] = densities7;
        tmp_cells->speeds6[ii + jj*params.nx] = densities8;
        tmp_cells->speeds7[ii + jj*params.nx] = densities5;
        tmp_cells->speeds8[ii + jj*params.nx] = densities6;
      }

      /* COLLISION STEP */

      /* don't consider occupied cells */
      else
      {
        /* compute local density total */
        const float local_density = densities0 + densities1 + densities2 + densities3 + densities4 + densities5 + densities6 + densities7 + densities8;

        /* compute inverse of local density total */
        const float local_density_inv = 1 / local_density;

        /* compute x velocity component */
        const float u_x = (densities1 + densities5 + densities8 - (densities3 + densities6 + densities7)) * local_density_inv;

        /* compute y velocity component */
        const float u_y = (densities2 + densities5 + densities6 - (densities4 + densities7 + densities8)) * local_density_inv;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /* equilibrium densities */

        /* zero velocity density: weight w0 */
        const float d_equ0 = w0 * local_density * (1.f - u_sq * 1.5f);
        /* axis speeds: weight w1 */
        const float d_equ1 = w1 * local_density * (1.f + u_x * 3.f + u_x * u_x * 4.5f - u_sq * 1.5f);
        const float d_equ2 = w1 * local_density * (1.f + u_y * 3.f + u_y * u_y * 4.5f - u_sq * 1.5f);
        const float d_equ3 = w1 * local_density * (1.f - u_x * 3.f + u_x * u_x * 4.5f - u_sq * 1.5f);
        const float d_equ4 = w1 * local_density * (1.f - u_y * 3.f + u_y * u_y * 4.5f - u_sq * 1.5f);
        /* diagonal speeds: weight w2 */
        const float d_equ5 = w2 * local_density * (1.f + (u_x + u_y) * 3.f + (u_x + u_y) * (u_x + u_y) * 4.5f - u_sq * 1.5f);
        const float d_equ6 = w2 * local_density * (1.f + (- u_x + u_y) * 3.f + (- u_x + u_y) * (- u_x + u_y) * 4.5f - u_sq * 1.5f);
        const float d_equ7 = w2 * local_density * (1.f + (- u_x - u_y) * 3.f + (u_x + u_y) * (u_x + u_y) * 4.5f - u_sq * 1.5f);
        const float d_equ8 = w2 * local_density * (1.f + (u_x - u_y) * 3.f + (u_x - u_y) * (u_x - u_y) * 4.5f - u_sq * 1.5f);

        /* relaxation step */
        tmp_cells->speeds0[ii + jj*params.nx] = densities0 + params.omega * (d_equ0 - densities0);
        tmp_cells->speeds1[ii + jj*params.nx] = densities1 + params.omega * (d_equ1 - densities1);
        tmp_cells->speeds2[ii + jj*params.nx] = densities2 + params.omega * (d_equ2 - densities2);
        tmp_cells->speeds3[ii + jj*params.nx] = densities3 + params.omega * (d_equ3 - densities3);
        tmp_cells->speeds4[ii + jj*params.nx] = densities4 + params.omega * (d_equ4 - densities4);
        tmp_cells->speeds5[ii + jj*params.nx] = densities5 + params.omega * (d_equ5 - densities5);
        tmp_cells->speeds6[ii + jj*params.nx] = densities6 + params.omega * (d_equ6 - densities6);
        tmp_cells->speeds7[ii + jj*params.nx] = densities7 + params.omega * (d_equ7 - densities7);
        tmp_cells->speeds8[ii + jj*params.nx] = densities8 + params.omega * (d_equ8 - densities8);

        /* AVERAGE VELOCITIES */

        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf(u_sq);

        /* ACCELERATE FLOW STEP */

        /* if this is the second row of the top rank, 
        ** we're not in the final round, the cell is not occupied and
        ** we don't send a negative density */
        if (jj == mpi_params.local_rows - 1
            && mpi_params.rank == mpi_params.nprocs - 1
            && !finalRound
            && (tmp_cells->speeds3[ii + jj*params.nx] - w_1) > 0.f
            && (tmp_cells->speeds6[ii + jj*params.nx] - w_2) > 0.f
            && (tmp_cells->speeds7[ii + jj*params.nx] - w_2) > 0.f)
        {
          /* increase 'east-side' densities */
          tmp_cells->speeds1[ii + jj*params.nx] += w_1;
          tmp_cells->speeds5[ii + jj*params.nx] += w_2;
          tmp_cells->speeds8[ii + jj*params.nx] += w_2;
          /* decrease 'west-side' densities */
          tmp_cells->speeds3[ii + jj*params.nx] -= w_1;
          tmp_cells->speeds6[ii + jj*params.nx] -= w_2;
          tmp_cells->speeds7[ii + jj*params.nx] -= w_2;
        }
      }
    }
  }

  /* Halo exchange */

  /* Send up, receive from below */
  const int sendup = params.nx * mpi_params.local_rows;
  MPI_Sendrecv(&(tmp_cells->speeds2[sendup]), params.nx, MPI_FLOAT, mpi_params.above, 0, 
                 &(tmp_cells->speeds2[0]), params.nx, MPI_FLOAT, mpi_params.below, 0, MPI_COMM_WORLD, mpi_params.status);
  MPI_Sendrecv(&(tmp_cells->speeds5[sendup]), params.nx, MPI_FLOAT, mpi_params.above, 0, 
                 &(tmp_cells->speeds5[0]), params.nx, MPI_FLOAT, mpi_params.below, 0, MPI_COMM_WORLD, mpi_params.status);
  MPI_Sendrecv(&(tmp_cells->speeds6[sendup]), params.nx, MPI_FLOAT, mpi_params.above, 0, 
                 &(tmp_cells->speeds6[0]), params.nx, MPI_FLOAT, mpi_params.below, 0, MPI_COMM_WORLD, mpi_params.status);

  /* Send down, receive from above */
  const int recabv = params.nx * (mpi_params.local_rows + 1);
  MPI_Sendrecv(&(tmp_cells->speeds4[params.nx]), params.nx, MPI_FLOAT, mpi_params.below, 0, 
                 &(tmp_cells->speeds4[recabv]), params.nx, MPI_FLOAT, mpi_params.above, 0, MPI_COMM_WORLD, mpi_params.status);
  MPI_Sendrecv(&(tmp_cells->speeds7[params.nx]), params.nx, MPI_FLOAT, mpi_params.below, 0, 
                 &(tmp_cells->speeds7[recabv]), params.nx, MPI_FLOAT, mpi_params.above, 0, MPI_COMM_WORLD, mpi_params.status);
  MPI_Sendrecv(&(tmp_cells->speeds8[params.nx]), params.nx, MPI_FLOAT, mpi_params.below, 0, 
                 &(tmp_cells->speeds8[recabv]), params.nx, MPI_FLOAT, mpi_params.above, 0, MPI_COMM_WORLD, mpi_params.status);

  return tot_u;
}

int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles, const float w1, const float w2, const t_mpi mpi_params)
{
  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;

  /* only accelerate flow if the 2nd row of the grid is within your jurisdiction */
  if (jj < mpi_params.start_row || jj > mpi_params.end_row) return EXIT_SUCCESS;

  const int jjj = mpi_params.local_rows - 1;

  /* tell the compiler alignment information */
  __assume_aligned(cells->speeds0, 64);
  __assume_aligned(cells->speeds1, 64);
  __assume_aligned(cells->speeds2, 64);
  __assume_aligned(cells->speeds3, 64);
  __assume_aligned(cells->speeds4, 64);
  __assume_aligned(cells->speeds5, 64);
  __assume_aligned(cells->speeds6, 64);
  __assume_aligned(cells->speeds7, 64);
  __assume_aligned(cells->speeds8, 64);

  __assume_aligned(obstacles, 64);

  __assume((params.nx)%2==0);

  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jjj*params.nx]
        && (cells->speeds3[ii + jjj*params.nx] - w1) > 0.f
        && (cells->speeds6[ii + jjj*params.nx] - w2) > 0.f
        && (cells->speeds7[ii + jjj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speeds1[ii + jjj*params.nx] += w1;
      cells->speeds5[ii + jjj*params.nx] += w2;
      cells->speeds8[ii + jjj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speeds3[ii + jjj*params.nx] -= w1;
      cells->speeds6[ii + jjj*params.nx] -= w2;
      cells->speeds7[ii + jjj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* restrict cells, int* restrict obstacles)
{
  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */

  /* tell the compiler alignment information */
  __assume_aligned(cells->speeds0, 64);
  __assume_aligned(cells->speeds1, 64);
  __assume_aligned(cells->speeds2, 64);
  __assume_aligned(cells->speeds3, 64);
  __assume_aligned(cells->speeds4, 64);
  __assume_aligned(cells->speeds5, 64);
  __assume_aligned(cells->speeds6, 64);
  __assume_aligned(cells->speeds7, 64);
  __assume_aligned(cells->speeds8, 64);

  __assume_aligned(obstacles, 64);

  __assume((params.nx)%2==0);

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        const float local_density = cells->speeds0[ii + jj*params.nx]
                                    + cells->speeds1[ii + jj*params.nx]
                                    + cells->speeds2[ii + jj*params.nx]
                                    + cells->speeds3[ii + jj*params.nx]
                                    + cells->speeds4[ii + jj*params.nx]
                                    + cells->speeds5[ii + jj*params.nx]
                                    + cells->speeds6[ii + jj*params.nx]
                                    + cells->speeds7[ii + jj*params.nx]
                                    + cells->speeds8[ii + jj*params.nx];

        /* inverse of local density total */
        const float local_density_inv = 1 / local_density;

        /* x-component of velocity */
        const float u_x = (cells->speeds1[ii + jj*params.nx]
                      + cells->speeds5[ii + jj*params.nx]
                      + cells->speeds8[ii + jj*params.nx]
                      - (cells->speeds3[ii + jj*params.nx]
                         + cells->speeds6[ii + jj*params.nx]
                         + cells->speeds7[ii + jj*params.nx]))
                     * local_density_inv;
        /* compute y velocity component */
        const float u_y = (cells->speeds2[ii + jj*params.nx]
                      + cells->speeds5[ii + jj*params.nx]
                      + cells->speeds6[ii + jj*params.nx]
                      - (cells->speeds4[ii + jj*params.nx]
                         + cells->speeds7[ii + jj*params.nx]
                         + cells->speeds8[ii + jj*params.nx]))
                     * local_density_inv;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int get_rows_for_rank(int rank, int nprocs, int total_rows)
{
  int rows = total_rows / nprocs;
  int remainder = total_rows % nprocs;
  if (rank < remainder) rows++;
  return rows;
}

int get_start_row(int rank, int nprocs, int total_rows)
{
  int rows_per_rank = total_rows / nprocs;
  int remainder = total_rows % nprocs;
  int add_on = (rank < remainder) ? rank : remainder;
  int start_row = rank * rows_per_rank + add_on;
  return start_row;
}

int get_tot_cells(const t_param params, const t_mpi mpi_params, const int* restrict obstacles) {
  int tot_cells = 0;
  for (int jj=1; jj <= mpi_params.local_rows; jj++) {
    for (int ii=0; ii < params.nx; ii++) {
      if (!(obstacles[jj * params.nx + ii])) tot_cells++;
    }
  }
  return tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_mpi* mpi_params) {
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /* Determine the number of rows to allocate to this process */
  mpi_params->local_rows = get_rows_for_rank(mpi_params->rank, mpi_params->nprocs, params->ny);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */

  cells_ptr->speeds0 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  cells_ptr->speeds1 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  cells_ptr->speeds3 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  cells_ptr->speeds2 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  cells_ptr->speeds4 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  cells_ptr->speeds5 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  cells_ptr->speeds6 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  cells_ptr->speeds7 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  cells_ptr->speeds8 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);

  if (cells_ptr->speeds0 == NULL) die("cannot allocate memory for cells0", __LINE__, __FILE__);
  if (cells_ptr->speeds1 == NULL) die("cannot allocate memory for cells1", __LINE__, __FILE__);
  if (cells_ptr->speeds2 == NULL) die("cannot allocate memory for cells2", __LINE__, __FILE__);
  if (cells_ptr->speeds3 == NULL) die("cannot allocate memory for cells3", __LINE__, __FILE__);
  if (cells_ptr->speeds4 == NULL) die("cannot allocate memory for cells4", __LINE__, __FILE__);
  if (cells_ptr->speeds5 == NULL) die("cannot allocate memory for cells5", __LINE__, __FILE__);
  if (cells_ptr->speeds6 == NULL) die("cannot allocate memory for cells6", __LINE__, __FILE__);
  if (cells_ptr->speeds7 == NULL) die("cannot allocate memory for cells7", __LINE__, __FILE__);
  if (cells_ptr->speeds8 == NULL) die("cannot allocate memory for cells8", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  tmp_cells_ptr->speeds0 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  tmp_cells_ptr->speeds1 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  tmp_cells_ptr->speeds2 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  tmp_cells_ptr->speeds3 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  tmp_cells_ptr->speeds4 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  tmp_cells_ptr->speeds5 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  tmp_cells_ptr->speeds6 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  tmp_cells_ptr->speeds7 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);
  tmp_cells_ptr->speeds8 = (float*)_mm_malloc(sizeof(float) * ((mpi_params->local_rows + 2) * params->nx), 64);

  if (tmp_cells_ptr->speeds0 == NULL) die("cannot allocate memory for tmp_cells0", __LINE__, __FILE__);
  if (tmp_cells_ptr->speeds1 == NULL) die("cannot allocate memory for tmp_cells1", __LINE__, __FILE__);
  if (tmp_cells_ptr->speeds2 == NULL) die("cannot allocate memory for tmp_cells2", __LINE__, __FILE__);
  if (tmp_cells_ptr->speeds3 == NULL) die("cannot allocate memory for tmp_cells3", __LINE__, __FILE__);
  if (tmp_cells_ptr->speeds4 == NULL) die("cannot allocate memory for tmp_cells4", __LINE__, __FILE__);
  if (tmp_cells_ptr->speeds5 == NULL) die("cannot allocate memory for tmp_cells5", __LINE__, __FILE__);
  if (tmp_cells_ptr->speeds6 == NULL) die("cannot allocate memory for tmp_cells6", __LINE__, __FILE__);
  if (tmp_cells_ptr->speeds7 == NULL) die("cannot allocate memory for tmp_cells7", __LINE__, __FILE__);
  if (tmp_cells_ptr->speeds8 == NULL) die("cannot allocate memory for tmp_cells8", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * ((mpi_params->local_rows + 2) * params->nx), 64);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < (mpi_params->local_rows + 2); jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      cells_ptr->speeds0[ii + jj*params->nx] = w0;
      /* axis directions */
      cells_ptr->speeds1[ii + jj*params->nx] = w1;
      cells_ptr->speeds2[ii + jj*params->nx] = w1;
      cells_ptr->speeds3[ii + jj*params->nx] = w1;
      cells_ptr->speeds4[ii + jj*params->nx] = w1;
      /* diagonals */
      cells_ptr->speeds5[ii + jj*params->nx] = w2;
      cells_ptr->speeds6[ii + jj*params->nx] = w2;
      cells_ptr->speeds7[ii + jj*params->nx] = w2;
      cells_ptr->speeds8[ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < (mpi_params->local_rows + 2); jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  mpi_params->start_row = get_start_row(mpi_params->rank, mpi_params->nprocs, params->ny);
  mpi_params->end_row = mpi_params->start_row + mpi_params->local_rows - 1;

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    if (mpi_params->start_row <= yy && yy <= mpi_params->end_row) {
      yy = yy - mpi_params->start_row + 1;
      (*obstacles_ptr)[xx + yy*params->nx] = blocked;
    }
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr, int** obstacles_ptr, float** av_vels_ptr) {
  /*
  ** free up allocated memory
  */
  _mm_free(cells_ptr->speeds0);
  _mm_free(cells_ptr->speeds1);
  _mm_free(cells_ptr->speeds2);
  _mm_free(cells_ptr->speeds3);
  _mm_free(cells_ptr->speeds4);
  _mm_free(cells_ptr->speeds5);
  _mm_free(cells_ptr->speeds6);
  _mm_free(cells_ptr->speeds7);
  _mm_free(cells_ptr->speeds8);

  _mm_free(tmp_cells_ptr->speeds0);
  _mm_free(tmp_cells_ptr->speeds1);
  _mm_free(tmp_cells_ptr->speeds2);
  _mm_free(tmp_cells_ptr->speeds3);
  _mm_free(tmp_cells_ptr->speeds4);
  _mm_free(tmp_cells_ptr->speeds5);
  _mm_free(tmp_cells_ptr->speeds6);
  _mm_free(tmp_cells_ptr->speeds7);
  _mm_free(tmp_cells_ptr->speeds8);

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* restrict cells, int* restrict obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total += cells->speeds0[ii + jj*params.nx];
      total += cells->speeds1[ii + jj*params.nx];
      total += cells->speeds2[ii + jj*params.nx];
      total += cells->speeds3[ii + jj*params.nx];
      total += cells->speeds4[ii + jj*params.nx];
      total += cells->speeds5[ii + jj*params.nx];
      total += cells->speeds6[ii + jj*params.nx];
      total += cells->speeds7[ii + jj*params.nx];
      total += cells->speeds8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        local_density += cells->speeds0[ii + jj*params.nx];
        local_density += cells->speeds1[ii + jj*params.nx];
        local_density += cells->speeds2[ii + jj*params.nx];
        local_density += cells->speeds3[ii + jj*params.nx];
        local_density += cells->speeds4[ii + jj*params.nx];
        local_density += cells->speeds5[ii + jj*params.nx];
        local_density += cells->speeds6[ii + jj*params.nx];
        local_density += cells->speeds7[ii + jj*params.nx];
        local_density += cells->speeds8[ii + jj*params.nx];

        /* compute x velocity component */
        u_x = (cells->speeds1[ii + jj*params.nx]
               + cells->speeds5[ii + jj*params.nx]
               + cells->speeds8[ii + jj*params.nx]
               - (cells->speeds3[ii + jj*params.nx]
                  + cells->speeds6[ii + jj*params.nx]
                  + cells->speeds7[ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speeds2[ii + jj*params.nx]
               + cells->speeds5[ii + jj*params.nx]
               + cells->speeds6[ii + jj*params.nx]
               - (cells->speeds4[ii + jj*params.nx]
                  + cells->speeds7[ii + jj*params.nx]
                  + cells->speeds8[ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
