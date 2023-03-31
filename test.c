#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 4
#define COLS 16

int cols_for_rank(int rank, int nprocs);

void initialise(float** grid, float** sendbuf, float** recvbuf, float **printbuf, int cols, int rank, int nprocs);

void printGrid(float* grid, float* sendbuf, float* recvbuf, float* printbuf, int rank, int cols, int nprocs, MPI_Status* status);

void haloExchange(float* grid, float* sendbuf, float* recvbuf, int rank, int left, int right, int cols, MPI_Status* status);

void cleanup(float** grid, float** sendbuf, float** recvbuf);

int main(int argc, char* argv[])
{
    int nprocs;          // The total number of processes
    int rank;            // My rank
    int left;            // The rank of my predecessor
    int right;           // The rank of my successor
    int tag = 0;        // Tag used in Sendrecv calls
    int local_cols;     // My allocated number of columns
    float *grid;        // My portion of the grid
    float *sendbuf;     // Send buffer
    float *recvbuf;     // Receive buffer
    float *printbuf;    // Print buffer
    MPI_Status status;  // Status used in Recv calls

    MPI_Init(&argc, &argv);    
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Find the ranks of the left and right processes.
    right = (rank + 1) % nprocs;
    left = rank > 0 ? rank - 1 : nprocs - 1;

    local_cols = cols_for_rank(rank, nprocs);
    if (local_cols < 3) {
        fprintf(stderr,"Error: too many processes, local_cols < 3\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    initialise(&grid, &sendbuf, &recvbuf, &printbuf, local_cols, rank, nprocs);

    printGrid(grid, sendbuf, recvbuf, printbuf, rank, local_cols, nprocs, &status);

    haloExchange(grid, sendbuf, recvbuf, rank, left, right, local_cols, &status);

    printGrid(grid, sendbuf, recvbuf, printbuf, rank, local_cols, nprocs, &status);

    cleanup(&grid, &sendbuf, &recvbuf);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

int cols_for_rank(int rank, int nprocs) {
    int cols = COLS / nprocs;
    if (rank < COLS % nprocs) cols++;
    return cols;
}

void initialise(float** grid, float** sendbuf, float** recvbuf, float ** printbuf, int cols, int rank, int nprocs) {
    *grid = (float *)malloc(sizeof(float) * ROWS * (cols + 2));
    *sendbuf = (float *)malloc(sizeof(float) * ROWS);
    *recvbuf = (float *)malloc(sizeof(float) * ROWS);
    int max_cols = cols_for_rank(0, nprocs);
    *printbuf = (float *)malloc(sizeof(float) * (max_cols + 2));

    for (int ii=0; ii < ROWS; ii++) {
        for (int jj=0; jj < cols + 2; jj++) {
            (*grid)[ii * (cols + 2) + jj] = (float)rank;
        }
    }
}

void printGrid(float* grid, float* sendbuf, float* recvbuf, float* printbuf, int rank, int cols, int nprocs, MPI_Status* status) {
    int remote_cols;
    for (int ii=0; ii < ROWS; ii++) {
        // Only the master rank (0) prints the grid
        if (rank == 0) {
            // First it prints its own row
            printf("%2.1f|", grid[ii * (cols + 2)]);
            for (int jj=1; jj < cols + 1; jj++) {
                printf("%2.1f ", grid[ii * (cols + 2) + jj]);
            }
            printf("|%2.1f  ", grid[ii * (cols + 2) + cols + 1]);
            // Then for each subsequent rank...
            for (int kk=1; kk < nprocs; kk++) {
                // It receives its row
                remote_cols = cols_for_rank(kk, nprocs);
                MPI_Recv(printbuf, remote_cols + 2, MPI_FLOAT, kk, 0, MPI_COMM_WORLD, status);
                // And prints its values
                printf("%2.1f|", printbuf[0]);
                for (int jj=1; jj < remote_cols + 1; jj++) {
                    printf("%2.1f", printbuf[jj]);
                    if (jj != remote_cols) printf(" ");
                }
                printf("|%2.1f  ", printbuf[remote_cols + 1]);
            }
            printf("\n");
        } else {
            MPI_Send(&grid[ii * (cols + 2)], cols + 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }
    if (rank == 0) printf("\n");
}

void haloExchange(float* grid, float* sendbuf, float* recvbuf, int rank, int left, int right, int cols, MPI_Status* status) {
    // Send right

    // Pack the send buffer
    for (int ii=0; ii < ROWS; ii++)
        sendbuf[ii] = grid[ii * (cols + 2) + cols];

    // Make the sendrecv call
    MPI_Sendrecv(sendbuf, ROWS, MPI_FLOAT, right, 0, 
                 recvbuf, ROWS, MPI_FLOAT, left, 0, MPI_COMM_WORLD, status);
    
    // Unpack the receive buffer
    for (int ii=0; ii < ROWS; ii++)
        grid[ii * (cols + 2)] = recvbuf[ii];


    // Send left

    // Pack the send buffer
    for (int ii=0; ii < ROWS; ii++)
        sendbuf[ii] = grid[ii * (cols + 2) + 1];
    
    // Make the sendrecv call
    MPI_Sendrecv(sendbuf, ROWS, MPI_FLOAT, left, 0, 
                 recvbuf, ROWS, MPI_FLOAT, right, 0, MPI_COMM_WORLD, status);
    
    // Unpack the receive buffer
    for (int ii=0; ii < ROWS; ii++)
        grid[ii * (cols + 2) + cols + 1] = recvbuf[ii];
}

void cleanup(float** grid, float** sendbuf, float** recvbuf) {
    free(*grid);
    free(*sendbuf);
    free(*recvbuf);
}
