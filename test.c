#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int nprocs, rank, flag, strLen;
char hostname[MPI_MAX_PROCESSOR_NAME];
char hostid[3];

void print(char* msg) {
    printf("(%s,%d): %s", hostid, rank, msg);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    MPI_Initialized(&flag);
    if ( flag != 1 ) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Get_processor_name(hostname, &strLen);
    strncpy(hostid, hostname+7, 3);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    print("Hello\n");
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
