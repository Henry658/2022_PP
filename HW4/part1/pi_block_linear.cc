#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    // Get the number of processes
    // https://www.open-mpi.org/doc/v4.0/man3/MPI_Comm_size.3.php
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    // https://www.open-mpi.org/doc/v4.0/man3/MPI_Comm_rank.3.php
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    unsigned int seed = (time(NULL) * world_rank);
    long long int count = 0, total = 0, n = tosses / world_size;
    float x , y , z;

    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);

    while (n--) {
        x = rand_r(&seed) / ((float)RAND_MAX);
        y = rand_r(&seed) / ((float)RAND_MAX);
        z = x * x + y * y;
        if ( z <= 1.0 ) {
            count++;
        }
    }

    if (world_rank > 0) {
        // TODO: handle workers
        //send
        // MPI_Send( void* data ,int count, MPI_Datatype datatype, int destination,
        //           int tag, MPI_Comm communicator)
        MPI_Send(&count, 1, MPI_LONG_LONG_INT, 0, world_rank, MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
    }
    else {
        // TODO: master
        //recive
        total = count;
        for (int i = 1 ; i < world_size ; ++i) {
            // MPI_Recv(void* data, int count, MPI_Datatype datatype,
            // int source, int tag, MPI_Comm communicator, MPI_Status* status)
            MPI_Recv(&count, 1, MPI_LONG_LONG_INT, i, i, MPI_COMM_WORLD, &status);
            total += count;
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result

        pi_result = (double)total / tosses;
        pi_result *= 4.0;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
