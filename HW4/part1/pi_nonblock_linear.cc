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
    MPI_Request requests[world_size - 1];

    MPI_Barrier(MPI_COMM_WORLD);

    while (n--) {
        x = rand_r(&seed) / ((float)RAND_MAX);
        y = rand_r(&seed) / ((float)RAND_MAX);
        z = x * x + y * y;
        if ( z <= 1.0 ) {
            count++;
        }
    }

    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Isend(&count, 1, MPI_LONG_LONG_INT, 0, world_rank, MPI_COMM_WORLD, &requests[world_rank - 1]);
        MPI_Finalize();
        return 0;
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        
        total = count;
        for (int i = 1 ; i < world_size ; ++i) {
            MPI_Irecv(&count, 1, MPI_LONG_LONG_INT, i, i, MPI_COMM_WORLD, &requests[i - 1]);
            total += count;
        }
        MPI_Waitall(world_size - 1, requests, MPI_STATUS_IGNORE);
    }

    if (world_rank == 0)
    {
        // TODO: PI result

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
