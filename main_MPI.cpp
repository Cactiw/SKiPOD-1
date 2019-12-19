//
// Created by Cactiw on 18.12.2019.
//

#include <iostream>
#include <vector>
#include <ctime>
#include <mpi.h>
#include <random>

const double EPS = 1E-9;

int NUM_THREADS[] = {1, 2, 4, 8, 16, 32, 64};
int NUMBER_THREADS = 7;
int SIZES[] = {4, 128, 256, 512, 1024, 2048, 4096};
int SIZES_NUM = 6;

void print_matrix(int *a[], long long m, long long n) {
    for (long long i = 0; i < m; ++i) {
        for (long long j = 0; j < n; ++j) {
            std::cout << a[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int count_rank(int *a[], long long m, long long n, int curr_proc, int proc_num) {
//    if (a) {
//        return 0;
//    }
    std::cout << proc_num << " " << curr_proc << std::endl;
    int rank = std::max(n, m);
    std::vector<bool> line_used(n);

    for (int i = 0; i < m; ++i) {
        int j;
        for (j = 0; j < n; ++j)
            if (!line_used[j] && abs(a[j][i]) > EPS)
                break;
        if (j == n)
            --rank;
        else {
            line_used[j] = true;
            // parallel for
            int process = m / proc_num;  // Сколько строк обработает каждый процесс
            // (может быть больше на 1 для конкретного процесса
            int recvcounts[proc_num], displs[proc_num];
            recvcounts[0] = process;
            displs[0] = 0;
            int offset = recvcounts[0], remain = m % proc_num;
            for (int z = 1; z < proc_num; ++z) {
                recvcounts[z] = process;
                displs[z] = offset;
                offset += process;
                if (remain > 0) {
                    recvcounts[z] += 1;
                    offset += 1;
                }
            }
            for (int p = displs[curr_proc]; p < displs[curr_proc] + recvcounts[curr_proc]; ++p)
                a[j][p] /= a[j][i];
            //auto buf = (int *) malloc(m * n * sizeof(int));
            //std::cout << "Initializing first gather..." << std::endl;
            //print_matrix(a, m, n);
            MPI_Allgatherv(&a[displs[curr_proc]], recvcounts[curr_proc], MPI_INT, a, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            //std::cout << "Gathered first" << std::endl;
            //print_matrix(a, m, n);
            // parallel for
            process = n / proc_num;
            //std::cout << process * curr_proc << " " << process * curr_proc + process << std::endl;
            for (int k = process * curr_proc; k < process * curr_proc + process; ++k) {
                //std::cout << k << std::endl;
                if (k != j && abs(a[k][i]) > EPS)
                    for (int p = i + 1; p < m; ++p)
                        a[k][p] -= a[j][p] * a[k][i];
            }
            //std::cout << "Initializing second gather..." << std::endl;
            MPI_Allgather(&a[curr_proc * process], process, MPI_INT, a, process, MPI_INT, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            //std::cout << "Gathered second" << std::endl;
        }
    }
    return rank;
}


int main(int argc, char **argv) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 2000);
    int errCode, numtasks, rank;

    if ((errCode = MPI_Init(&argc, &argv)) != MPI_SUCCESS) {
        std::cerr << "MPI_Init error: code " << errCode << std::endl;
        MPI_Abort(MPI_COMM_WORLD, errCode);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Hello World from process " << rank << " of " << numtasks << " processes" << std::endl;

    for (int z = 0; z < SIZES_NUM; ++z) {
        long long m = SIZES[z], n = SIZES[z];
        //a.resize(m);
        std::cout << "Generating matrix with " << m << "x" << n << "..." << std::endl;
        int **a = (int **) malloc(m * n * sizeof(int*) + MPI_BSEND_OVERHEAD);
        for (long long i = 0; i < m; ++i) {
            a[i] = (int *) malloc(n * sizeof(int));
            if (rank == 0) {
                for (long long j = 0; j < n; ++j) {
                    a[i][j] = distribution(generator);
                }
            }
        }
        std::cout << "Matrix generated." << std::endl;
        if (rank == 0)
            print_matrix(a, m, n);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(a, n * m, MPI_INT, 0, MPI_COMM_WORLD);
        //MPI_Scatter(a, n * m, MPI_INT, a, n * m, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Matrix broadcasted." << std::endl;
        if (rank != 0)
            print_matrix(a, m, n);
        MPI_Barrier(MPI_COMM_WORLD);

        double start_time = MPI_Wtime();
        std::cout << "Counting rank with " << numtasks << " threads ..." << std::endl;
        //std::cout << rank(a, n, m);
        std::cout << count_rank(reinterpret_cast<int **>(reinterpret_cast<int *>(a)), m, n, rank, numtasks)
                  << std::endl;
        std::cout << "Computation took " << MPI_Wtime() - start_time << " seconds to complete" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Freeing memory..." << std::endl;
        for (int i = 0; i < m; ++i) {
            free(a[i]);
        }
        free(a);
        std::cout << "Memory freed" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
