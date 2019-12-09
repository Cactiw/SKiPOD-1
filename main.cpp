#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include <random>

const double EPS = 1E-9;

int NUM_THREADS[] = {1, 2, 4, 8, 16, 32, 64};
int NUMBER_THREADS = 7;

int rank(int * a[], long long m, long long n) {
//    if (a) {
//        return 0;
//    }

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
            #pragma omp parallel for
            for (int p = i + 1; p < m; ++p)
                a[j][p] /= a[j][i];
            #pragma omp parallel for
            for (int k = 0; k < n; ++k)
                if (k != j && abs(a[k][i]) > EPS)
                    for (int p = i + 1; p < m; ++p)
                        a[k][p] -= a[j][p] * a[k][i];
        }
    }
    return rank;
}


int main() {
    //std::vector<std::vector<int>> a;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,2000);
    // long long m = double (std::rand()) / RAND_MAX * 10000, n = double (std::rand()) / RAND_MAX * 1000;
    long long m = 2000, n = 2000;
    //a.resize(m);
    std::cout << "Generating matrix with " << m << "x" << n << "..." << std::endl;
    int** a = (int**)malloc(m * n * sizeof(int));
    for (long long i = 0; i < m; ++i) {

        a[i] = (int*)malloc(n * sizeof(int));
        for (long long j = 0; j < n; ++j) {
            a[i][j] = distribution(generator);
        }
    }
    std::cout << "Matrix generated." << std::endl;
    #pragma omp barrier
    for (int i = 0; i < NUMBER_THREADS; ++i) {
        omp_set_num_threads(NUM_THREADS[i]);
        double start_time = omp_get_wtime();
        std::cout << "Counting rank with " << NUM_THREADS[i] << " threads ..." << std::endl;
        //std::cout << rank(a, n, m);
        std::cout << rank(reinterpret_cast<int **>(reinterpret_cast<int *>(a)), m, n) << std::endl;
        std::cout << "Computation took " << omp_get_wtime() - start_time << " seconds to complete" << std::endl;
        #pragma omp barrier
    }
    std::cout << "Freeing memory..." << std::endl;
    for (int i = 0; i < m; ++i) {
        free(a[i]);
    }
    free(a);
    std::cout << "Memory freed" << std::endl;

    return 0;
}
