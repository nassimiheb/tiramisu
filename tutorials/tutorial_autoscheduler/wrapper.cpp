#include "Halide.h"
#include "wrapper.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

#define MAX_RAND 200

using namespace std::chrono;
using namespace std;

int main(int, char **argv)
{
//     Halide::Buffer<int32_t> b_A(N, N);


//     int *b_At = (int*)malloc(N * N * sizeof(int));
//     parallel_init_buffer(b_At, N * N,  (int32_t)49);
//     Halide::Buffer<int32_t> b_A(b_At, N, N);

    Halide::Buffer<double> b_A(N,N);
    Halide::Buffer<double> b_B(N,N);
    
    std::vector<double> duration_vector;
    double start, end;
    
    for (int i = 0; i < 2; ++i) 
        jacobi_2d(b_A.raw_buffer(), b_B.raw_buffer());
    
    for (int i = 0; i < 15; i++)
    {
        start = rtclock();
        jacobi_2d(b_A.raw_buffer(), b_B.raw_buffer());
        end = rtclock();
        
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << median(duration_vector) << std::endl;
    
    return 0;
}
