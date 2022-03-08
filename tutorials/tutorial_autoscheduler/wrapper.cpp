#include <iostream>
#include "Halide.h"
#include "tiramisu/utils.h"

#include "wrapper.h"

using namespace std;

int main(int, char **argv)
{
    Halide::Buffer<double> buf00(128,129,256);
    Halide::Buffer<double> buf02(384,129,256);
    Halide::Buffer<double> buf01(130);


    std::vector<double> duration_vector;
    double start, end;
    
//    for (int i = 0; i < 1; ++i)
//        conv(buf01.raw_buffer(),buf001.raw_buffer());
    
    for (int i = 0; i < 1; i++)
    {
        start = rtclock();
        conv(buf00.raw_buffer(),buf01.raw_buffer(),buf02.raw_buffer());
        end = rtclock();
        
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << median(duration_vector) << std::endl;
    
    return 0;
}
