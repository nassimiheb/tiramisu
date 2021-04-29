#include <iostream>
#include "Halide.h"
#include "tiramisu/utils.h"

#include "wrapper.h"

using namespace std;

int main(int, char **argv)
{
    Halide::Buffer<_Float64> buf01(3,1024, 1024,"buffer01");
    init_buffer(buf01, (_Float64)2);

    Halide::Buffer<_Float64> buf02(3,1024, 1024,"buffer01");

    init_buffer(buf02, (_Float64)2);
    
    std::vector<double> duration_vector;
    double start, end;
    
    
    
    for (int i = 0; i < 2; i++)
    {
        start = rtclock();
        conv(buf01.raw_buffer(),buf02.raw_buffer());
        end = rtclock();
        
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << median(duration_vector) << std::endl;
    
    return 0;
}
