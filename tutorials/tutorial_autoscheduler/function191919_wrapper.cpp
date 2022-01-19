#include "Halide.h"
#include "function191919_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>
using namespace std::chrono;
using namespace std;                
int main(int, char **argv)
{
	double *c_buf00 = (double*)malloc(64*64* sizeof(double));
	parallel_init_buffer(c_buf00, 64*64, (double)63);
	Halide::Buffer<double> buf00(c_buf00, 64,64);
    static int cpt=0;
	double *c_buf01 = (double*)malloc(64*64*320* sizeof(double));
	parallel_init_buffer(c_buf01, 64*64*320, (double)12);
	Halide::Buffer<double> buf01(c_buf01, 64,64,320);
  
    ofstream myfile;
  myfile.open ("example.txt",std::ios_base::app);
    myfile<<"******************************\n Start new"<<cpt<< "\n*************************************\n";
    cpt++;
    bool nb_runs_dynamic = is_nb_runs_dynamic();
    
    if (!nb_runs_dynamic){ 
        
        int nb_exec = get_max_nb_runs(); 
       
        for (int i = 0; i < nb_exec; i++) 
        {  
            myfile<<"first\n";
            auto begin = std::chrono::high_resolution_clock::now(); 
            function191919(buf00.raw_buffer(),buf01.raw_buffer());
           
         for (int y = 0; y < buf00.extent(1); y++) 
             for (int x = 0; x < buf00.extent(0); x++)  myfile << buf00(x, y)<<",";
            auto end = std::chrono::high_resolution_clock::now(); 

            std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush;
            myfile.close();

        }
    }
    
    else{ // Adjust the number of runs depending on the measured time on the firs runs
    
        std::vector<double> duration_vector;
        double duration;
        int nb_exec = get_min_nb_runs();    
        
        for (int i = 0; i < nb_exec; i++) 
        {  
             myfile<<"second\n";
            auto begin = std::chrono::high_resolution_clock::now(); 
            function191919(buf00.raw_buffer(),buf01.raw_buffer());
            auto end = std::chrono::high_resolution_clock::now(); 
            for (int y = 0; y < buf00.extent(1); y++) {
             for (int x = 0; x < buf00.extent(0); x++) 
                  myfile << buf00(x, y)<<",";
            myfile<<"\n";
            }
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000;
            std::cout << duration << " "<< std::flush; 
            duration_vector.push_back(duration);
            myfile.close();
        }

        int nb_exec_remaining = choose_nb_runs(duration_vector);

        for (int i = 0; i < nb_exec_remaining; i++) 
        {  
             myfile<<"3d\n";
            auto begin = std::chrono::high_resolution_clock::now(); 
            function191919(buf00.raw_buffer(),buf01.raw_buffer());
            auto end = std::chrono::high_resolution_clock::now(); 
            for (int y = 0; y < buf00.extent(1); y++) 
             for (int x = 0; x < buf00.extent(0); x++) 
                  myfile << buf00(x, y)<<",";
            std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
        }
    }
     
    std::cout << std::endl;

	return 0; 
}

        