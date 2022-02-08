#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function750061_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function750061");
	var i0("i0", 0, 64), i1("i1", 0, 16), i2("i2", 0, 128);
	input icomp00("icomp00", {i0,i1,i2}, p_float64);
	input input01("input01", {i1,i0}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression(expr(2.000)*icomp00(i0, i1, i2) + input01(i1, i0) + 0.130);
	buffer buf00("buf00", {64,16,128}, p_float64, a_output);
	buffer buf01("buf01", {16,64}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	comp00.store_in(&buf00);
	tiramisu::codegen({&buf00,&buf01}, "function750061.o"); 
	return 0; 
}