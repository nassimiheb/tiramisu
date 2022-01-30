#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function191919_wrapper.h"

using namespace tiramisu;
// Set to true to perform autoscheduling
bool perform_autoscheduling = true;

// Path to python (please give absolute path)
const std::string py_cmd_path = "/usr/bin/python";

// Path to a script that executes the ML model (please give absolute path)
const std::string py_interface_path = "/data/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";
int main(int argc, char **argv){                
	tiramisu::init("function191919");
	var i0("i0", 1, 2049), i1("i1", 1, 257), i2("i2", 0, 256), i0_p1("i0_p1", 0, 2050), i1_p1("i1_p1", 0, 258);
	input icomp00("icomp00", {i0_p1,i1_p1}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression(((icomp00(i0, i1) - icomp00(i0 + 1, i1))*(icomp00(i0, i1 - 1) + icomp00(i0, i1 + 1) + icomp00(i0 - 1, i1 - 1) + icomp00(i0 - 1, i1 + 1) + icomp00(i0 + 1, i1 - 1) + icomp00(i0 + 1, i1 + 1) + 1.220) + icomp00(i0 - 1, i1))/(icomp00(i0, i1) - icomp00(i0 + 1, i1)));
	buffer buf00("buf00", {2050,258}, p_float64, a_output);
	icomp00.store_in(&buf00);
	comp00.store_in(&buf00, {i0,i1});

	if (!perform_autoscheduling)
    	{
        	tiramisu::codegen({&buf00}, "function191919.o");

        	return 0;
    	}


	prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = 10;
	const int max_depth = 3;
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00}, "function191919.o", "./function191919_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space_random_matrix("./function191919_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}
