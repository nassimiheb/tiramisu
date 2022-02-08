#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function750060_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function750060");
	var i0("i0", 0, 32), i1("i1", 0, 128), i2("i2", 0, 32), i3("i3", 0, 32), i4("i4", 0, 64), i5("i5", 0, 32);
	input icomp00("icomp00", {i0,i1,i2,i3}, p_float64);
	input input01("input01", {i4,i5}, p_float64);
	computation comp00("comp00", {i0,i1,i2,i3,i4,i5},  p_float64);
	comp00.set_expression(icomp00(i0, i1, i2, i3)*input01(i4, i5));
	buffer buf00("buf00", {32,128,32,32}, p_float64, a_output);
	buffer buf01("buf01", {64,32}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	comp00.store_in(&buf00, {i0,i1,i2,i3});

	prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00,&buf01}, "function750060.o", "./function750060_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function750060_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}