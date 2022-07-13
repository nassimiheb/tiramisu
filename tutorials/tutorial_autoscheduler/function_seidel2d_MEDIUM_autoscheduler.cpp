#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_seidel2d_MEDIUM_wrapper.h"


using namespace tiramisu;

const std::string py_cmd_path = "/usr/bin/python";
const std::string py_interface_path = "/home/afif/multi/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";

int main(int argc, char **argv)

{
    tiramisu::init("function_seidel2d_MEDIUM");
   
    var i_f("i_f", 0, 450), j_f("j_f", 0, 450);
    var t("t", 0, 128), i("i", 1, 450-1), j("j", 1, 450-1);

    input A("A", {i_f, j_f}, p_float64);

    computation comp_A_out("comp_A_out", {t,i,j}, (A(i-1, j-1) + A(i-1, j) + A(i-1, j+1)
		                               + A(i, j-1) + A(i, j) + A(i, j+1)
		                               + A(i+1, j-1) + A(i+1, j) + A(i+1, j+1))*0.11111111);

    buffer b_A("b_A", {450,450}, p_float64, a_output);    

    A.store_in(&b_A);
    comp_A_out.store_in(&b_A, {i,j});

    prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A}, "function_seidel2d_MEDIUM.o", "./function_seidel2d_MEDIUM_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, model_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function_seidel2d_MEDIUM_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}
