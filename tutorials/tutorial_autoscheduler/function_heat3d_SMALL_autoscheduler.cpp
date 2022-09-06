#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_heat3d_SMALL_wrapper.h"

using namespace tiramisu;
const std::string py_cmd_path = "/usr/bin/python";
const std::string py_interface_path = "/home/afif/multi/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";
int main(int argc, char **argv)
{
    tiramisu::init("function_heat3d_SMALL");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------     
    //for heat3d_init
    var x_in("x_in", 0, 34), y_in("y_in", 0, 34), z_in("z_in", 0, 34), t_in("t_in", 0, 96);
    //for heat3d_c
    var x("x",1,34-1), y("y",1,34-1), z("z",1,34-1), t("t",0,96);
    
    input icomp_heat3d("icomp_heat3d", {z,y,x}, p_float64);

    computation comp_heat3d("comp_heat3d",{t,z,y,x},p_float64);
    comp_heat3d.set_expression(
		icomp_heat3d(z,y,x) +
		    (icomp_heat3d(z-1,y,x) - icomp_heat3d(z,y,x)*2.0 + icomp_heat3d(z+1,y,x)
			 + icomp_heat3d(z,y-1,x) - icomp_heat3d(z,y,x)*2.0 + icomp_heat3d(z,y+1,x)
			 + icomp_heat3d(z,y,x-1) - icomp_heat3d(z,y,x)*2.0 + icomp_heat3d(z,y,x+1))*0.125);
  
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------    

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------    
    //buffers
    buffer b_out("b_out",{34,34,34},p_float64,a_output);
    
    //Store inputs
    icomp_heat3d.store_in(&b_out);


    //Store computations  
    comp_heat3d.store_in(&b_out,{z,y,x});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_out}, "function_heat3d_SMALL.o", "./function_heat3d_SMALL_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, model_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function_heat3d_SMALL_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete model_eval;
	delete bs;
	return 0;
}
