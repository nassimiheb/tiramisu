#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

// Set to true to perform autoscheduling


// Path to python (please give absolute path)
const std::string py_cmd_path = "/usr/bin/python3";

// Path to a script that executes the ML model (please give absolute path)
const std::string py_interface_path = "/home/nassim/Desktop/tiramisu_raw/tutorials/tutorial_autoscheduler/model/main.py";

int main(int argc, char **argv)
{
    tiramisu::init("conv");

    var i0("i0", 0, 192), i1("i1", 0, 64), i2("i2", 0, 128), i3("i3", 0, 128);
    input icomp00("icomp00", {i1,i2,i3}, p_float64);
    input input01("input01", {i1}, p_float64);
    input input02("input02", {i1,i2}, p_float64);
    computation comp00("comp00", {i0,i1,i2,i3},  p_float64);
    comp00.set_expression(icomp00(i1, i2, i3) + input01(i1) + input02(i1, i2));
    buffer buf00("buf00", {64,128,128}, p_float64, a_output);
    buffer buf01("buf01", {64}, p_float64, a_temporary);
    buffer buf02("buf02", {64,128}, p_float64, a_temporary);
    icomp00.store_in(&buf00);
    input01.store_in(&buf01);
    input02.store_in(&buf02);
    comp00.store_in(&buf00, {i1,i2,i3});

    prepare_schedules_for_legality_checks();
    performe_full_dependency_analysis();

//    bool perform_autoscheduling = false;
//
//    perform_autoscheduling=true;
//
//    // Generate a program with no schedule
//    if (!perform_autoscheduling)
//    {
//        //conv.skew(t,x,2,1,t1,x0);
//        //conv.tile(x0,y,64,32,x1,y1,x2,y2);
//        //conv.parallelize(x1);
//
//        tiramisu::codegen({
//            &buf_output
//        }, "function.o");
//
//        return 0;
//    }
//
//    // Some parameters for the search methods
//    const int beam_size = 4;
//    const int max_depth = 6;
//
//    const int nb_samples = 5;
//    const int topk = 1;
//
//    // An object used by search methods to generate schedules
//    auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
//
//    // An evaluation function that measures execution time by compiling and executing the program
//    auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf_output},
//                                      "function.o", "./wrapper");
//
//    // An evaluation function that uses an ML model to estimate speedup
//    auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
//
//    // Two search methods : Beam Search and MCTS
//    auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, model_eval, scheds_gen);
//    auto_scheduler::mcts *mcts = new auto_scheduler::mcts(nb_samples, topk, max_depth, model_eval, exec_eval, scheds_gen);
//
//    // Create the autoscheduler and start search
//    auto_scheduler::auto_scheduler as(bs, exec_eval);
//    as.set_exec_evaluator(exec_eval);
//    as.find_schedule();
//    as.apply_best_schedule();
//
//    delete scheds_gen;
//    delete exec_eval;
//    delete bs;
//    delete mcts;
//
//    return 0;



    const int beam_size = 1;
    const int max_depth = 1;


    auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
    auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00}, "function.o", "./wrapper");
    auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
    auto_scheduler::auto_scheduler as(bs, exec_eval);
    as.set_exec_evaluator(exec_eval);
    as.sample_search_space("./tutorial_explored_schedules3.json", true);
    delete scheds_gen;
    delete exec_eval;
    delete bs;
    return 0;
}
