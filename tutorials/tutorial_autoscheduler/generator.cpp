#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include <vector>

using namespace tiramisu;

// Set to true to perform autoscheduling


// Path to python (please give absolute path)
const std::string py_cmd_path = "/usr/bin/python3";

// Path to a script that executes the ML model (please give absolute path)
const std::string py_interface_path = "/home/nassim/Desktop/tiramisu_raw/tutorials/tutorial_autoscheduler/model/main.py";

int main(int argc, char **argv)
{
    tiramisu::init("conv");

    var i0("i0", 0, 256), i1("i1", 1, 129), i2("i2", 0, 128), i3("i3", 0, 384), i4("i4", 0, 64), i1_p1("i1_p1", 0, 130);
    input input01("input01", {i1_p1}, p_float64);
    computation comp00("comp00", {i0,i1,i2}, 10.0);
    computation comp01("comp01", {i0,i1,i3,i4}, 20.0);
    comp00.then(comp01, i1);
    buffer buf00("buf00", {256,129,12}, p_float64, a_output);
    buffer buf01("buf01", {130}, p_float64, a_input);
    buffer buf02("buf02", {256,129,384}, p_float64, a_output);
    input01.store_in(&buf01);
    comp00.store_in(&buf00, {i0,i1,i2});
    comp01.store_in(&buf02, {i0,i1,i3});
//    B_out.interchange(1,2);
    // the code above is the initial unfused code since we used "B_out.then(A_out, t)" 
    // we want to dependency analysis to be performed on the original code correctly

    prepare_schedules_for_legality_checks(true);
    performe_full_dependency_analysis();
    // result stored in the function class


    // this is a major change in the program, as we fuse to the yy loop level
    //B_out.then(A_out,yy);

    // we must prepare schedules for our solvers like this, since we want applied changes on sched_graph to be reflected in schedules
    //prepare_schedules_for_legality_checks(true);

    bool perform_autoscheduling = false;

    // this is the fusion solver
    /*auto shiftings = global::get_implicit_function()->correcting_loop_fusion_with_shifting({&B_out},A_out,{t,xx,yy});

    assert(shiftings.size() > 0);// asserts that a fusion is possible (shiftings.size() == 0 means impossible fusion)

    // shift A_out loops which was the target computation to shift in the solver
    for(auto const& tup:shiftings)
    {
        A_out.shift(
            std::get<0>(tup),
            std::get<1>(tup)
            );
    }*/

    perform_autoscheduling= true;
    
    // Generate a program with no schedule
    if (!perform_autoscheduling)
    {
//        A_out.interchange(1,2);
//        global::get_implicit_function()->reset_schedules();
//        B_out.then(A_out, t);
//        A_out.interchange(1,2);
//
//        B_out.parallelize(xx);

//        B_out.then(A_out, 4);
//        A_out.shift(1,1);
//        A_out.interchange(0,1);
//        B_out.interchange(0,1);
//        A_out.tile(1,2,20,10);
//        A_out.tile(1,2,20,10);
//        A_out.tile(1,2,32,10);
//        A_out.skew(0,1,4,1);
//        B_out.skew(0,1,4,1);
//        A_out.tag_parallel_level(1);
//        B_out.tag_parallel_level(1);
//        A_out.tile(0,1,64,32);
//        B_out.tile(0,1,64,32);

        std::vector<tiramisu::computation*> comps = {&comp00,&comp01};
        tiramisu::block b(comps);
        b.tile(0,1,32,32);
//        comp00.tile(0,1,32,32);
//        comp01.tile(0,1,32,32);
        comp00.then(comp01, 3);

        tiramisu::codegen({&buf00,&buf01,&buf02}, "function.o");
       
        return 0;
    }

    // Some parameters for the search methods
    const int beam_size = 5;
    const int max_depth = 6;

    const int nb_samples = 5;
    const int topk = 1;
    
    // An object used by search methods to generate schedules
    auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
    
    // An evaluation function that measures execution time by compiling and executing the program
    auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00,&buf01,&buf02},
                                      "function.o", "./wrapper");
    
    // An evaluation function that uses an ML model to estimate speedup
    auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
    
    // Two search methods : Beam Search and MCTS
    auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, model_eval, scheds_gen);
//    auto_scheduler::mcts *mcts = new auto_scheduler::mcts(nb_samples, topk, max_depth, model_eval, exec_eval, scheds_gen);
    
    // Create the autoscheduler and start search
    auto_scheduler::auto_scheduler as(bs, exec_eval);
    as.set_exec_evaluator(exec_eval);
//    as.find_schedule();
    as.sample_search_space("test.json");
//    as.apply_best_schedule();

    delete scheds_gen;
    delete exec_eval;
    delete bs;
//    delete mcts;
    
    return 0;
}
