#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>


using namespace tiramisu;
const std::string py_cmd_path = "/usr/bin/python3";

// Path to a script that executes the ML model (please give absolute path)
const std::string py_interface_path = "/home/nassim/Desktop/tiramisu_raw/tutorials/tutorial_autoscheduler/model/main.py";


int main(int argc, char **argv)
{
    tiramisu::init("bench_function");
    
    // iterators
    int c00(770), c01(898), c02(1026);
    var i00("i00", 0, c00 - 2) , i01("i01", 0, c01 - 2) , i02("i02", 0, c02 - 2) ;
    
    // computations
    input input00("input00", {i00, i01, i02}, p_int32);

    computation comp02("comp02", {i00, i01, i02}, input00(i00, i01 + 1 ,i02 + 1) + 6 * input00(i00 + 1, i01 + 1 ,i02 + 1) -  input00(i00 + 2, i01 + 1 ,i02 + 1) + input00(i00 + 1, i01 ,i02 + 1) -  input00(i00 + 1, i01 + 2 ,i02 + 1) + input00(i00 + 1, i01 + 1, i02) -  input00(i00 + 1, i01 + 1 ,i02 + 2));
    
    // buffers
    buffer buf02("buf02", {768, 896, 1024}, p_int32, a_output);
    buffer buf00("buf00", {770, 898, 1026}, p_int32, a_input);

    comp02.store_in(&buf02);
    input00.store_in(&buf00);
    
    std::vector<tiramisu::buffer*> buffs_list = {&buf02, &buf00};
    
      prepare_schedules_for_legality_checks();
    performe_full_dependency_analysis();

    bool perform_autoscheduling = false;

    perform_autoscheduling=true;
    
    // Generate a program with no schedule
    if (!perform_autoscheduling)
    {
        tiramisu::codegen(buffs_list, "function.o");
       
        return 0;
    }

    // Some parameters for the search methods
    const int beam_size = 4;
    const int max_depth = 6;

    const int nb_samples = 5;
    const int topk = 1;
    
    // An object used by search methods to generate schedules
    auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
    
    // An evaluation function that measures execution time by compiling and executing the program
    auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution(buffs_list, 
                                      "function.o", "./wrapper");
    
    // An evaluation function that uses an ML model to estimate speedup
    auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
    
    // Two search methods : Beam Search and MCTS
    auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, model_eval, scheds_gen);
    auto_scheduler::mcts *mcts = new auto_scheduler::mcts(nb_samples, topk, max_depth, model_eval, exec_eval, scheds_gen);
    
    // Create the autoscheduler and start search
    auto_scheduler::auto_scheduler as(bs, exec_eval);
    as.set_exec_evaluator(exec_eval);
    as.find_schedule();
    as.apply_best_schedule();

    delete scheds_gen;
    delete exec_eval;
    delete bs;
    delete mcts;
    
    return 0;
}
