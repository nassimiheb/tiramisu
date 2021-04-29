#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

// Set to true to perform autoscheduling


// Path to python (please give absolute path)
const std::string py_cmd_path = "/usr/bin/python3";

// Path to a script that executes the ML model (please give absolute path)
const std::string py_interface_path = "/data/scratch/tiramisu_git/tiramisu_lanka/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";

int main(int argc, char **argv)
{
    tiramisu::init("conv");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    //constant width("width", 1024), height("height", 1024), channels("channels", 3);
    var x("x", 0, 1024), y("y", 0, 1024), c("c", 0, 3);

    //inputs
    input input_img("input_img", {c, y, x}, p_int32);

    //Computations

    //RGB2Gray[y,x] <- (input_img[2,y,x]*1868 + input_img[1,y,x]*9617 + input_img[0,y,x]*4899 + 8192) / 16384
    computation RGB2Gray("RGB2Gray", {y, x}, cast(p_int32, (input_img(2, y, x) * 1868 +  input_img(1, y, x) * 9617 + input_img(0, y, x) * 4899 + 8192) / 16384) );
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Buffers
    buffer input_buf("input_buf", {3, 1024, 1024}, p_int32, a_input);
    buffer RGB2Gray_buf("RGB2Gray_buf", {1024, 1024}, p_int32, a_output);

    //Store inputs
    input_img.store_in(&input_buf);

    //Store computations
    RGB2Gray.store_in(&RGB2Gray_buf);
 
    

    prepare_schedules_for_legality_checks();
    performe_full_dependency_analysis();

    bool perform_autoscheduling = false;

    perform_autoscheduling=true;
    
    // Generate a program with no schedule
    if (!perform_autoscheduling)
    {
        //conv.skew(t,x,2,1,t1,x0);
        //conv.tile(x0,y,64,32,x1,y1,x2,y2);
        //conv.parallelize(x1);

        tiramisu::codegen({
           &input_buf, &RGB2Gray_buf
        }, "function.o");
       
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
    auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({ &input_buf, &RGB2Gray_buf}, 
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
