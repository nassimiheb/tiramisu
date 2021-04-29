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

    //constant width("width", 120), height("height", HEIGHT), channels("channels", 3);
    var xi("xi", 0, 1024), yi("yi", 0, 1024);
    var x("x", 1, 1023), y("y", 1, 1023), c("c", 0, 3);

    //inputs
    input input_img("input_img", {c, yi, xi}, p_float64);

    //Computations

    computation blur("blur", {c, y, x},p_float64 );
    blur.set_expression( (input_img(c, y + 1, x - 1) + input_img(c, y + 1, x) + input_img(c, y + 1, x + 1) + 
                                         input_img(c, y, x - 1)     + input_img(c, y, x)     + input_img(c, y, x + 1) + 
                                         input_img(c, y - 1, x - 1) + input_img(c, y - 1, x) + input_img(c, y - 1, x + 1))/9.0);
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Buffers
    buffer input_buf("input_buf", {3, 1024, 1024}, p_float64, a_input);
    buffer output_buf("output_buf", {3,1024, 1024}, p_float64, a_output);

    //Store inputs
    input_img.store_in(&input_buf);

    //Store computations
    blur.store_in(&output_buf);
 
 
    

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
           &input_buf, &output_buf
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
    auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({ &input_buf, &output_buf}, 
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
