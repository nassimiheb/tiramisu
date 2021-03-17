#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "wrapper.h"

using namespace tiramisu;

bool perform_autoscheduling = true;

const std::string py_cmd_path = "/usr/bin/python";
const std::string py_interface_path = "/home/masci/tiramisu_new/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";


int main(int argc, char **argv)
{
     tiramisu::init("heat_3d");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i_f("i_f", 0, N), j_f("j_f", 0, N), k_f("k_f", 0, N);
    var t("t", 0, TSTEPS), i("i", 0, N-2), j("j", 0, N-2), k("k", 0, N-2);
    
    //inputs
    input A("A", {i_f, j_f, k_f}, p_float64);
    input B("B", {i_f, j_f, k_f}, p_float64);

    //Computations
    // computation B_out("B_out", {t,i,j,k}, (A(i+2, j+1, k+1) - A(i+1, j+1, k+1)*2 + A(i , j+1, k+1))*0.125
    //                                     + (A(i+1, j+2, k+1) - A(i+1, j+1, k+1)*2 + A(i+1, j , k+1))*0.125
    //                                     + (A(i+1, j+1, k+2) - A(i+1, j+1, k+1)*2 + A(i+1, j+1, k ))*0.125
    //                                     + A(i+1, j+1, k+1));

    // computation A_out("A_out", {t,i,j,k}, (B(i+2, j+1, k+1) - B(i+1, j+1, k+1)*2 + B(i , j+1, k+1))*0.125
    //                                     + (B(i+1, j+2, k+1) - B(i+1, j+1, k+1)*2 + B(i+1, j , k+1))*0.125
    //                                     + (B(i+1, j+1, k+2) - B(i+1, j+1, k+1)*2 + B(i+1, j+1, k ))*0.125
    //                                     + B(i+1, j+1, k+1));

    computation B_out("B_out", {t,i,j,k}, A(i, j, k));

    computation A_out("A_out", {t,i,j,k}, B(i, j, k));
    computation C_out("C_out", {t,i,j,k}, B(i, j, k)+1);
    computation D_out("D_out", {t,i,j,k}, B(i, j, k)+2);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    B_out.then(A_out, k)
            .then(C_out, i)
        .then(D_out, k);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {N,N,N}, p_float64, a_output);    
    buffer b_B("b_B", {N,N,N}, p_float64, a_output);    

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);

    //Store computations
    A_out.store_in(&b_A, {i,j,k});
    B_out.store_in(&b_B, {i,j,k});
    C_out.store_in(&b_B, {i,j,k});
    D_out.store_in(&b_B, {i,j,k});


    if (!perform_autoscheduling)
    {
        tiramisu::codegen({&b_A, &b_B}, "function.o");
        return 0;
    }
    
    // autoscheduling
    const int beam_size = 2;
    const int max_depth = 4;

    const int nb_samples = 5;
    const int topk = 1;
    
    const int Beam_Searchtopk = 5;
    
    auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
    
    auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution(
        {&b_A, &b_B},"function.o", "./build/wrapper");
    
    auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
    auto_scheduler::search_method *Beam_Searchk = new auto_scheduler::beam_search_topk(beam_size, Beam_Searchtopk, max_depth, model_eval,exec_eval, scheds_gen);
    auto_scheduler::search_method *Beam_Search = new auto_scheduler::beam_search(beam_size, max_depth, model_eval, scheds_gen);
    auto_scheduler::beam_search_accuracy_evaluator *BS_Acc = new auto_scheduler::beam_search_accuracy_evaluator(beam_size, max_depth, model_eval, exec_eval, scheds_gen);
    auto_scheduler::mcts *mcts = new auto_scheduler::mcts(nb_samples, topk, max_depth, model_eval, exec_eval, scheds_gen);
    
    
    //  auto_scheduler::auto_scheduler as(BS_Acc, model_eval);
    //  auto_scheduler::auto_scheduler as(Beam_Search, model_eval);
    auto_scheduler::auto_scheduler as(Beam_Search, exec_eval);
    
    
//     auto_scheduler::auto_scheduler as(Beam_Searchk, model_eval);
//     auto_scheduler::auto_scheduler as(mcts, model_eval);
    
    as.set_exec_evaluator(exec_eval);
    // as.find_schedule();
    as.sample_search_space();
    // as.apply_best_schedule();
    // BS_Acc->print_evals_list();
    
    delete scheds_gen;
    delete exec_eval;
    delete Beam_Search;
    delete mcts;
    
    return 0;
}
// 