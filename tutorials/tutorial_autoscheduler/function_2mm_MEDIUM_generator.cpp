#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_2mm_MEDIUM_wrapper.h"

using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("function_2mm_MEDIUM");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
     var i("i", 0, 128), j("j", 0, 320), k("k", 0, 256), l("l", 0, 192);

    //inputs
    input A("A", {i, k}, p_float64);
    input B("B", {k, l}, p_float64);
    input C("C", {l, j}, p_float64);
    input D("D", {i, j}, p_float64);
    
    //Computations
    computation D_init("D_init", {i,j}, D(i,j)*1.2);
    computation D_out("D_out", {i,j,k,l}, p_float64);
    D_out.set_expression(D(i,j) + A(i,k)*B(k,l)*C(l,j)*1.5);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
     D_init.then(D_out, i);
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_D("b_D", {128,320}, p_float64, a_output);
    buffer b_A("b_A", {128,256}, p_float64, a_input);
    buffer b_B("b_B", {256,192}, p_float64, a_input);
    buffer b_C("b_C", {192,320}, p_float64, a_input);

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    C.store_in(&b_C);
    D.store_in(&b_D);

    //Store computations
    D_init.store_in(&b_D);
    D_out.store_in(&b_D, {i,j});
   

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_B, &b_C, &b_D}, "function_2mm_MEDIUM.o");
    
    return 0;
}