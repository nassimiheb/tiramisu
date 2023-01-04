import warnings
import logging
import numpy as np
from os import environ
import sys
import json

from hier_lstm import Model_Recursive_LSTM_v2
from json_to_tensor import *
import random
import time

import os

environ["MKL_THREADING_LAYER"] = "GNU"
logging.basicConfig(filename="log_new_model.txt", filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# model_path = '/data/scratch/mmerouani/tiramisu2/tiramisu/tutorials/tutorial_autoscheduler/model/multi_model_all_data_12+4+1.3.pkl'
# model_path = '/data/scratch/hbenyamina/bidirection_with_final_matrix.pt'
# model_path = '/data/scratch/mmerouani/tiramisu2/tiramisu/tutorials/tutorial_autoscheduler/model/best_model_bidirectional_new_data_fixed_inversed_matrices_98c0.pt'
# model_path = '/data/scratch/mmerouani/tiramisu2/tiramisu/tutorials/tutorial_autoscheduler/model/MAPE_base_13+4+2.6_22.7.pkl'
# model_path = '/data/scratch/hbenyamina/best_model_bidirectional_new_data_static_input_paper_4cb2.pt'
model_path = (
    "/home/islem/pfe/tiramisu_work/tiramisu/tutorials/tutorial_autoscheduler/model/best_model_bidirectional_static_input_nn_bd67.pt"
)

MAX_DEPTH = 5

def seperate_vector(
    X: torch.Tensor, num_matrices: int = 4, pad: bool = True, pad_amount: int = 5
) -> torch.Tensor:
    batch_size, _ = X.shape
    first_part = X[:, :33]
    second_part = X[:, 33 : 33 + 169 * num_matrices]
    third_part = X[:, 33 + 169 * num_matrices :]
    vectors = []
    for i in range(num_matrices):
        vector = second_part[:, 169 * i : 169 * (i + 1)].reshape(batch_size, 1, -1)
        vectors.append(vector)

    if pad:
        for i in range(pad_amount):
            vector = torch.zeros_like(vector)
            vectors.append(vector)
    return (first_part, vectors[0], torch.cat(vectors[1:], dim=1), third_part)

with torch.no_grad():
    device = "cpu"
    torch.device("cpu")

    environ["layers"] = "600 350 200 180"
    environ["dropouts"] = "0.05 " * 4
    logging.info("got here")
    input_size = 1056
    output_size = 1

    layers_sizes = list(map(int, environ.get("layers", "600 350 200 180").split()))
    drops = list(map(float, environ.get("dropouts", "0.05 0.05 0.05 0.05 0.05").split()))

    model = Model_Recursive_LSTM_v2(
        input_size=input_size,
        comp_embed_layer_sizes=layers_sizes,
        drops=drops,
        transformation_matrix_dimension=13,
        loops_tensor_size=33,
        expr_embed_size=100,
        train_device=device,
        bidirectional=True,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    model.to(device)
    model.eval()

    with torch.no_grad():
        try:
            while True:
                
                
                prog_json = input()
                no_sched_json = input()
                sched_json = input()
                # prog_json = """
                # {"memory_size" : "0.088989" ,"iterators" : {"jmean_init0" : {"lower_bound" : 0,"upper_bound" : 28,"parent_iterator" : null,"child_iterators" : [],"computations_list" : ["mean_init"]},"lmean0" : {"lower_bound" : 0,"upper_bound" : 32,"parent_iterator" : null,"child_iterators" : ["jmean1"],"computations_list" : []},"jmean1" : {"lower_bound" : 0,"upper_bound" : 28,"parent_iterator" : "lmean0","child_iterators" : [],"computations_list" : ["mean"]},"iconv_init0" : {"lower_bound" : 0,"upper_bound" : 28,"parent_iterator" : null,"child_iterators" : ["jconv_init1"],"computations_list" : []},"jconv_init1" : {"lower_bound" : 0,"upper_bound" : 28,"parent_iterator" : "iconv_init0","child_iterators" : [],"computations_list" : ["conv_init"]},"icov0" : {"lower_bound" : 0,"upper_bound" : 28,"parent_iterator" : null,"child_iterators" : ["jcov1"],"computations_list" : []},"jcov1" : {"lower_bound" : 0,"upper_bound" : 28,"parent_iterator" : "icov0","child_iterators" : ["kcov2"],"computations_list" : []},"kcov2" : {"lower_bound" : 0,"upper_bound" : 32,"parent_iterator" : "jcov1","child_iterators" : [],"computations_list" : ["cov"]}},"computations" : {"mean_init" : {"absolute_order" : 1,"iterators" : ["jmean_init0"],"comp_is_reduction" : false,"number_of_additions" : 0,"number_of_subtraction" : 0,"number_of_multiplication" : 0,"number_of_division" : 0,"write_access_relation" : "{ mean_init[j] -> b_mean[j' = j] }","write_buffer_id" : 1,"data_type" : "float64","data_type_size" : 0,"accesses" : [],"expression_representation" : {"expr_type" :  "value",  "children" : []}},"mean" : {"absolute_order" : 2,"iterators" : ["lmean0","jmean1"],"comp_is_reduction" : true,"number_of_additions" : 1,"number_of_subtraction" : 0,"number_of_multiplication" : 0,"number_of_division" : 1,"write_access_relation" : "{ mean[l, j] -> b_mean[j] }","write_buffer_id" : 1,"data_type" : "float64","data_type_size" : 0,"accesses" : [{"access_is_reduction" : true,"buffer_id" : 1,"access_matrix" : [[1, 0, 0],[0, 1, 0]]},{"access_is_reduction" : false,"buffer_id" : 0,"access_matrix" : [[1, 0, 0],[0, 1, 0]]}],"expression_representation" : {"expr_type" :  "add",  "children" : [{"expr_type" :  "access",  "children" : []},{"expr_type" :  "div",  "children" : [{"expr_type" :  "access",  "children" : []},{"expr_type" :  "cast",  "children" : [{"expr_type" :  "value",  "children" : []}]}]}]}},"conv_init" : {"absolute_order" : 3,"iterators" : ["iconv_init0","jconv_init1"],"comp_is_reduction" : false,"number_of_additions" : 0,"number_of_subtraction" : 0,"number_of_multiplication" : 0,"number_of_division" : 0,"write_access_relation" : "{ conv_init[i, j] -> b_cov[i' = i, j' = j] }","write_buffer_id" : 2,"data_type" : "float64","data_type_size" : 0,"accesses" : [],"expression_representation" : {"expr_type" :  "value",  "children" : []}},"cov" : {"absolute_order" : 4,"iterators" : ["icov0","jcov1","kcov2"],"comp_is_reduction" : true,"number_of_additions" : 1,"number_of_subtraction" : 2,"number_of_multiplication" : 1,"number_of_division" : 1,"write_access_relation" : "{ cov[i, j, k] -> b_cov[i, j] }","write_buffer_id" : 2,"data_type" : "float64","data_type_size" : 0,"accesses" : [{"access_is_reduction" : true,"buffer_id" : 2,"access_matrix" : [[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]]},{"access_is_reduction" : false,"buffer_id" : 0,"access_matrix" : [[0, 0, 1, 0],[1, 0, 0, 0]]},{"access_is_reduction" : false,"buffer_id" : 1,"access_matrix" : [[0, 0, 0, 0],[1, 0, 0, 0]]},{"access_is_reduction" : false,"buffer_id" : 0,"access_matrix" : [[0, 0, 1, 0],[0, 1, 0, 0]]},{"access_is_reduction" : false,"buffer_id" : 1,"access_matrix" : [[0, 0, 0, 0],[0, 1, 0, 0]]}],"expression_representation" : {"expr_type" :  "add",  "children" : [{"expr_type" :  "access",  "children" : []},{"expr_type" :  "div",  "children" : [{"expr_type" :  "mul",  "children" : [{"expr_type" :  "sub",  "children" : [{"expr_type" :  "access",  "children" : []},{"expr_type" :  "access",  "children" : []}]},{"expr_type" :  "sub",  "children" : [{"expr_type" :  "access",  "children" : []},{"expr_type" :  "access",  "children" : []}]}]},{"expr_type" :  "cast",  "children" : [{"expr_type" :  "value",  "children" : []}]}]}]}}}}
                # """
                # no_sched_json = """
                # {"mean_init" : {"shiftings" : null,"interchange_dims" : [],"transformation_matrices" : [],"transformation_matrix" : [],"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "skewing" : null}, "mean" : {"shiftings" : null,"interchange_dims" : [],"transformation_matrices" : [],"transformation_matrix" : [],"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "skewing" : null}, "conv_init" : {"shiftings" : null,"interchange_dims" : [],"transformation_matrices" : [],"transformation_matrix" : [],"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "skewing" : null}, "cov" : {"shiftings" : null,"interchange_dims" : [],"transformation_matrices" : [],"transformation_matrix" : [],"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "skewing" : null}, "fusions" : null, "sched_str": "", "tree_structure": {"roots" : [{"loop_name" : "jmean_init0","computations_list" : ["mean_init"],"child_list" : []},{"loop_name" : "lmean0","computations_list" : [],"child_list" : [{"loop_name" : "jmean1","computations_list" : ["mean"],"child_list" : []}]},{"loop_name" : "iconv_init0","computations_list" : [],"child_list" : [{"loop_name" : "jconv_init1","computations_list" : ["conv_init"],"child_list" : []}]},{"loop_name" : "icov0","computations_list" : [],"child_list" : [{"loop_name" : "jcov1","computations_list" : [],"child_list" : [{"loop_name" : "kcov2","computations_list" : ["cov"],"child_list" : []}]}]}]}}
                # """
                # sched_json = """
                # {"mean_init" : {"shiftings" : null,"interchange_dims" : [],"transformation_matrices" : [],"transformation_matrix" : [],"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "skewing" : null}, "mean" : {"shiftings" : null,"interchange_dims" : [],"transformation_matrices" : [],"transformation_matrix" : [],"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "skewing" : null}, "conv_init" : {"shiftings" : null,"interchange_dims" : [],"transformation_matrices" : [],"transformation_matrix" : [],"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "skewing" : null}, "cov" : {"shiftings" : null,"interchange_dims" : [],"transformation_matrices" : [],"transformation_matrix" : [],"tiling" : {},"unrolling_factor" : null,"parallelized_dim" : null, "skewing" : null}, "fusions" : [ ["mean","conv_init",1]], "sched_str": "F({C1,C2},L1)", "tree_structure": {"roots" : [{"loop_name" : "jmean_init0","computations_list" : ["mean_init"],"child_list" : []},{"loop_name" : "lmean0","computations_list" : [],"child_list" : [{"loop_name" : "jmean1","computations_list" : ["mean","conv_init"],"child_list" : []}]},{"loop_name" : "icov0","computations_list" : [],"child_list" : [{"loop_name" : "jcov1","computations_list" : [],"child_list" : [{"loop_name" : "kcov2","computations_list" : ["cov"],"child_list" : []}]}]}]}}
                # """
                

                program_json = json.loads(prog_json)
                sched_json = json.loads(sched_json)
                no_sched_json = json.loads(no_sched_json)
                (
                    prog_tree,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                    comps_expr_tensor,
                    comps_expr_lengths,
                ) = get_representation_template(program_json, no_sched_json, MAX_DEPTH)
                comps_tensor, loops_tensor = get_schedule_representation(
                    program_json,
                    no_sched_json,
                    sched_json,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                    max_depth=5,
                )
                
                x = comps_tensor
                batch_size, num_comps, __dict__ = x.shape
                x = x.view(batch_size * num_comps, -1)
                (first_part, final_matrix, vectors, third_part) = seperate_vector(
                        x, num_matrices=5, pad=False
                    )
                x = torch.cat(
                    (
                        first_part,
                        third_part,
                        final_matrix.reshape(batch_size * num_comps, -1),
                    ),
                    dim=1,
                ).view(batch_size, num_comps, -1)
                
                tree_tensor = (prog_tree, x, vectors, loops_tensor, comps_expr_tensor, comps_expr_lengths)

                speedup = model.forward(tree_tensor)
                print(float(speedup.item()))
                # print(random.uniform(0.5, 2))
                # print(random.randint(0,3))
        except EOFError:
            exit()