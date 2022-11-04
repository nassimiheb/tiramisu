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
logging.basicConfig(filename="log_new_model.txt")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# model_path = '/data/scratch/mmerouani/tiramisu2/tiramisu/tutorials/tutorial_autoscheduler/model/multi_model_all_data_12+4+1.3.pkl'
# model_path = '/data/scratch/hbenyamina/bidirection_with_final_matrix.pt'
# model_path = '/data/scratch/mmerouani/tiramisu2/tiramisu/tutorials/tutorial_autoscheduler/model/best_model_bidirectional_new_data_fixed_inversed_matrices_98c0.pt'
# model_path = '/data/scratch/mmerouani/tiramisu2/tiramisu/tutorials/tutorial_autoscheduler/model/MAPE_base_13+4+2.6_22.7.pkl'
# model_path = '/data/scratch/hbenyamina/best_model_bidirectional_new_data_static_input_paper_4cb2.pt'
model_path = (
    "/home/afif/multi/tiramisu/tutorials/tutorial_autoscheduler/model/best_model_bidirectional_static_input_nn_bd67.pt"
)

MAX_DEPTH = 5

with torch.no_grad():
    device = "cpu"
    torch.device("cpu")

    environ["layers"] = "600 350 200 180"
    environ["dropouts"] = "0.05 " * 4

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
                sched_json = input()
                no_sched_json = input()

                program_json = json.loads(prog_json)
                sched_json = json.loads(sched_json)
                # no_sched_json = json.loads(no_sched_json)

                (
                    prog_tree,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                    comps_expr_repr_templates_list
                ) = get_representation_template(program_json, no_sched_json, MAX_DEPTH)
                comps_tensor, loops_tensor = get_schedule_representation(
                    program_json,
                    sched_json,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                    max_depth=5,
                )
                tree_tensor = (prog_tree, comps_tensor, loops_tensor)

                speedup = model.forward(tree_tensor)
                print(float(speedup.item()))
        except EOFError:
            exit()
