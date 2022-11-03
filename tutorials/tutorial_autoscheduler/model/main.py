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
# model_path = (
#     "/data/scratch/hbenyamina/best_model_bidirectional_new_data_static_input_nn_a0b1.pt"
# )
model_path = '/data/mk8958/new_model/weights/best_model_bidirectional_static_input_nn_3326.pt'
with open("/data/mk8958/tiramisu/tutorials/tutorial_autoscheduler/model/logs.txt", "w") as f:
    f.write("Hello Wod")

MAX_DEPTH = 5

with torch.no_grad():
    device = "cpu"
    torch.device("cpu")

    environ["layers"] = "600 350 200 180"
    environ["dropouts"] = "0.225 " * 4

    input_size = 912
    output_size = 1

    layers_sizes = list(map(int, environ.get("layers", "300 200 120 80 30").split()))
    drops = list(map(float, environ.get("dropouts", "0.2 0.2 0.1 0.1 0.1").split()))

    model = Model_Recursive_LSTM_v2(input_size, drops=[0.050, 0.05, 0.05, 0.05, 0.05])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        try:
            while True:
                prog_json = input()
                sched_json = input()

                program_json = json.loads(prog_json)
                sched_json = json.loads(sched_json)
                (
                    prog_tree,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                    comps_expr_tensor,
                ) = get_sched_rep(program_json, sched_json, max_depth=MAX_DEPTH)
                computations_tensor, loops_tensor = get_schedule_representation(
                    program_json,
                    sched_json,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                    max_depth=MAX_DEPTH,
                )
                tree_tensor = (prog_tree, computations_tensor, loops_tensor, comps_expr_tensor)

                speedup = model.forward(tree_tensor)
                print(float(speedup.item()))
        except EOFError:
            exit()
