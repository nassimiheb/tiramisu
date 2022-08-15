from os import environ
import sys, json

from hier_lstm import Model_Recursive_LSTM_v2
from json_to_tensor import *

import os
environ['MKL_THREADING_LAYER'] = 'GNU'
# import logging
# logging.basicConfig(filename="log_old_model.txt")
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) 
warnings.filterwarnings('ignore', category=UserWarning)

model_path = '/data/scratch/mmerouani/tiramisu2/tiramisu/tutorials/tutorial_autoscheduler/model/best_model_bidirectional_new_data_fixed_inversed_matrices_98c0.pt'
MAX_DEPTH = 5
MAX_MATRICES = 8

with torch.no_grad():
        device = 'cpu'
        torch.device('cpu')

        environ['layers'] = '600 350 200 180'
        environ['dropouts'] = '0.225 ' * 4

        input_size = 776
        output_size = 1

        layers_sizes = list(map(int, environ.get('layers', '300 200 120 80 30').split()))
        drops = list(map(float, environ.get('dropouts', '0.2 0.2 0.1 0.1 0.1').split()))

        model = Model_Recursive_LSTM_v2(input_size, drops=[0.050, 0.05, 0.05, 0.05, 0.05])
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)
        model.eval()

        try:
            while True:
                prog_json = input()
                sched_json = input()

                
                program_json = json.loads(prog_json)
                sched_json = json.loads(sched_json)

                prog_tree, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict = get_sched_rep(program_json, sched_json, max_depth=MAX_DEPTH )
                computations_tensor, loops_tensor, factors = get_schedule_representation(program_json, sched_json, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict, max_depth=MAX_DEPTH )
                factors = factors.reshape(1,-1,(MAX_DEPTH+1)**2)
                zeros_to_add = torch.zeros((1,MAX_MATRICES-factors.shape[1], (MAX_DEPTH+1)**2))
                factors = torch.concat((factors, zeros_to_add), dim=1)
                tree_tensor = (prog_tree, computations_tensor, loops_tensor, factors )
                
                # logging.warning(computations_tensor.shape)
                # logging.warning(factors.shape)
                speedup = model.forward(tree_tensor)
                print(float(speedup.item()))

        except EOFError:
            exit()
        
