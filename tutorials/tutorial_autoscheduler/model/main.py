from os import environ
import sys, json

from hier_lstm import Model_Recursive_LSTM_v2
from json_to_tensor import *

import os
environ['MKL_THREADING_LAYER'] = 'GNU'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) 
warnings.filterwarnings('ignore', category=UserWarning)

model_path = '/data/scratch/mmerouani/tiramisu2/tiramisu/tutorials/tutorial_autoscheduler/model/multi_model_all_data_model_with_24_val.pkl'

with torch.no_grad():
        device = 'cpu'
        torch.device('cpu')

        environ['layers'] = '600 350 200 180'
        environ['dropouts'] = '0.225 ' * 4

        input_size = 1266
        output_size = 1

        layers_sizes = list(map(int, environ.get('layers', '300 200 120 80 30').split()))
        drops = list(map(float, environ.get('dropouts', '0.2 0.2 0.1 0.1 0.1').split()))

        model = Model_Recursive_LSTM_v2(input_size,drops=[0.112, 0.112, 0.112, 0.112])
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)
        model.eval()

        try:
            while True:
                prog_json = input()
                sched_json = input()

                prog_json = json.loads(prog_json)
                sched_json = json.loads(sched_json)

                tree_tensor = get_representation(prog_json, sched_json)

                speedup = model.forward(tree_tensor)
                print(float(speedup.item()))

        except EOFError:
            exit()
        
