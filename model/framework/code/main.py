# imports
import os
import csv
import sys
from rdkit import Chem
import os
from os.path import join, exists, basename
import argparse
import numpy as np
import multiprocessing
multiprocessing.set_start_method('fork') 

import paddle
import paddle.nn as nn
import pgl

from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.datasets import *

from GEM.src.model import DownstreamModel
from GEM.src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from GEM.src.utils import get_dataset, create_splitter, get_downstream_task_names, \
        calc_rocauc_score, exempt_parameters

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]


# current file directory
root = os.path.dirname(os.path.abspath(__file__))
two_dir_up =  (os.path.dirname(os.path.dirname(root))).resolve()
compound_encoder_path = "checkpoints/model_configs/geognn_l8.json"
compound_encoder_dir = os.path.join(two_dir_up, compound_encoder_path)
model_config_path = "checkpoints/model_configs/down_mlp3.json"
model_config_dir = os.path.join(two_dir_up, model_config_path)
model_params_path = "pretrain_models-chemrl_gem/class.pdparams"
model_params_dir = os.path.join(two_dir_up, model_params_path)




with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header.
    smiles_list = [r[1] for r in reader]


#inputs for the model are the task names, the task_type, and the number of tasks. 

# my model
def my_model(smiles_list):
    task_names = get_default_toxcast_task_names  #edit this based on what we want output tasks to be.
    print(task_names)

#task_type = "class" or "regr". 

    print(smiles_list)

    compound_encoder_config = load_json_config(compound_encoder_dir)
    model_config = load_json_config(model_config_dir)
    model_config['num_tasks'] = len(task_names)
    model_config['task_type'] = "class"
    output = []
#


    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder)
# criterion = nn.BCELoss(reduction='none')
# opt = paddle.optimizer.Adam(0.001, parameters=model.parameters())

    model.set_state_dict(paddle.load(model_params_dir))

#SMILES="CCCCC(CC)COC(=O)c1ccc(C(=O)OCC(CC)CCCC)c(C(=O)OCC(CC)CCCC)c1"
    transform_fn = DownstreamTransformFn(is_inference=True)
    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type = "class",
            is_inference=True)
    for smiles in smiles_list:
        graph1, graph2 = collate_fn([transform_fn({'smiles': smiles})])
        preds = model(graph1.tensor(), graph2.tensor()).numpy()[0]
        print('SMILES:%s' % smiles)
        print('Predictions:')  #add another for loop here to account for each task. For each task, write task name and then the preds.
        for name, prob in zip(task_names, preds):
            output.append("  %s:\t%s" % (name, prob))
            print("  %s:\t%s" % (name, prob))
    
    return output

# run model
outputs = my_model(smiles_list)

# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["value"])  # header
    for o in outputs:
        writer.writerow([o])
