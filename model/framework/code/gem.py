# imports
import os
import csv
import sys


import numpy as np
import multiprocessing
multiprocessing.set_start_method('fork') 

import paddle

from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.datasets import *

from GEM.src.model import DownstreamModel
from GEM.src.featurizer import DownstreamTransformFn, DownstreamCollateFn




# current file directory
root = os.path.dirname(os.path.abspath(__file__))
two_dir_up =  (os.path.dirname(os.path.dirname(root)))
compound_encoder_path = "checkpoints/model_configs/geognn_l8.json"
compound_encoder_dir = os.path.join(two_dir_up, compound_encoder_path)
model_config_path = "checkpoints/model_configs/down_mlp3.json"
model_config_dir = os.path.join(two_dir_up, model_config_path)
model_params_path = "checkpoints/pretrain_models-chemrl_gem/class.pdparams"
model_params_dir = os.path.join(two_dir_up, model_params_path)




def get_gem_pred(smiles_list):
    task_names = get_default_tox21_task_names()  
    task_type = "regr"  #task_type can be "class" or "regr". 

    compound_encoder_config = load_json_config(compound_encoder_dir)
    model_config = load_json_config(model_config_dir)
    model_config['num_tasks'] = len(task_names)
    model_config['task_type'] = task_type
    output = []


    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder)


    model.set_state_dict(paddle.load(model_params_dir))

    transform_fn = DownstreamTransformFn(is_inference=True)
    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type = task_type,
            is_inference=True)
    for smiles in smiles_list:
        tasks_list = []
        graph1, graph2 = collate_fn([transform_fn({'smiles': smiles})])
        preds = model(graph1.tensor(), graph2.tensor()).numpy()[0]
        for name, prob in zip(task_names, preds):
            tasks_list.append("%f" % (prob))
        output.append(tasks_list)
    return output


