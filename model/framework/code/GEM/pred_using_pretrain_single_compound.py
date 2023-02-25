import os
import numpy as np
import paddle
import paddle.nn as nn
import pgl

import sys
sys.path.insert(0, '/Users/karthik/eos1bba/model/framework/code')

import csv


from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.utils.splitters import \
    RandomSplitter, IndexSplitter, ScaffoldSplitter
from pahelix.datasets import *
from src.model import DownstreamModel
from src.featurizer import DownstreamTransformFn, DownstreamCollateFn

task_names = get_default_bbbp_task_names()
print(task_names)

with open("/Users/karthik/Downloads/eml_canonical.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header.
    smiles_list = [r[1] for r in reader]

print(smiles_list)

compound_encoder_config = load_json_config("/Users/karthik/eos1bba/model/checkpoints/model_configs/geognn_l8.json")
model_config = load_json_config("/Users/karthik/eos1bba/model/checkpoints/model_configs/down_mlp3.json")
model_config['num_tasks'] = len(task_names)
model_config['task_type'] = "class"
#


compound_encoder = GeoGNNModel(compound_encoder_config)
model = DownstreamModel(model_config, compound_encoder)
# criterion = nn.BCELoss(reduction='none')
# opt = paddle.optimizer.Adam(0.001, parameters=model.parameters())

model.set_state_dict(paddle.load('/Users/karthik/eos1bba/model/checkpoints/pretrain_models-chemrl_gem/class.pdparams')) #edit this

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
    print('Predictions:')
    for name, prob in zip(task_names, preds):
        print("  %s:\t%s" % (name, prob))
