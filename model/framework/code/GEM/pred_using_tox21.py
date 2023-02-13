import os
import numpy as np
import paddle
import paddle.nn as nn
import pgl

import sys
sys.path.insert(0, '/Users/karthik/eos1bba/model/framework/code')


from pahelix.model_zoo.pretrain_gnns_model import PretrainGNNModel, AttrmaskModel
from pahelix.datasets.zinc_dataset import load_zinc_dataset
from pahelix.utils.splitters import RandomSplitter
from pahelix.featurizers.pretrain_gnn_featurizer import AttrmaskTransformFn, AttrmaskCollateFn
from pahelix.utils import load_json_config

from pahelix.utils.splitters import \
    RandomSplitter, IndexSplitter, ScaffoldSplitter
from pahelix.datasets import *

from src.model import DownstreamModel
from src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from src.utils import calc_rocauc_score, exempt_parameters

task_names = get_default_tox21_task_names()
print(task_names)



compound_encoder_config = load_json_config("/Users/karthik/eos1bba/model/framework/code/GEM/model_configs/geognn_l8.json")
model_config = load_json_config("/Users/karthik/eos1bba/model/framework/code/GEM/model_configs/down_mlp3.json")
model_config['num_tasks'] = len(task_names)
model_config['task_type'] = "class"

# compound_encoder = PretrainGNNModel(compound_encoder_config)
# model = DownstreamModel(model_config, compound_encoder)
# criterion = nn.BCELoss(reduction='none')
# opt = paddle.optimizer.Adam(0.001, parameters=model.parameters())

compound_encoder = PretrainGNNModel(compound_encoder_config)
model = DownstreamModel(model_config, compound_encoder)

model.set_state_dict(paddle.load('/Users/karthik/eos1bba/model/framework/code/GEM/pretrain_models-chemrl_gem/regr.pdparams')) #edit this

SMILES="O=C1c2ccccc2C(=O)C1c1ccc2cc(S(=O)(=O)[O-])cc(S(=O)(=O)[O-])c2n1"
transform_fn = DownstreamTransformFn(is_inference=True)
collate_fn = DownstreamCollateFn(
        atom_names=compound_encoder_config['atom_names'], 
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'],
        bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
        task_type = "class",
        is_inference=True)
graph1, graph2 = collate_fn([transform_fn({'smiles': SMILES})])
print(graph1)
preds = model(graph1.tensor()).numpy()[0]
print('SMILES:%s' % SMILES)
print('Predictions:')
for name, prob in zip(task_names, preds):
    print("  %s:\t%s" % (name, prob))
