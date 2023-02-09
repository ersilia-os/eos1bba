import os
from os.path import join, exists, basename
import sys
import argparse
import time
import numpy as np
from glob import glob
import logging

import paddle
import paddle.distributed as dist

from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.utils import load_json_config
from pahelix.featurizers.gem_featurizer import GeoPredTransformFn, GeoPredCollateFn
from pahelix.model_zoo.gem_model import GeoGNNModel, GeoPredModel
from src.utils import exempt_parameters

def main(args):
    """tbd"""
    compound_encoder_config = load_json_config(args.compound_encoder_config)  #this line encodes the configurations from the model. Change this to include the direct file path
    model_config = load_json_config(args.model_config)  #change this to include the direct file path. Then I can change parameters of the function as well. 
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate
        model_config['dropout_rate'] = args.dropout_rate

    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = GeoPredModel(model_config, compound_encoder)
    if args.distributed:
        model = paddle.DataParallel(model)
    opt = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    print('Total param num: %s' % (len(model.parameters())))
    for i, param in enumerate(model.named_parameters()):
        print(i, param[0], param[1].name)

    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))  #edit this to change args.init_model to be a directory path. 
        print('Load state_dict from %s' % args.init_model)  # edit this to load the finetuned model. 
    # get dataset
    dataset = load_smiles_to_dataset(args.data_path)
    if args.DEBUG:
        dataset = dataset[100:180]
    dataset = dataset[dist.get_rank()::dist.get_world_size()]
    smiles_lens = [len(smiles) for smiles in dataset]
    print('Total size:%s' % (len(dataset)))
    print('Dataset smiles min/max/avg length: %s/%s/%s' % (
            np.min(smiles_lens), np.max(smiles_lens), np.mean(smiles_lens)))
    transform_fn = GeoPredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])  # the GeoPred transform function and the colalte function uses MMFF 3D optimization. 
    # this step will be time consuming due to rdkit 3d calculation
    dataset.transform(transform_fn, num_workers=args.num_workers)
    test_index = int(len(dataset) * (1 - args.test_ratio))
    # train_dataset = dataset[:test_index]
    # test_dataset = dataset[test_index:]
    # print("Train/Test num: %s/%s" % (len(train_dataset), len(test_dataset)))

    collate_fn = GeoPredCollateFn(
            atom_names=compound_encoder_config['atom_names'],
            bond_names=compound_encoder_config['bond_names'], 
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            pretrain_tasks=model_config['pretrain_tasks'],
            mask_ratio=model_config['mask_ratio'],
            Cm_vocab=model_config['Cm_vocab'])
    # train_data_gen = train_dataset.get_data_loader(
    #         batch_size=args.batch_size, 
    #         num_workers=args.num_workers, 
    #         shuffle=True, 
    #         collate_fn=collate_fn)
    
    # list_test_loss = []
    # for epoch_id in range(args.max_epoch):
    #     s = time.time()
    #     train_loss = train(args, model, opt, train_data_gen)
    #     test_loss = evaluate(args, model, test_dataset, collate_fn)
    #     if not args.distributed or dist.get_rank() == 0:
    #         paddle.save(compound_encoder.state_dict(), 
    #             '%s/epoch%d.pdparams' % (args.model_dir, epoch_id))
    #         list_test_loss.append(test_loss['loss'])
    #         print("epoch:%d train/loss:%s" % (epoch_id, train_loss))
    #         print("epoch:%d test/loss:%s" % (epoch_id, test_loss))
    #         print("Time used:%ss" % (time.time() - s))
    
    # if not args.distributed or dist.get_rank() == 0:
    #     print('Best epoch id:%s' % np.argmin(list_test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", action='store_true', default=False)
    parser.add_argument("--distributed", action='store_true', default=False)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--dataset", type=str, default='zinc')
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    args = parser.parse_args()

    if args.distributed:
        dist.init_parallel_env()

    main(args)