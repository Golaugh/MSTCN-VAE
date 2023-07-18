#!/usr/bin/env python
import argparse
import os
import csv
import sys
import random
import copy
import io
import yaml

# torchlight
import torchlight
from torchlight import import_class

# import ST-GCN model
from net.st_gcn import Model

# define function to optimize hyperparameters
def optimize_hyperparams(hyperparams, structparams):
    # create new train.yaml file with hyperparameters set to values passed in
    with open('config/mstcn_vae/imigue/train_rawdata.yaml', 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_config['weight_decay'] = hyperparams['weight_decay']
    train_config['base_lr'] = hyperparams['base_lr']
    train_config['step'] = hyperparams['step']
    train_config['model_args']['dilations'] = [i+1 for i in range(structparams['len_dilations'])]
    train_config['model_args']['branch_channels'] = structparams['branch_channels']

    num_branches = structparams['len_dilations'] + 2
    train_config['model_args']['num_output'] = num_branches * structparams['branch_channels']

    with open('config/mstcn_vae/imigue/train_new.yaml', 'w') as f:
        yaml.dump(train_config, f, default_flow_style=None)

    # train model using new train.yaml file
    Processor = import_class('processor.recognition.REC_Processor')
    p = Processor(['--config', 'config/mstcn_vae/imigue/train_new.yaml'])
    outcome = p.start()

    # return cross-validation accuracy
    return outcome

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['demo_old'] = import_class('processor.demo_old.Demo')
    processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    # endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # perform random search to optimize hyperparameters
    search_space = {
        'weight_decay': [0.00001, 0.0001, 0.001],
        'base_lr': [0.0001, 0.001, 0.01],
        'step': [[250, 300], [300, 350], [350, 400]]
    }
    struct_space = {
        'len_dilations': [3, 4, 5, 6],
        'branch_channels': [8, 16, 32, 64]
    }

    struct_trials = 2
    num_trials = 1
    best1_hyperparams = None
    best1_strucparams = None
    best5_hyperparams = None
    best5_strucparams = None
    best1_acc = -1
    best5_acc = -1
    count = 0
    with open('config/mstcn_vae/imigue/train_rawdata.yaml', 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
        num_epoch = train_config['num_epoch']

    for j in range(struct_trials):
        structparams = {
            'struct_id': j,
            'len_dilations': random.choice(struct_space['len_dilations']),
            'branch_channels': random.choice(struct_space['branch_channels'])
        }

        # Open a CSV file to write the results
        with open('results.csv', 'a', newline='') as R_csvfile:
            writer = csv.writer(R_csvfile)
            if os.stat('results.csv').st_size == 0:  # check if file is empty
                writer.writerow(['struct_id', 'weight_decay', 'base_lr', 'step1', 'step2', 'Top1acc', 'Top5acc'])

            for i in range(num_trials):
                hyperparams = {
                    'weight_decay': random.choice(search_space['weight_decay']),
                    'base_lr': random.choice(search_space['base_lr']),
                    'step': random.choice(search_space['step'])
                }
                outcome = optimize_hyperparams(hyperparams, structparams)
                top1acc = round(outcome[1], 4)
                top5acc = round(outcome[5], 4)
                print('\n\tTrial {}/{}: {} -> Top1: {}%, Top5: {}%\n'
                      .format(i + 1, num_trials, hyperparams, top1acc * 100, top5acc * 100))

                count += 1
                print('\n\tTotal finished -> {}/{}\n'
                      .format(count, num_trials * struct_trials))

                if top1acc > best1_acc:
                    best1_hyperparams = copy.deepcopy(hyperparams)
                    best1_strucparams = copy.deepcopy(structparams)
                    best1_acc = top1acc
                if top5acc > best5_acc:
                    best5_hyperparams = copy.deepcopy(hyperparams)
                    best5_strucparams = copy.deepcopy(structparams)
                    best5_acc = top5acc

                writer.writerow([j, hyperparams['weight_decay'], hyperparams['base_lr'],
                                 hyperparams['step'][0], hyperparams['step'][1], top1acc, top5acc])

        with open('struct.csv', 'a', newline='') as S_csvfile:
            writer = csv.writer(S_csvfile)
            if os.stat('struct.csv').st_size == 0:  # check if file is empty
                writer.writerow(['struct_id', 'len_dilations', 'branch_channels'])
            writer.writerow([j, structparams['len_dilations'], structparams['branch_channels']])

        # save best hyperparameters and their accuracy to a CSV file
        with open('best_hyperparams.csv', 'a', newline='') as B_csvfile:
            writer = csv.writer(B_csvfile)
            if os.stat('best_hyperparams.csv').st_size == 0:  # check if file is empty
                writer.writerow(['Top1_hyperparams', 'Top1_strucparams', 'Top1_acc',
                                 'Top5_hyperparams', 'Top1_strucparams', 'Top5_acc'])
            writer.writerow([best1_hyperparams, best1_strucparams, best1_acc,
                             best5_hyperparams, best5_strucparams, best5_acc])

    print('\tBest Top1 hyperparameters: {} + {} -> {}%'.
          format(best1_strucparams, best1_hyperparams, best1_acc * 100))
    print('\tBest Top5 hyperparameters: {} + {} -> {}%'.
          format(best5_strucparams, best5_hyperparams, best5_acc * 100))
