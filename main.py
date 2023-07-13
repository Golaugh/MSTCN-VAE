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
def optimize_hyperparams(hyperparams):
    # create new train.yaml file with hyperparameters set to values passed in
    with open('config/mstcn_vae/imigue/train.yaml', 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_config['weight_decay'] = hyperparams['weight_decay']
    train_config['base_lr'] = hyperparams['base_lr']
    train_config['step'] = hyperparams['step']
    with open('config/mstcn_vae/imigue/train_new.yaml', 'w') as f:
        yaml.dump(train_config, f)

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
        'step': [[5, 10], [10, 20], [20, 40]]
    }

    num_trials = 3
    best1_hyperparams = None
    best5_hyperparams = None
    best1_acc = -1
    best5_acc = -1

    # Open a CSV file to write the results
    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat('results.csv').st_size == 0:  # check if file is empty
            writer.writerow(['weight_decay', 'base_lr', 'step1', 'step2', 'Top1acc', 'Top5acc'])

        for i in range(num_trials):
            hyperparams = {
                'weight_decay': random.choice(search_space['weight_decay']),
                'base_lr': random.choice(search_space['base_lr']),
                'step': random.choice(search_space['step'])
            }
            outcome = optimize_hyperparams(hyperparams)
            top1acc = round(outcome[1],4)
            top5acc = round(outcome[5],4)
            print('\n\tTrial {}/{}: {} -> Top1: {}%, Top5: {}%\n'
                  .format(i + 1, num_trials, hyperparams, top1acc * 100, top5acc * 100))
            if top1acc > best1_acc:
                best1_hyperparams = copy.deepcopy(hyperparams)
                best1_acc = top1acc
            if top5acc > best5_acc:
                best5_hyperparams = copy.deepcopy(hyperparams)
                best5_acc = top5acc

            writer.writerow([hyperparams['weight_decay'], hyperparams['base_lr'],
                             hyperparams['step'][0], hyperparams['step'][1], top1acc, top5acc])

    # save best hyperparameters and their accuracy to a CSV file
    with open('best_hyperparams.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat('best_hyperparams.csv').st_size == 0:  # check if file is empty
            writer.writerow(['Top1_hyperparams', 'Top1_acc', 'Top5_hyperparams', 'Top5_acc'])
        writer.writerow([best1_hyperparams, best1_acc, best5_hyperparams, best5_acc])

    print('\tBest Top1 hyperparameters: {} -> {}%'.format(best1_hyperparams, best1_acc * 100))
    print('\tBest Top5 hyperparameters: {} -> {}%'.format(best5_hyperparams, best5_acc * 100))
