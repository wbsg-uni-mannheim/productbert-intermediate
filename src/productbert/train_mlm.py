import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import TrainerMLM

import transformers

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config['optimizer']['args']['weight_decay']
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0}
    ]

    optimizer = config.init_obj('optimizer', transformers, optimizer_grouped_parameters)

    num_training_steps_per_epoch = np.ceil(data_loader.n_samples / (data_loader.batch_size * config['trainer']['accumulation_steps']))
    epochs = config['trainer']['epochs']
    if 'num_training_steps' in config['lr_scheduler']['args'] and config['lr_scheduler']['args']['num_training_steps'] == -1:
        config['lr_scheduler']['args']['num_training_steps'] = num_training_steps_per_epoch * epochs
    if config.config['lr_scheduler']['args'].get('num_warmup_steps') == -1:
        config['lr_scheduler']['args']['num_warmup_steps'] = num_training_steps_per_epoch
    try:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    except AttributeError:
        lr_scheduler = config.init_obj('lr_scheduler', transformers, optimizer)

    trainer = TrainerMLM(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
