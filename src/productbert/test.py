import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

from utils import calculate_prec_rec_f1


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['batch_size'],
        config['data_loader']['args']['file'],
        shuffle=False,
        validation_split=0.0,
        num_workers=8
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    preds_acc = None
    if config.config.get('save_predictions'):
        preds_acc = []

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    total_tp_fp_tn_fn = {'tp':0,'fp':0,'tn':0,'fn':0}

    with torch.no_grad():
        for i, (data, token_ids, attn_mask, target) in enumerate(tqdm(data_loader)):
            data, token_ids, attn_mask, target = data.to(device), token_ids.to(device), attn_mask.to(device), target.to(device)
            output = model(data, token_ids, attn_mask)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            if config.config.get('pos_neg_ratio'):
                loss = loss_fn(output, target, config['pos_neg_ratio'])
            else:
                loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

            output = output.sigmoid()

            if preds_acc is not None:
                preds_acc.extend(list(output))

            for i, metric in enumerate(metric_fns):
                result = metric(output, target)
                if metric.__name__ in list(total_tp_fp_tn_fn.keys()):
                    total_tp_fp_tn_fn[metric.__name__] += result
                total_metrics[i] += result * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    if preds_acc is not None:
        preds_to_write = data_loader.dataset.data
        preds_to_write['predictions'] = preds_acc
        preds_to_write = preds_to_write.loc[:,('pair_id','label','predictions')]
        preds_to_write['label'] = preds_to_write['label'].apply(lambda x: x.item())
        preds_to_write['predictions'] = preds_to_write['predictions'].apply(lambda x: x.item())
        preds_to_write.to_pickle(f'{config.save_dir}/predictions.pkl.gz')
    precision, recall, f1 = calculate_prec_rec_f1(total_tp_fp_tn_fn)
    additional_log = {"tp": total_tp_fp_tn_fn['tp'], "fp": total_tp_fp_tn_fn['fp'], "tn": total_tp_fp_tn_fn['tn'], "fn": total_tp_fp_tn_fn['fn'], "precision": precision, "recall": recall, "f1": f1}
    log.update(additional_log)

    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
