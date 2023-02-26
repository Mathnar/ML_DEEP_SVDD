import datetime
import random
import logging
from utils.utility import *
from deepSVDD import DeepSVDD
from datasets.main import load_dataset
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='cifar10_config.json', help='config file path (default: cifar10_config.json)')

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

dataset_name = config['dataset_name']
net_name = config['net_name']
xp_path = config['xp_path']
data_path = config['data_path']
load_model = config.get('load_model', None)
objective = config.get('type', 'one-class')
nu = config.get('v', 0.1)
device = config.get('device', 'cuda')
seed = config.get('seed', -1)
optimizer_name = config.get('optimizer', 'adam')
lr = config.get('lr', 0.001)
n_epochs = config.get('n_epochs', 50)
lr_milestone = config.get('lr_milestone', [0])
batch_size = config.get('batch_size', 128)
weight_decay = config.get('weight_decay', 1e-6)
pretrain = config.get('pretrain', True)
ae_optimizer_name = config.get('ae_optimizer', 'adam')
ae_lr = config.get('ae_lr', 0.001)
ae_n_epochs = config.get('ae_n_epochs', 100)
ae_lr_milestone = config.get('ae_lr_milestone', [0])
ae_batch_size = config.get('ae_batch_size', 128)
ae_weight_decay = config.get('ae_weight_decay', 1e-6)
normal_class = config.get('normal_class', 0)

def main():
    inliners = None
    outliners = None
    conf = Config(config)
    logger = start_logger()

    logger.info('Deep SVDD objective f: %s' % conf.settings['type'])
    logger.info('v: %.2f' % conf.settings['v'])

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'gpu'
    logger.info('Device: %s' % device)

    # _____________________ Load dataset
    dataset = load_dataset(dataset_name, data_path, normal_class)

    # _____________________ Init Deep_svdd
    deep_SVDD = DeepSVDD(conf.settings['type'], conf.settings['v'])
    deep_SVDD.build_network(net_name)

    if pretrain:
        logger.info('Pretraining: %s' % pretrain)
        pretrain_model(conf, dataset, deep_SVDD, device, logger)

    logger.info('Training optimizer: %s' % conf.settings['optimizer'])
    logger.info('Training learning rate: %g' % conf.settings['lr'])
    logger.info('Training epochs: %d' % conf.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (conf.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % conf.settings['batch_size'])
    logger.info('Training weight decay: %g' % conf.settings['weight_decay'])
    deep_SVDD.train(dataset,
                    optimizer=conf.settings['optimizer'],
                    lr=conf.settings['lr'],
                    n_epochs=conf.settings['n_epochs'],
                    lr_milestones=conf.settings['lr_milestone'],
                    batch_size=conf.settings['batch_size'],
                    weight_decay=conf.settings['weight_decay'],
                    device=device)

    deep_SVDD.test(dataset, device=device)

    # Plot
    indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

    extract_data_and_save(conf, dataset, deep_SVDD, idx_sorted, inliners, outliners)

    exit(0)


def extract_data_and_save(conf, dataset, deep_SVDD, idx_sorted, inliners, outliners):
    if dataset_name in ('mnist', 'cifar10'):
        if dataset_name == 'mnist':
            inliners = dataset.testing_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
            outliners = dataset.testing_set.test_data[idx_sorted[-32:], ...].unsqueeze(1)

        if dataset_name == 'cifar10':
            inliners = torch.tensor(np.transpose(dataset.testing_set.data[idx_sorted[:32], ...], (0, 3, 1, 2)))
            outliners = torch.tensor(np.transpose(dataset.testing_set.data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

        plot_imgs_tensor(inliners,
                         export_img=xp_path + '/'+objective + '_normals_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
                         title='Most normal imgs', padding=3)
        plot_imgs_tensor(outliners,
                         export_img=xp_path + '/'+objective + '_outliers_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
                         title='Most anomalous imgs', padding=3)
    # Save results
    deep_SVDD.save_results(
        export_json=xp_path + '/'+objective + '_results_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '.json')
    deep_SVDD.save_model(
        export_model=xp_path + '/'+objective + '_model_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '.tar')
    conf.save_config(
        export_json=xp_path + '/'+objective + '_'+dataset_name+'_config_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '.json')


def pretrain_model(conf, dataset, deep_SVDD, device, logger):
    logger.info('Pretraining- optimizer: %s' % conf.settings['optimizer'])
    logger.info('Pretraining- learning rate: %g' % conf.settings['ae_lr'])
    logger.info('Pretraining- epochs: %d' % conf.settings['ae_n_epochs'])
    logger.info('Pretraining- batch size: %d' % conf.settings['ae_batch_size'])
    logger.info('Pretraining- weight decay: %g' % conf.settings['ae_weight_decay'])
    deep_SVDD.pretrain(dataset,
                       ae_optimizer=conf.settings['ae_optimizer'],
                       lr=conf.settings['ae_lr'],
                       n_epochs=conf.settings['ae_n_epochs'],
                       lr_milestones=conf.settings['ae_lr_milestone'],
                       batch_size=conf.settings['ae_batch_size'],
                       weight_decay=conf.settings['ae_weight_decay'],
                       device=device)


def start_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Log file is %s.' % log_file)
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Network: %s' % net_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)
    return logger


if __name__ == '__main__':
    main()
