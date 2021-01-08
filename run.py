import os
import yaml
import time
import warnings
from shutil import copy
from argparse import ArgumentParser

from data import get_loader
from modules import build_model

from train import train
from test import reconstruction, animate

warnings.filterwarnings('ignore')

SLURM_JOBID = os.getenv('SLURM_JOBID')
SLURM_JOB_USER = os.getenv('SLURM_JOB_USER')
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.safe_load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # build model
    model_dict = build_model(config['model_params'])

    # root_dir = os.path.join('/ssd', SLURM_JOB_USER, SLURM_JOBID, 'vox_video')
    # config['dataset_params']['root_dir'] = root_dir
    data_loader = get_loader(mode=opt.mode, loader_params=config['loader_params'],
                             dataset_params=config['dataset_params'])

    # create log dir, copy config file to log dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    # print each model's parameters amount
    print()
    count = 0
    for key in model_dict:
        num = sum(param.numel() for param in model_dict[key].parameters())
        print('[%s] --- [%d]' % (key, num))
        count += num
    print('[all params] --- [%d]' % count)
    print()

    # enter corresponding mode code
    if opt.mode == 'train':
        print("Training...")
        train(config, model_dict, opt.checkpoint, log_dir, data_loader)
    elif opt.mode == 'reconstruction':
        print('Reconstruction...')
        reconstruction(config, model_dict, opt.checkpoint, log_dir, data_loader)
    elif opt.mode == 'animate':
        print('Animate...')
        animate(config, model_dict, opt.checkpoint, log_dir, data_loader)
    else:
        print('No implementation...')
