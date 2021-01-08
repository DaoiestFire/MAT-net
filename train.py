import time
import datetime
import torch

from logger import Logger
from data import InputFetcher
from full_model import FullModel

from torch.optim.lr_scheduler import MultiStepLR


def train(config, model_dict, checkpoint, log_dir, data_loader):
    train_params = config['train_params']

    optimizer_regress = torch.optim.Adam(model_dict.regression_module.parameters(),
                                         lr=train_params['lr_regression_module'], betas=(0.5, 0.999))
    optimizer_gen = torch.optim.Adam(model_dict.generator.parameters(), lr=train_params['lr_generator'],
                                     betas=(0.5, 0.999))
    optimizer_dis = torch.optim.Adam(model_dict.discriminator.parameters(),
                                     lr=train_params['lr_discriminator'], betas=(0.5, 0.999))

    optimizer = {'opt_regress': optimizer_regress,
                 'opt_gen': optimizer_gen,
                 'opt_dis': optimizer_dis}

    if checkpoint is not None:
        start_iteration = Logger.load_cpk(checkpoint, model_dict, optimizer)
    else:
        start_iteration = 0

    scheduler_regress = MultiStepLR(optimizer_regress, train_params['epoch_milestones'], gamma=0.1,
                                    last_epoch=start_iteration - 1)
    scheduler_gen = MultiStepLR(optimizer_gen, train_params['epoch_milestones'], gamma=0.1,
                                last_epoch=start_iteration - 1)
    scheduler_dis = MultiStepLR(optimizer_dis, train_params['epoch_milestones'], gamma=0.1,
                                last_epoch=start_iteration - 1)

    full_model = FullModel(model_dict, train_params)
    if torch.cuda.is_available():
        full_model.cuda()

    fetcher = InputFetcher(data_loader)
    start_time = time.time()
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for iteration in range(start_iteration, train_params['num_iterations']):
            x = next(fetcher)
            # update generator
            loss_gen, loss_values, generated = full_model.update_gen(x)
            loss_gen.backward()
            optimizer_regress.step()
            optimizer_regress.zero_grad()
            optimizer_gen.step()
            optimizer_gen.zero_grad()

            # update discriminator
            loss_dis, loss_values_dis = full_model.update_dis()
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()
            optimizer_dis.zero_grad()
            # log
            loss_values.update(loss_values_dis)
            losses = {key: value.mean().detach().data.cpu().numpy() for key, value in loss_values.items()}
            logger.log_iter(losses=losses)
            if (iteration + 1) % train_params["log_freq"] == 0:
                elapsed_time = time.time() - start_time
                elapsed_time = str(datetime.timedelta(seconds=elapsed_time))[:-7]
                print("[current iteration]: %d ; [elapsed time]: %s" % (iteration + 1, elapsed_time))

            if (iteration + 1) % train_params["eval_freq"] == 0:
                logger.log_epoch(iteration, model_dict, optimizer, inp=x, out=generated)

            scheduler_regress.step()
            scheduler_gen.step()
            scheduler_dis.step()
