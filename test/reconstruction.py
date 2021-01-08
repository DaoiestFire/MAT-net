import os
import time
import imageio
import numpy as np
from tqdm import tqdm

import torch
from logger import Logger, Visualizer


def reconstruction(config, model_dict, checkpoint, log_dir, data_loader):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, model_dict=model_dict)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    generator = model_dict.generator
    regression_module = model_dict.regression_module
    if torch.cuda.is_available():
        generator = generator.cuda()
        regression_module = regression_module.cuda()

    generator.eval()
    regression_module.eval()

    fps_list = []
    for it, x in tqdm(enumerate(data_loader)):
        start_time = time.time()
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                transform_params = regression_module.encode_transform_params(source, driving)
                grid_dict = regression_module.generate_grid(source, transform_params)
                out = generator(source, grid_dict)
                del out['sparse_deformed']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                inp = {'source': source, 'driving': driving}
                visualization = Visualizer(**config['visualizer_params']).visualize(inp=inp, out=out)
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

            fps = x['video'].shape[2] / (time.time() - start_time)
            fps_list.append(fps)

            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("Reconstruction loss: %s" % np.mean(loss_list))
    print("fps : %s" % np.mean(fps_list))
