import os
import time
import imageio
import numpy as np
from tqdm import tqdm

import torch

from logger import Logger, Visualizer


def animate(config, model_dict, checkpoint, log_dir, data_loader):
    log_dir = os.path.join(log_dir, 'animation')
    png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']
    use_relative = animate_params['use_relative']

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, model_dict=model_dict)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

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
        with torch.no_grad():
            predictions = []
            visualizations = []
            driving_video = x['driving_video'].cuda()
            source_frame = x['source_video'][:, :, 0, :, :].cuda()
            driving_frame_initial = driving_video[:, :, 0]

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                if use_relative:
                    transform_params = regression_module.encode_transform_params(driving_frame_initial, driving_frame)
                else:
                    transform_params = regression_module.encode_transform_params(source_frame, driving_frame)
                grid_dict = regression_module.generate_grid(source_frame, transform_params)
                out = generator(source_frame, grid_dict)
                del out['sparse_deformed']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                inp = {'source': source_frame, 'driving': driving_frame}
                visualization = Visualizer(**config['visualizer_params']).visualize(inp=inp, out=out)
                visualization = visualization
                visualizations.append(visualization)
            fps = x['driving_video'].shape[2] / (time.time() - start_time)
            fps_list.append(fps)

            predictions = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions).astype(np.uint8))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("fps : %s" % np.mean(fps_list))
