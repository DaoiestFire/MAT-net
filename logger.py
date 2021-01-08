"""
code from https://github.com/AliaksandrSiarohin/first-order-model
"""
import os
import imageio
import collections
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):
        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.iteration = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.iteration + 1).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp, out)
        imageio.imsave(
            os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.iteration + 1).zfill(self.zfill_num)),
            image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.model_dict.items()}
        cpk.update({k: v.state_dict() for k, v in self.optimizer.items()})
        cpk['iteration'] = self.iteration
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.iteration + 1).zfill(self.zfill_num))
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, model_dict, optimizer=None, cpu=False):
        if cpu is True:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)
        for key in model_dict:
            if key in checkpoint:
                model_dict[key].load_state_dict(checkpoint[key])

        if optimizer is not None:
            for key in optimizer:
                if key in checkpoint:
                    optimizer[key].load_state_dict(checkpoint[key])
        iteration = checkpoint['iteration'] if 'iteration' in checkpoint else 0
        return iteration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, iteration, model_dict, optimizer, inp, out):
        self.iteration = iteration
        self.model_dict = model_dict
        self.optimizer = optimizer
        if (self.iteration + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, draw_border=False, colormap='gist_rainbow'):
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, inp, out):
        images = []

        key_list = ['source', 'driving']
        for key in key_list:
            temp = inp[key].data.cpu().numpy()
            temp = np.transpose(temp, [0, 2, 3, 1])
            images.append(temp)

        # Deformed image
        interpolate_size = images[0].shape[1:3]
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu()
            deformed = F.interpolate(deformed, size=interpolate_size).numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # Result
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        # Occlusion map
        if 'occlusion_mask' in out:
            occlusion_mask = out['occlusion_mask'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_mask = F.interpolate(occlusion_mask, size=interpolate_size).numpy()
            occlusion_mask = np.transpose(occlusion_mask, [0, 2, 3, 1])
            images.append(occlusion_mask)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out:
            full_mask = []
            for i in range(out['sparse_deformed'].shape[1]):
                image = out['sparse_deformed'][:, i].data.cpu()
                image = F.interpolate(image, size=interpolate_size)
                mask = out['mask'][:, i:(i + 1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=interpolate_size)
                image = np.transpose(image.numpy(), (0, 2, 3, 1))
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                images.append(image)
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

                full_mask.append(mask * color)

            images.append(sum(full_mask))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
