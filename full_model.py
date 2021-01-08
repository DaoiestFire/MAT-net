"""
merger all model in one model.
this is a trainer.
"""
import numpy as np

import torch
import torch.nn as nn
from torchvision import models


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class FullModel(nn.Module):
    def __init__(self, model_dict, train_params):
        super(FullModel, self).__init__()
        self.generator = model_dict.generator
        self.discriminator = model_dict.discriminator
        self.regression_module = model_dict.regression_module
        self.loss_weights = train_params['loss_weights']

        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()

    @staticmethod
    def recon_criterion(input, target):
        diff = input - target.detach()
        return torch.mean(torch.abs(diff[:]))

    @staticmethod
    def r1_rg(d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

    def cal_gen_loss(self, input_fake):
        prediction_maps = self.discriminator(input_fake)
        loss = ((1 - prediction_maps) ** 2).mean()
        return loss

    def cal_dis_loss(self, fake_image, real_image):
        real_image.requires_grad_()
        predictions_fake = self.discriminator(fake_image.detach())
        predictions_real = self.discriminator(real_image)

        loss = ((1 - predictions_real) ** 2 + predictions_fake ** 2).mean()
        reg = self.r1_rg(predictions_real, real_image).mean()
        return loss, reg

    def cal_vgg_loss(self, fake_image, real_image):
        x_vgg = self.vgg(fake_image)
        y_vgg = self.vgg(real_image)
        loss = 0
        for i in range(5):
            loss += self.recon_criterion(x_vgg[i], y_vgg[i])
        return loss

    def update_gen(self, x):
        self.x = x
        source = x['source']
        driving = x['driving']

        grid_dict = self.regression_module(source, driving)
        generated = self.generator(source, grid_dict)
        self.generated = generated

        # loss
        loss_vgg = self.cal_vgg_loss(generated['prediction'], driving)
        loss_gen = self.cal_gen_loss(generated['prediction'])

        loss_values = dict()
        loss_values['vgg'] = loss_vgg
        loss_values['gen'] = loss_gen

        loss_total = loss_vgg * self.loss_weights['vgg'] + \
                     loss_gen * self.loss_weights['gan_weight']

        return loss_total, loss_values, generated

    def update_dis(self):
        fake_image = self.generated['prediction']
        real_image = self.x['driving']
        loss, reg = self.cal_dis_loss(fake_image, real_image)

        loss_values = dict()
        loss_values['dis'] = loss
        loss_values['reg'] = reg
        loss_total = loss * self.loss_weights['gan_weight'] + reg * self.loss_weights['reg']
        return loss_total, loss_values

    def forward(self):
        pass
