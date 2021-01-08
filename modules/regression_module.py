import torch
from torch import nn
import torch.nn.functional as F

from modules.utils import ResBlock2d, DownBlock2d, SameBlock2d
from modules.utils import AntiAliasInterpolation2d, deform_input, make_coordinate_grid
from modules.u_net import Unet


class RegressionNet(nn.Module):
    """regress a set of affine params between source image and driving image"""

    def __init__(self, dim_in, max_features, block_expansion, num_transforms, scale_factor=0.25):
        super(RegressionNet, self).__init__()
        self.from_rgb = SameBlock2d(dim_in, block_expansion, kernel_size=3, padding=1)

        in_channels = block_expansion
        backbone = []
        for _ in range(3):
            out_channels = min(in_channels * 2, max_features)
            backbone.append(DownBlock2d(in_channels, out_channels))
            backbone.append(ResBlock2d(out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        self.backbone = nn.Sequential(*backbone)

        num_params = num_transforms * 6
        self.params_map_gen = nn.Conv2d(in_channels, num_params, kernel_size=4, padding=0)
        self.params_decoder = nn.Conv2d(num_params, num_params, kernel_size=5, groups=num_params)
        self.params_decoder.weight.data.zero_()
        self.params_decoder.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0] * num_transforms, dtype=torch.float))

        self.num_transforms = num_transforms
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(dim_in, self.scale_factor)

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)
        out = self.from_rgb(x)
        out = self.backbone(out)
        params_map = self.params_map_gen(out)
        params = self.params_decoder(params_map)
        return params.view(params.size(0), self.num_transforms, 2, 3)


class GridGenerator(nn.Module):

    def __init__(self, dim_in, num_blocks, block_expansion, max_features, num_transforms,
                 estimate_occlusion_mask=False, scale_factor=0.25):
        super(GridGenerator, self).__init__()
        self.u_net = Unet(block_expansion=block_expansion, in_features=(num_transforms + 1) * dim_in,
                          max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.u_net.out_filters, num_transforms + 1, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_mask is True:
            self.occlusion = nn.Conv2d(self.u_net.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_transforms = num_transforms
        self.scale_factor = scale_factor

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(dim_in, self.scale_factor)

    @staticmethod
    def create_sparse_motions(source_image, matrix_transform):
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), data_type=matrix_transform.type())
        identity_grid = identity_grid.view(1, 1, h, w, 3)
        coordinate_grid = identity_grid
        transformation = matrix_transform.unsqueeze(-3).unsqueeze(-3)
        transformation = transformation.repeat(1, 1, h, w, 1, 1)
        coordinate_grid = torch.matmul(transformation, coordinate_grid.unsqueeze(-1))
        driving_to_source = coordinate_grid.squeeze(-1)

        # adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)[..., :2]
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_transforms + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_transforms + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_transforms + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_transforms + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, matrix_transform):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        sparse_motion = self.create_sparse_motions(source_image, matrix_transform)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source

        input_all = deformed_source.view(bs, -1, h, w)

        prediction = self.u_net(input_all)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation
        out_dict['deformed'] = deform_input(source_image, deformation)

        if self.occlusion is not None:
            occlusion_mask = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_mask'] = occlusion_mask

        return out_dict


class RegressionModule(nn.Module):
    def __init__(self, dim_in=6, max_features=256, num_transforms=10, block_expansion=32, grid_params=None):
        super(RegressionModule, self).__init__()
        self.regression_net = RegressionNet(dim_in, max_features, block_expansion, num_transforms)
        self.grid_generator = GridGenerator(num_transforms=num_transforms, **grid_params)

    def encode_transform_params(self, source_image, driving_image):
        inp_all = torch.cat([source_image, driving_image], dim=1)
        transform_params = self.regression_net(inp_all)
        return transform_params

    def generate_grid(self, source_image, transform_params):
        grid_dict = self.grid_generator(source_image, transform_params)
        return grid_dict

    def forward(self, source_image, driving_image):
        transform_params = self.encode_transform_params(source_image, driving_image)
        grid_dict = self.generate_grid(source_image, transform_params)
        return grid_dict
