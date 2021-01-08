"""
code from https://github.com/AliaksandrSiarohin/first-order-model
"""
import torch
from torch import nn
import torch.nn.functional as F
from .utils import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, deform_input


class Generator(nn.Module):
    """
    Generator that given source image and and sampling grid try to transform image.
    Generator follows Johnson architecture.
    """

    def __init__(self, block_expansion, max_features, num_down_blocks, num_bottleneck_blocks):
        super(Generator, self).__init__()
        self.first = SameBlock2d(3, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, 3, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, source_image, warp_grid_dict):
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = dict()
        output_dict['mask'] = warp_grid_dict['mask']
        output_dict['sparse_deformed'] = warp_grid_dict['sparse_deformed']

        if 'occlusion_mask' in warp_grid_dict:
            occlusion_mask = warp_grid_dict['occlusion_mask']
            output_dict['occlusion_mask'] = occlusion_mask
        else:
            occlusion_mask = None

        deformation = warp_grid_dict['deformation']
        out = deform_input(out, deformation)

        if occlusion_mask is not None:
            out = out * occlusion_mask

        if "deformed" in warp_grid_dict:
            output_dict['deformed'] = warp_grid_dict['deformed']

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict
