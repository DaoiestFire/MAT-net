from munch import Munch

from .regression_module import RegressionModule
from .generator import Generator
from .discriminator import Discriminator


def build_model(model_params, test=False):
    regress = RegressionModule(**model_params['regression_module'])
    gen = Generator(**model_params['generator'])
    if test is False:
        dis = Discriminator(**model_params['discriminator'])
        return Munch({'regression_module': regress,
                      'generator': gen,
                      'discriminator': dis})
    else:
        return Munch({'regression_module': regress,
                      'generator': gen})
