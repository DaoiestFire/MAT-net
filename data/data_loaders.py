from torch.utils.data import DataLoader

from .datasets import FramesDataset, PairedDataset


def get_loader(mode='train', loader_params=None, dataset_params=None):
    print('TYPE of dataset-video or image sequence...')
    if mode == 'train':
        print("---Preparing training DataLoader ...")
        train_params = loader_params['train']
        dataset = FramesDataset(is_train=True, **dataset_params)
        return DataLoader(dataset=dataset,
                          batch_size=train_params['batch_size'],
                          shuffle=train_params['shuffle'],
                          num_workers=train_params['num_workers'],
                          pin_memory=True,
                          drop_last=train_params['drop_last'])
    elif mode == 'reconstruction':
        print("---Preparing reconstruction DataLoader ...")
        reconstruction_params = loader_params['reconstruction']
        dataset = FramesDataset(is_train=False, **dataset_params)
        return DataLoader(dataset=dataset,
                          batch_size=reconstruction_params['batch_size'],
                          shuffle=reconstruction_params['shuffle'],
                          num_workers=reconstruction_params['num_workers'],
                          drop_last=reconstruction_params['drop_last'])
    elif mode == 'animate':
        print("---Preparing animate DataLoader ...")
        animate_params = loader_params['animate']
        dataset = FramesDataset(is_train=False, **dataset_params)
        dataset = PairedDataset(initial_dataset=dataset,
                                number_of_pairs=animate_params['num_pairs'])
        return DataLoader(dataset=dataset,
                          batch_size=animate_params['batch_size'],
                          shuffle=animate_params['shuffle'],
                          num_workers=animate_params['num_workers'],
                          drop_last=animate_params['drop_last'])
    else:
        print("No DataLoader ...")
        return None


class InputFetcher:
    def __init__(self, loader):
        self.loader = loader

    def _fetch_inputs(self):
        try:
            x = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x = next(self.iter)
        return x

    def __next__(self):
        x = self._fetch_inputs()
        tmp = dict()
        for key in x:
            tmp.update({key: x[key]} if key == 'name' else {key: x[key].cuda()})
        return tmp
