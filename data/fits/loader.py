from typing import Iterable

from jax import tree_util
from torch.utils.data import Dataset, DataLoader

import multiprocessing as mp
import ctypes
import numpy as np
import jax.numpy as jnp
import os
import glob
import astropy.io.fits as fits

import torch

from simulations import SimulationTargets
from utils import instantiate_from_config
from .noise import NumpyNoise

def parse_fits_file(file_path):

    hdul = fits.open(file_path, memmap=False)
    data = np.array(hdul[0].data)
    parameters = np.array(hdul[1].data.values)
    hdul.close()

    return data, parameters

class FITSDataset(Dataset):
    def __init__(self, img_dir, mask, transform=None, target_transform=None):
        self.img_files = glob.glob(img_dir + '/*.fits')

        # sort
        self.img_files.sort()

        self.img_dir = img_dir
        self.mask = mask
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_files)

    def load_samples(self, num_samples: int):

        data = []

        for i in range(num_samples):
            data.append(self.__getitem__(i))

        return numpy_collate(data)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_files[idx])


        data, parameters = parse_fits_file(img_path)
        parameters_masked = parameters[:len(self.mask)][self.mask]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            parameters = self.target_transform(parameters)

        return np.expand_dims(data, axis=0), parameters, parameters_masked

class SharedFITSDataset(FITSDataset):

    _use_cache = False
    _fill_cache = False

    def __init__(self, img_dir, mask, transform=None, target_transform=None):
        super(SharedFITSDataset, self).__init__(img_dir, mask, transform, target_transform)


        num_samples = self.__len__()

        cached_base = mp.Array(ctypes.c_bool, num_samples)
        cached = np.ctypeslib.as_array(cached_base.get_obj())
        cached[:] = False
        self.cached = cached

        example_data, example_parameters, example_parameters_masked = self.__getitem__(0)

        shared_data_base = mp.Array(ctypes.c_float, num_samples * int(np.prod(example_data.shape)))
        shared_data = np.ctypeslib.as_array(shared_data_base.get_obj())
        self.shared_data = shared_data.reshape(num_samples, *example_data.shape)

        shared_parameters_base = mp.Array(ctypes.c_float, num_samples * int(np.prod(example_parameters.shape)))
        shared_parameters = np.ctypeslib.as_array(shared_parameters_base.get_obj())
        self.shared_parameters = shared_parameters.reshape(num_samples, *example_parameters.shape)

        shared_parameters_masked_base = mp.Array(ctypes.c_float, num_samples * int(np.prod(example_parameters_masked.shape)))
        shared_parameters_masked = np.ctypeslib.as_array(shared_parameters_masked_base.get_obj())
        self.shared_parameters_masked = shared_parameters_masked.reshape(num_samples, *example_parameters_masked.shape)

        self.use_cache(True)
        self.fill_cache(True)

    def __getitem__(self, idx):

        if self._use_cache and self.cached[idx]:
            return self.shared_data[idx], self.shared_parameters[idx], self.shared_parameters_masked[idx]

        data, parameters, parameters_masked = super(SharedFITSDataset, self).__getitem__(idx)

        if self._fill_cache:
            self.cached[idx] = True
            self.shared_data[idx] = data
            self.shared_parameters[idx] = parameters
            self.shared_parameters_masked[idx] = parameters_masked

        return data, parameters, parameters_masked

    def use_cache(self, flag: bool):
        self._use_cache = flag

    def fill_cache(self, flag: bool):
        self._fill_cache = flag


class CachedDataset(FITSDataset):

    def __init__(self, img_dir, mask, transform=None, target_transform=None):

        super(CachedDataset, self).__init__(img_dir, mask, transform, target_transform)

        self.cached = set([])
        self.data_cache = {}

    def __getitem__(self, idx):

        if idx in self.cached:
            return self.data_cache[idx]
        else:
            elem = super(CachedDataset, self).__getitem__(idx)
            self.data_cache[idx] = elem
            self.cached.add(idx)
            return elem

def numpy_collate(batch):

    data, parameters, parameters_masked = zip(*batch)
    data = np.stack(data)
    parameters = np.stack(parameters)
    parameters_masked = np.stack(parameters_masked)

    return {'parameters': parameters_masked, 'conditioning': (data, parameters),
            'weighting': np.ones(shape=parameters_masked.shape[0])
            }

class FITSLoader(DataLoader):

    dataset: FITSDataset

    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)

    def __getitem__(self, item):
        sample = self.dataset[item]
        return {'parameters': sample[2], 'conditioning': (sample[0], sample[1]),
                'weighting': 1}

    def load_samples(self, num_samples: int):
        return self.dataset.load_samples(num_samples)

def noise(npix, background_rms, exposure_time):
    noise_generator = NumpyNoise(npix, npix, background_rms=background_rms,
                  exposure_time=exposure_time)
    def add_noise(data):
        return data + noise_generator.realisation(data)

    return add_noise


class FITSLoaderJAX(Iterable):
    def __init__(self, fits_loader):
        self.fits_loader = fits_loader
        self.batch_size = fits_loader.batch_size
        self.dataset = fits_loader.dataset

    def __iter__(self):
        for batch in self.fits_loader:
            batch = tree_util.tree_map(lambda x: jnp.array(x), batch)
            yield batch

    def __getitem__(self, item):
        sample = self.fits_loader[item]
        return tree_util.tree_map(lambda x: jnp.array(x), sample)

    def __len__(self):
        return len(self.fits_loader)

    def load_samples(self, num_samples: int):
        return tree_util.tree_map(lambda x: jnp.array(x), self.fits_loader.load_samples(num_samples))

def get_fits_dataloader(fits_dir, batch_size, simulation_target, num_workers=0, cached=False, shuffle=True, add_noise=True, **noise_kwargs):

    if add_noise:
        transform = noise(**noise_kwargs)
    else:
        transform = None

    simulation_target: SimulationTargets = instantiate_from_config(simulation_target)
    mask = simulation_target.get_target_mask()

    if cached:
        fits_dataset = SharedFITSDataset(fits_dir, mask, transform=transform)
    else:
        fits_dataset = FITSDataset(fits_dir, mask, transform=transform)

    fits_loader = FITSLoader(fits_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return FITSLoaderJAX(fits_loader)