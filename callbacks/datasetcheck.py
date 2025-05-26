from typing import Iterable, Tuple, Sized

import jax.random as jr
from tqdm import tqdm

from callbacks.callback import Callback
from flows.flowbase import FlowBase


class DatasetCheck(Callback):
    max_num_batches: int = 1000

    name: str = 'dataset_check'

    def __init__(self, max_num_batches: int = 1000, cache: bool = False):
        self.max_num_batches = max_num_batches
        self.cache = cache

    def __call__(self, logs: dict, rng: jr.PRNGKey, train_loader: Iterable,
                 val_loader: Iterable, *args, **kwargs) -> Tuple[dict, jr.PRNGKey]:

        import time

        if self.cache:

            if hasattr(train_loader.dataset, 'fill_cache'):

                start = time.time()
                print("Caching training dataset...")

                train_loader.dataset.fill_cache(True)

                for batch in tqdm(train_loader):
                    pass

                if hasattr(train_loader.dataset, 'use_cache'):
                    train_loader.dataset.use_cache(True)

                train_loader.dataset.fill_cache(False)

                end = time.time()
                print(f"Time to cache training dataset: {end - start:.3f} seconds")

            if hasattr(val_loader.dataset, 'fill_cache'):

                start = time.time()
                print("Caching validation dataset...")

                val_loader.dataset.fill_cache(True)

                for batch in tqdm(val_loader):
                    pass

                if hasattr(val_loader, 'use_cache'):
                    val_loader.dataset.use_cache(True)

                val_loader.dataset.fill_cache(False)

                end = time.time()
                print(f"Time to cache validation dataset: {end - start:.3f} seconds")

        else:

            print("checking dataset...")
            # measure time for one dataset iteration

            start = time.time()

            if issubclass(type(train_loader), Sized):
                num_batches = min(len(train_loader), self.max_num_batches)
            else:
                num_batches = self.max_num_batches

            for i, batch in tqdm(enumerate(iter(train_loader))):
                if i >= num_batches:
                    break

            end = time.time()
            print(f"time to iterate {num_batches} training dataset batches: {end - start:.3f} seconds")

        return logs, rng

    def on_train_begin(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)