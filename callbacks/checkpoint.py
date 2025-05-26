from pathlib import Path
from typing import Tuple

import orbax
from flax.training import orbax_utils

from callbacks.callback import Callback

import jax.random as jr


class Checkpoint(Callback):

    name: str = 'checkpoint'

    def __init__(self, savedir: str, key: str = 'val_loss', create: bool = True):

        timeout_secs = 30
        self.orbax_checkpointer_latest = orbax.checkpoint.AsyncCheckpointer(
            orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=timeout_secs
        )
        self.orbax_checkpointer_best_train = orbax.checkpoint.AsyncCheckpointer(
            orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=timeout_secs
        )
        self.orbax_checkpointer_best_val = orbax.checkpoint.AsyncCheckpointer(
            orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=timeout_secs
        )

        def best_fn_train(metrics):
            return metrics['loss']

        def best_fn_val(metrics):
            return metrics[key]

        self.options_latest = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=1, create=create)
        self.options_best_train = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=1, create=create, best_fn=best_fn_train, best_mode='min')
        self.options_best_val = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=1, create=create, best_fn=best_fn_val, best_mode='min')

        savedir_best_train = Path(savedir).joinpath('best_train')
        savedir_best_val = Path(savedir).joinpath('best_val')
        savedir_latest = Path(savedir).joinpath('latest')

        self.checkpoint_manager_best_train = orbax.checkpoint.CheckpointManager(
            savedir_best_train, self.orbax_checkpointer_best_train, self.options_best_train)
        self.checkpoint_manager_best_val = orbax.checkpoint.CheckpointManager(
            savedir_best_val, self.orbax_checkpointer_best_val, self.options_best_val)
        self.checkpoint_manager_latest = orbax.checkpoint.CheckpointManager(
            savedir_latest, self.orbax_checkpointer_latest, self.options_latest)

    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, ckpt=None, **kwargs) -> Tuple[dict, jr.PRNGKey]:

        if not ckpt is None:
            self.save_checkpoint(int(logs['epoch']), ckpt)

        return logs, rng

    def on_train_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return logs, rng

    def save_checkpoint(self, epoch, ckpt):
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager_best_train.save(epoch, ckpt,
                                          save_kwargs={'save_args': save_args},
                                          metrics=ckpt["metrics"])

        self.checkpoint_manager_best_val.save(epoch, ckpt,
                                                save_kwargs={'save_args': save_args},
                                                metrics=ckpt["metrics"])

        self.checkpoint_manager_latest.save(epoch, ckpt,
                                          save_kwargs={'save_args': save_args})

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self.__call__(logs, rng, *args, **kwargs)
