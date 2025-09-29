import os
from pathlib import Path

from typing import Sized

import jax.numpy as jnp
import jax.random as jr
from flax.training.early_stopping import EarlyStopping
from jax import device_get
from jax.tree_util import tree_leaves

import numpy as np
import wandb
from tqdm import tqdm

from .optimization.optimization import Optimization, OptState
from .utils import restore_checkpoint
from .strategy import Strategy


class TrainerModule:

    def __init__(self, config: dict, strategy: Strategy, opt: Optimization,
                 train_loader, val_loader, callbacks, checkpoint=None,
                 full_config=None):
        super().__init__()

        self.config = full_config
        self.strategy = strategy
        self.validation = True
        self.log_dir = config['logdir']
        self.seed = config['seed']
        self.num_epochs = config['num_epochs']
        self.early_stopping_config = config.get('early_stopping', {'patience': float('inf')})
        self.early_stopping = EarlyStopping(**self.early_stopping_config)

        if 'num_steps_per_epoch' in config:
            self.num_steps_per_epoch = config['num_steps_per_epoch']
        else:
            self.num_steps_per_epoch = len(train_loader)

        self.opt = opt

        self.callbacks = callbacks

        self.example_data = next(iter(train_loader))
        self.log_dir = os.path.join(self.log_dir)
        self.model_name = 'flow_model'

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epoch = 0
        self.global_step = 0

        # declare variables
        self.get_params = None
        self.opt_update = None
        self.rng = None
        self.callback_rng = None
        self.eval_step = None
        self.train_step = None
        self.metrics = {}

        self.init_strategy()

        self.restore_from_checkpoint(checkpoint)

    def init_strategy(self):

        self.rng = jr.PRNGKey(self.seed)
        self.rng, self.callback_rng = jr.split(self.rng, 2)

        init_, rng = self.strategy.setup(self.opt, self.example_data,
                                         self.rng, self.train_loader.batch_size)

        self.rng = rng

        params = init_['params']

        # log number of parameters
        wandb.run.summary["num_params"] = sum([p.size for p in
                                               tree_leaves(params)])

        print("Number of parameters: ", wandb.run.summary["num_params"])

        # Check if JAX is using GPU
        from jax.lib import xla_bridge
        platform = xla_bridge.get_backend().platform
        if platform == 'gpu':
            print("JAX is using GPU")
        else:
            print(f"JAX is using {platform}")

    def pack_(self):
        self.strategy.bind(self.opt.get_params())
        return {'rng': self.callback_rng, 'strategy': self.strategy,
                'train_loader': self.train_loader, 'val_loader': self.val_loader,
                'ckpt': self.create_checkpoint()}

    def create_checkpoint(self):
        return {
                'state': self.opt.get_state(),
                'global_step': self.global_step,
                'metrics': {key: float(value) for key, value in self.metrics.items()},
                'epoch': self.epoch,
                'rng': self.rng,
                'config': self.config
                }

    def restore_from_checkpoint(self, checkpoint, type: str='latest'):

        if not checkpoint is None:

            checkpoint = Path(checkpoint)

            try:

                ckpt_target = self.create_checkpoint()

                ckpt = restore_checkpoint(checkpoint, target=ckpt_target, type=type)

                self.opt.set_state(ckpt['state'])
                self.global_step = ckpt['global_step']
                self.epoch = ckpt['epoch']
                self.rng = ckpt['rng']

                print(f"Restored from checkpoint {checkpoint} at global step {self.global_step}")

            except Exception as e:
                print(f"Could not restore from checkpoint: {e}")


    def train_model(self):

        if self.epoch >= self.num_epochs:
            print("Model already trained for", self.num_epochs, "epochs")
            return {}, self.rng

        rng = self.rng
        logs = {'epoch': self.epoch, 'global_step': self.global_step, 'max_epochs': self.num_epochs}

        if self.epoch == 0:
            logs = self.on_train_begin(logs, **self.pack_())

        for epoch_idx in range(self.epoch + 1, self.num_epochs + 1):
            self.epoch = epoch_idx
            logs['epoch'] = epoch_idx

            logs = self.on_epoch_begin(logs, **self.pack_())

            logs, rng = self.train_epoch(logs, rng)

            logs = self.on_epoch_end(logs, **self.pack_())

            self.opt.update_scheduler(logs["global_step"], logs)

            self.early_stopping = self.early_stopping.update(logs['val_loss'])

            if self.early_stopping.should_stop:
                print(f'Met early stopping criteria, breaking after epoch {self.epoch}')
                break

            logs = {'epoch': self.epoch, 'global_step': self.global_step, 'max_epochs': self.num_epochs}

        logs = self.on_train_end(logs, **self.pack_())

        return logs, rng

    def train_epoch(self, logs, rng):

        avg_loss = 0.0

        num_batches = self.num_steps_per_epoch
        if issubclass(type(self.train_loader), Sized):
            num_batches = min(len(self.train_loader), num_batches)

        p = tqdm(self.train_loader, leave=True, position=0, total=num_batches)
        opt_state = self.opt.get_state()
        global_step = self.global_step

        logs["lr"] = self.opt.get_learning_rate(global_step)
        logs["train/loss"] = 0.0
        logs["val/loss"] = 0.0

        logs = {x: jnp.array(v).astype(jnp.float32) for x, v in logs.items()}

        step_ = 0
        for batch in p:

            opt_state, rng, logs = self.strategy.train_step(self.global_step, opt_state,
                                                   rng, logs, batch)
            avg_loss += logs['train/loss']
            global_step += 1
            p.set_description(f'{self.epoch}/{self.num_epochs} loss: {avg_loss/(step_+1):.3f}')

            step_ += 1
            if step_ >= num_batches:
                break

        avg_loss /= step_
        logs['loss'] = avg_loss
        self.metrics['loss'] = avg_loss

        if self.validation:

            avg_loss = 0.0

            num_val_batches = min(len(self.val_loader), self.num_steps_per_epoch)
            p = tqdm(self.val_loader, leave=True, position=0, total=num_val_batches)

            step_ = 0
            for batch in p:

                rng, logs = self.strategy.eval_step(opt_state,
                                                    rng, logs, batch, testing=False)
                avg_loss += logs['val/loss']
                p.set_description(f'{self.epoch}/{self.num_epochs} val loss: {avg_loss/(step_+1):.3f}')

                step_ += 1
                if step_ >= num_val_batches:
                    break

            avg_loss /= step_

            logs['val_loss'] = avg_loss
            self.metrics['val_loss'] = avg_loss

        self.global_step = global_step
        self.opt.set_state(opt_state)


        return logs, rng

    def eval_model(self, data_loader, testing=False):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            self.rng, logs = self.strategy.eval_step(self.opt.get_params(), self.rng, batch, testing=testing)
            losses.append(jnp.sum(logs['val/loss']))
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss

    def on_train_begin(self, logs, *args, **kwargs):

        for callback in self.callbacks:
            try:
                logs, _ = callback.on_train_begin(logs, *args, **kwargs)
            except Exception as e:
                print(f"Callback {callback.name} failed with exception {e}")
                raise e

        wandb.log(logs)

        logs = {'epoch': self.epoch, 'global_step': self.global_step, 'max_epochs': self.num_epochs}

        return logs

    def on_train_end(self, logs, *args, **kwargs):

        for callback in self.callbacks:
            try:
                logs, _ = callback.on_train_end(logs, *args, **kwargs)
            except Exception as e:
                print(f"Callback {callback.name} failed with exception {e}")

        wandb.log(logs)

        logs = {'epoch': self.epoch, 'global_step': self.global_step}

        return logs

    def on_epoch_begin(self, logs, *args, **kwargs):

        for callback in self.callbacks:
            try:
                logs, _ = callback.on_epoch_begin(logs, *args, **kwargs)
            except Exception as e:
                print(f"Callback {callback.name} failed with exception: {e}")

        return logs

    def on_epoch_end(self, logs, *args, **kwargs):

        for callback in self.callbacks:
            try:
                logs, _ = callback.on_epoch_end(logs, *args, **kwargs)
            except Exception as e:
                print(f"Callback {callback.name} failed with exception: {e}")
                raise e

        wandb.log(logs)

        return logs
