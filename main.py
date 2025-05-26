import argparse
import copy
import os
import datetime
import sys
from gc import callbacks
from pathlib import Path

import wandb
from omegaconf import OmegaConf

import jax.random as jr

from optimization.optimization import Optimization
from trainer import TrainerModule
from utils import instantiate_from_config, parse_config, restore_checkpoint


DEFAULT_ENVIRONMENT = './env/local.yaml'

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="Name of the experiment. Used for logging.",
    )
    parser.add_argument(
        "--dryrun",
        action='store_true',
        help="If true, don't log anything and don't create a logdir.",
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        help="If true, don't enable jit for debugging.",
    )

    parser.add_argument(
        "-c",
        "--config",
        nargs="*",
        metavar="flow.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )

    parser.add_argument(
        "-p",
        "--project",
        type=str,
        help="wandb project name",
        default='simulation-based-inference',
    )

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="path to environment config",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="./logs",
        help="directory for logging",
    )

    return parser

def get_modules(config, config_raw):

    config = copy.deepcopy(config)
    OmegaConf.resolve(config)

    data_config = config.pop("data", OmegaConf.create({}))
    data_config = OmegaConf.to_container(data_config, resolve=True)

    train_dataloader = instantiate_from_config(data_config["train"])
    val_dataloader = instantiate_from_config(data_config["val"])

    optimization = Optimization(config.pop("optimization", OmegaConf.create({})))

    train_config = config.pop("training", OmegaConf.create({}))
    train_config.logdir = config["runtime"]["logdir"]
    train_config.seed = config["runtime"]["seed"]

    callback_config = config.pop("callbacks", OmegaConf.create({}))
    callback_config = OmegaConf.to_container(callback_config, resolve=True)

    callbacks = [instantiate_from_config(config) for config in callback_config.values()]

    strategy = instantiate_from_config(config.pop("strategy", OmegaConf.create({})))
    strategy.set_batch_size(train_config.batch_size)

    trainer = TrainerModule(train_config, strategy, optimization,
                            train_dataloader, val_dataloader, callbacks, checkpoint=config.runtime.logdir,
                            full_config=OmegaConf.to_container(config_raw))

    return {
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'train_config': train_config,
        'optimization': optimization,
        'callbacks': callbacks,
        'strategy': strategy,
        'trainer': trainer
    }



def run(config):

    config_raw = copy.deepcopy(config)

    OmegaConf.resolve(config)

    module_dict = get_modules(config, config_raw)

    train_config = module_dict['train_config']
    train_dataloader = module_dict['train_dataloader']
    val_dataloader = module_dict['val_dataloader']
    trainer = module_dict['trainer']

    if not 'active' in train_config or train_config.active:
        trainer.train_model()

    test_config = config.pop("test", OmegaConf.create({'active': False}))

    if test_config.active:

        test_callbacks = [instantiate_from_config(config) for config in test_config.tests.values()]

        try:

            for checkpoint_type in ['best_val', 'best_train']: #, 'latest'

                print(f"Testing {checkpoint_type} checkpoint...")

                trainer.restore_from_checkpoint(config.runtime.logdir, type=checkpoint_type)
                trainer.pack_()

                if 'seed' in test_config:
                    test_rng = jr.PRNGKey(test_config.seed)
                else:
                    test_rng = jr.PRNGKey(0)

                logs = {}
                for callback in test_callbacks:

                    try:
                        logs, _ = callback.on_test(logs, test_rng, strategy=trainer.strategy, train_loader=train_dataloader,
                                                val_loader=val_dataloader)

                    except Exception as e:
                        print(f"Failed to run test callback {callback.name}: {e}")
                        raise e

                print("Test results: ", logs)

                for key in logs:
                    wandb.run.summary[f'{key}_{checkpoint_type}'] = logs[key]

        except Exception as e:
            raise e

def strip_hyphens(unknown_args):
    # remove hyphens from keys
    for i, arg in enumerate(unknown_args):
        if arg.startswith("--"):
            unknown_args[i] = arg[2:]
    return unknown_args

def load_environment(env):

    if env is None:
        print(f'Loading default environment at {DEFAULT_ENVIRONMENT}')
        env_config = OmegaConf.load(DEFAULT_ENVIRONMENT)
    else:
        try:
            env_config = OmegaConf.load(env)
        except Exception as e:
            print(f'Could not load environment {env}:')
            print(e)
            print(f'Loading environment {DEFAULT_ENVIRONMENT} instead')
            env_config = OmegaConf.load(DEFAULT_ENVIRONMENT)

    return env_config


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()


    try:

        configs = [parse_config(OmegaConf.load(cfg)) for cfg in opt.config]
        unknown = strip_hyphens(unknown)
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

    except Exception as e:
        print('Failed to load config')
        raise e

    # try loading checkpoint

    if opt.name == "" or opt.name is None:
        if "name" in config:
            opt.name = config["name"]
        else: # use timestamp as name
            opt.name = now

    logdir = Path(opt.logdir).joinpath(opt.name)

    # check if logdir exists
    if logdir.exists():
        print(f"Run directory {logdir} already exists. Trying to continue run...")

        try:
            checkpoint = restore_checkpoint(logdir)
            config_old = OmegaConf.create(checkpoint['config'])
            id = config_old.runtime.id
            print(f"Continuing run with id {id}")

        except Exception as e:
            print('Could not restore previous checkpoint.')
            print('Reason:', e)
            id = wandb.util.generate_id()
            print(f'Creating new run with id {id}')

    else:
        logdir.mkdir(parents=True, exist_ok=True)
        id = wandb.util.generate_id()
        print(f'Creating new run with id {id}')

    runtime_config = OmegaConf.create({'runtime': {'seed': opt.seed,
                                                   'logdir': logdir.absolute().as_posix(),
                                                   'name': opt.name, 'id': id}})

    env = load_environment(opt.env)

    config = OmegaConf.merge(config, runtime_config, env)

    config_full = copy.deepcopy(config)

    OmegaConf.resolve(config)

    # after resolving, delete environment entries in config
    for k, v in env.items():
        config.pop(k)

    if opt.dryrun:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"

    if opt.debug:
        from jax import config as jconfig
        jconfig.update('jax_disable_jit', True)
        print("JIT disabled for debugging")

    print("Config: ", config)

    wandb_logdir = logdir.joinpath(now)

    wandb_logdir.mkdir(parents=True, exist_ok=True)

    # setup wandb
    wandb.init(project=opt.project, id=id, name=opt.name, resume="allow", dir=wandb_logdir,
               mode=wandb_mode, config=OmegaConf.to_container(config, resolve=True))

    run(config_full)

    ###########################################################


