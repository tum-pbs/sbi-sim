import importlib
from pathlib import Path
import orbax
from flax.training import orbax_utils
from omegaconf import OmegaConf, DictConfig
import jax.random as jr

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def generate_apply_rngs(rng):
    s = jr.split(rng, 5)
    rngs = {'dropout': s[1],
            'drop_path': s[2],
            'dropout_probability': s[3],
            'simulator': s[4]}
    return rngs, s[0]

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    
    try:

        if not "params" in config or config["params"] is None:
            return get_obj_from_str(config["target"])()
        else:
            return get_obj_from_str(config["target"])(**config.get("params", dict()))
        
    except Exception as e:
        print(f"Error instantiating {config['target']}: {e}")
        raise e


def instantiate_from_config_(target, **params):
    return get_obj_from_str(target)(**params)

def parse_config(config):
    conf_ = OmegaConf.create({})
    for key, value in config.items_ex(resolve=False):
        if (isinstance(value, dict) or isinstance(value, OmegaConf)
                or isinstance(value, DictConfig)):
            conf_[key] = parse_config(value)
        elif isinstance(value, str) and key == "_file":
            conf_ = OmegaConf.merge(conf_, parse_config(OmegaConf.load(value)))
        else:
            conf_[key] = value
    return conf_

def restore_checkpoint(dir: Path, target=None, type: str="latest"):

    dir = dir.joinpath(type)
    orbax_checkpointer_latest = orbax.checkpoint.Checkpointer(
        orbax.checkpoint.PyTreeCheckpointHandler())
    options_latest = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=1, create=True)

    target=None # find out if this is really necessary
    restore_args = (orbax_utils.restore_args_from_target
                    (target, mesh=None))

    checkpoint_manager_latest = orbax.checkpoint.CheckpointManager(
        dir, orbax_checkpointer_latest, options_latest)

    step = checkpoint_manager_latest.latest_step()

    if step is None:
        raise FileNotFoundError(f"No checkpoint found in {dir}")
    else:
        print(f"Restoring {type} checkpoint...")

    return checkpoint_manager_latest.restore(step, items=target,
                                             restore_kwargs={'restore_args': restore_args})




