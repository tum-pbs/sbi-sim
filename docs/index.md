# SBI-SIM: Neural Posterior Estimation in JAX

Welcome to `sbisim`, a powerful library for **Generative Modeling** and **Simulation-Based Inference (SBI)** implemented in JAX. This library provides advanced methods for training and inference of conditional generative models, enabling efficient posterior estimation for complex simulators.

## 🚀 Key Features

- **JAX-powered**: Leverage the speed and automatic differentiation of JAX
- **Multiple SBI strategies**: Support for various simulation-based inference approaches
- **Flexible flows**: Advanced normalizing flow implementations

## 🎯 What is Simulation-Based Inference?

Simulation-Based Inference (SBI) is a powerful framework for performing Bayesian inference when you have access to a simulator but cannot easily compute the likelihood function.

- Train neural networks to approximate posterior distributions
- Perform inference on complex, high-dimensional problems
- Handle intractable likelihoods through simulation
- Scale to large datasets and complex models

## 📚 Quick Navigation

### Getting Started
- [Installation](getting-started/installation.md) - Set up `sbisim` in your environment
- [Quick Start](getting-started/quick-start.md) - Run your first experiment
- [Configuration](getting-started/configuration.md) - Configure `sbisim` for your needs

### User Guide
- [Overview](user-guide/overview.md) - Core concepts and architecture
- [Strategies](user-guide/strategies.md) - Different SBI approaches
- [Flows](user-guide/flows.md) - Normalizing flow implementations
- [Callbacks](user-guide/callbacks.md) - Customizing training and inference
- [Optimization](user-guide/optimization.md) - Training optimization techniques

### API Reference
- [Strategy](api/strategy.md) - Core strategy implementations
- [Flows](api/flows.md) - Flow model APIs
- [Callbacks](api/callbacks.md) - Callback system reference
- [Optimization](api/optimization.md) - Optimization utilities
- [Data](api/data.md) - Data handling and utilities

### Examples
- [Basic Examples](examples/basic-examples.md) - Simple SBI workflows
- [Advanced Examples](examples/advanced-examples.md) - Complex use cases
- [Benchmarks](examples/benchmarks.md) - Performance comparisons

## 💡 Quick Example

```python
import jax
import jax.numpy as jnp

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
                        train_dataloader, val_dataloader, callbacks, checkpoint=config.runtime.logdir, full_config=OmegaConf.to_container(config_raw))

trainer.train_model()

```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on how to:

- Report bugs
- Suggest new features
- Submit pull requests
- Contribute to documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready to get started?** Check out our [Quick Start Guide](getting-started/quick-start.md) to run your first experiment! 