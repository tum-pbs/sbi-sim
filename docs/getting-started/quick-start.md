# Quick Start Guide

This guide will help you get started with `sbisim` quickly using a simple example.

## Running Experiments

`sbisim` uses YAML configuration files to define experiments. We define a simple example for the Two Moons problem and a Continuous Normalizing Flow (CNF) model.

### Basic Command

To train the model and evaluate it, we use the following command:

```bash
python main.py --config=configs/experiments/sbi_variants/ffjord/two_moons.yaml --name=TwoMoonsCNF --dryrun
```

- `python main.py`: Runs the main training script
- `--config`: Specifies the path to the YAML configuration file that defines the experiment setup
  - In this case, we're using a FFJORD (Free-form Jacobian of Reversible Dynamics) model to learn a conditional generative model to solve the Two Moons inference problem.
- `--name`: Sets a custom name for this experiment run (TwoMoonsCNF). Checkpoints and visualizations are stored in the directory `./logs/` appended by the experiment name.
- `--dryrun`: We use `wandb` for logging. This flag indicates that the run should not be synrchronized with the cloud.

The configuration file (`two_moons.yaml`) contains all the necessary setting to define the experiment. 

```yaml
name: TWO_MOONS_10e5_FFJORD # experiment name. Can be overwritten in CLI
project: SBI # project name for logging in wandb

dim_flow: 2 # dimensionaly of the normalizing flow
dim_conditioning: 2 # dimensionality of the conditioning

batch_size: 256 # batch size for learning

learning_rate: 0.001 # learning rate

data:
  target: sbisim.data.benchmarks.TwoMoons # loads the TwoMoons dataset
  _file: configs/datasets/benchmarks/10e5_dataset.yaml # file containing data splits and settings.

strategy:
  target: sbisim.strategy.NeuralPosteriorEstimation # maximum likelihood training
  params:
    model:
      target: sbisim.flows.ffjord.StackedFFJORD # FFJORD neural network architecture
      params:
        dim_flow: ${dim_flow} # references dim_flow defined above
        dim_conditioning: ${dim_conditioning}
        width_size: 256 # FFJORD network width
        depth: 6 # FFJORD number of layers

training:
  num_epochs: 200 # train for 200 epochs
  batch_size: ${batch_size}
  early_stopping:
    patience: 50 # early stopping if the val loss doesn't decrease after 50 steps

test: 
  active: True # after training run tests
  tests:
    c2st:
      target: sbisim.callbacks.C2ST # C2ST and MMD tests

patience: 3 # reduce learning rate if val loss does not decrease after 3 epochs
optimization:
  _file: configs/optimization/adam_reduce_lr.yaml # contains settings for optimizer

callbacks:
  checkpoint:
    target: sbisim.callbacks.checkpoint.Checkpoint # checkpoint for model saving
    params:
      savedir: ${runtime.logdir} # reference runtime.logdir which is determined from CLI
      key: val_loss # key for best train/val selection
  scatter:
    target: sbisim.callbacks.BenchmarkScatterPlot # visualization of posterior samples
    params:
      save_every: 50 # save every 50 epochs in ./logs/${savedir}/pictures
      savedir: ${runtime.logdir}
```








