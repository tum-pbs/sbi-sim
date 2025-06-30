## Train the base network

For the Lotka Volterra Problem, we first train a neural network for estimate the posterior via flow matching. 

``
python main.py --config=configs/experiments/sbi_variants/fm-ot/lotka_volterra_jax.yaml --name=LotkaVolterraFM --dryrun
``

The weights of the trained models are stored at `./logs/LotkaVolterraFM`

## Finetune with simulator

Finetune the pretrained flow with the control network

``
python main.py --config=configs/experiments/control_signal/lv_differentiable_control.yaml --name=LotkaVolterraSimulator weight_file_pretrained=$PATH_TO_PRETRAINED_WEIGHTS
``

Make sure that `$PATH_TO_PRETRAINED_WEIGHTS` is absolute.
