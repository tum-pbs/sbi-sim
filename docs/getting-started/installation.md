# Installation

This guide will install `sbisim` from the development branch of the GitHub repository.

## Prerequisites

Before installing SBI-SIM, make sure you have the following prerequisites:

- **Python 3.11+**: SBI-SIM requires Python 3.11 or higher
- **Git**: For cloning the repository
- **pip**: Python package installer

## Installation Steps

### 1. Clone the Repository

First, clone the `sbisim` repository from the `dev` branch:

```bash
git clone -b dev https://github.com/tum-pbs/sbi-sim.git
cd sbi-sim
```

### 2. Install `sbisim`

Install `sbisim` in development mode:

```bash
pip install -e .
```

The `-e` flag installs the package in "editable" mode, which means you can modify the source code and see changes immediately without reinstalling.

### Installing Specific Dependencies

If you encounter dependency conflicts, you can install specific versions:

```bash
# Install JAX first (if needed)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Then install sbisim
pip install -e .
```

## Verification

To verify that `sbisim` is installed correctly, run:

```python
import sbisim
print(f"sbisim version: {sbisim.__version__}")
```