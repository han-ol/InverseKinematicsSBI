 # Inverse Kinematics Benchmark for Simulation-based Inference

Welcome to the repository of the inverse_kinematics_sbi package.  TODO: package name
This is a lightweight library enabling you to flexibly specify a robot arm, compute the forward process, and solve the inverse kinematic problem.

Inverse kinematics was first proposed as a benchmark for simulation-based inference in the following paper:

Kruse, J., Ardizzone, L., Rother, C., & Köthe, U. (2021). Benchmarking Invertible Architectures on Inverse Problems (arXiv:2101.10763). arXiv. https://doi.org/10.48550/arXiv.2101.10763

(~~the code builds on https://github.com/vislearn/inn_toy_data/.~~  TODO: not at the moment. mention related work appropriately)

## Development

We manage dependencies in `pyproject.toml` and lock them in `requirements.txt` using `pip-compile` from the [pip-tools suite](https://github.com/jazzband/pip-tools). Automated formatting and checks are achieved using `pre-commit`.


### Get started

1. Clone and enter the repository
```
git clone https://codeberg.org/han-ol/InverseKinematicsSBI.git && cd InverseKinematicsSBI
```
2. Create and activate an empty python environment, for example with `conda`
```
conda create -n ik-sbi python=3.10 && conda activate ik-sbi
```
3. Install dependencies with `pip`
```
pip install -r requirements.txt
pip install pre-commit  # TODO: as an optional dependency for development
```
4. Activate pre-commit hooks using
```
pre-commit install
```

TODO: Insert installation steps for pip-tools or optional dependency for development

### Add a dependency

1. Add it to `pyproject.toml`
2. Run `pip-compile`
3. Install the updated `requirements.txt` using `pip install -r requirements.txt`
