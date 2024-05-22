# Inverse Kinematics Benchmark for Inverse Modelling

This repository hosts the forward problem and solution methods for the inverse problem.

The code builds on https://github.com/vislearn/inn_toy_data/.

The benchmark was proposed first in the following paper:

Kruse, J., Ardizzone, L., Rother, C., & Köthe, U. (2021). Benchmarking Invertible Architectures on Inverse Problems (arXiv:2101.10763). arXiv. https://doi.org/10.48550/arXiv.2101.10763

## Development

We manage dependencies in `pyproject.toml` and lock them in `requirements.txt` using `pip-compile` from the [pip-tools suite](https://github.com/jazzband/pip-tools). Automated formatting and checks are achieved using `pre-commit`.

TODO: Insert all installation steps with corresponding commands.

For development:

1. Set up a new Python environment
2. Install dependencies using `pip install -r requirements.txt`
3. Activate pre-commit hooks using `pre-commit install`

For adding a dependency:

1. Add it to `pyproject.toml`
2. Run `pip-compile`
3. Install the updated `requirements.txt` using `pip install -r requirements.txt`
