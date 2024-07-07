# capfinder

[![PyPI](https://img.shields.io/pypi/v/capfinder?style=flat-square)](https://pypi.python.org/pypi/capfinder/)
[![PyPi Downloads](https://img.shields.io/pypi/dm/capfinder)](https://pypistats.org/packages/capfinder)
[![CI/CD](https://github.com/adnaniazi/capfinder/actions/workflows/release.yml/badge.svg)](https://github.com/adnaniazi/capfinder/actions/workflows/release.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/capfinder?style=flat-square)](https://pypi.python.org/pypi/capfinder/)
[![PyPI - License](https://img.shields.io/pypi/l/capfinder?style=flat-square)](https://pypi.python.org/pypi/capfinder/)


---

**Documentation**: [https://adnaniazi.github.io/capfinder](https://adnaniazi.github.io/capfinder)

**Source Code**: [https://github.com/adnaniazi/capfinder](https://github.com/adnaniazi/capfinder)

**PyPI**: [https://pypi.org/project/capfinder/](https://pypi.org/project/capfinder/)

---

A package for decoding RNA cap types

## Installation

### CPU installation
```sh
pip install capfinder[cpu]
```

### GPU installation (CUDA 12)
```sh
pip install capfinder[gpu] "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### TPU installation
```sh
pip install capfinder[tpu] "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```



## Development

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.7+
* Create a virtual environment and install the dependencies

### CPU installation
```sh
poetry install --extras cpu
```

### GPU installation (CUDA 12)
```sh
poetry install --extras gpu
poetry run pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### TPU installation
```sh
poetry install --extras tpu
poetry run pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

* Activate the virtual environment

```sh
poetry shell
```

### Testing

```sh
pytest
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

### Releasing

Trigger the [Draft release workflow](https://github.com/adnaniazi/capfinder/actions/workflows/draft_release.yml)
(press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the
[GitHub releases](https://github.com/adnaniazi/capfinder/releases) and publish it. When
 a release is published, it'll trigger [release](https://github.com/adnaniazi/capfinder/blob/master/.github/workflows/release.yml) workflow which creates PyPI
 release and deploys updated documentation.

### Pre-commit

Pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `flake8`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```
