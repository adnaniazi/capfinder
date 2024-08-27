### Clone this repository
```sh
git clone https://github.com/adnaniazi/capfinder.git
```
Next, `cd` into the clone repo.

### Creating dev enviornment
```sh
micromamba create -n capfinder_env python=3.12
micromamba activate capfinder_env
```

### Installation

First install `poetry`:

```sh
pip install poetry
```

Next install appropriate version of capfinder based on your hardware configuration:

=== "CPU"

    ```sh
    poetry install --extras cpu
    ```

=== "GPU (CUDA 12)"

    ```sh
    poetry install --extras gpu
    poetry run pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

=== "TPU"

    ```sh
    poetry install --extras tpu
    poetry run pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    ```

### Testing

```sh
pytest
```

### Documentation

The documentation is automatically generated from the content of the `docs` directory and from the docstrings
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
