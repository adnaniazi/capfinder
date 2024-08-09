Before installing capfinder, please make a fresh conda/micromamba env with required supported Python versions like so:
```sh
micromamba create -n capfinder_env python=3.12
```
Here we have created a Python 3.12 environment using micromamba. Next, we activate the newly created conda env:
```sh
micromamba activate capfinder_env
```

In the activated environment, Capfinder can be installed with support for different hardware configurations. Choose the appropriate installation tab based on your hardware setup. This ensures you get the right dependencies for optimal performance on your system:

=== "CPU-only"

    ```
    pip install capfinder -U jax
    ```

    This installation is suitable when you only have CPU and not GPUs.

=== "GPU (CUDA 12)"

    ```
    pip install capfinder -U "jax[cuda12]"
    ```

    Use this installation for systems with NVIDIA GPUs supporting CUDA 12. Capfinder depends on JAX internally for using GPUs. JAX requires CUDA to work. CUDA requirements for Capfinder are the same as the CUDA requirements for JAX.

    For more information on the required CUDA version for JAX, refer to the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

=== "TPU"

    ```
    pip install capfinder -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    ```

    This installation is for systems with TPU (Tensor Processing Unit) hardware.

!!! note

    Make sure to choose the installation command that matches your hardware configuration for optimal performance.

!!! failure "What if you get this error"
In case you encounter the following error:

´´´
ImportError: /tmp/pip-build-env-qsdot3t6/overlay/lib/python3.12/site-packages/Cython/Utils.cp
note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed building wheel for pysam

Failed to build pysam

ERROR: Could not build wheels for pysam, which is required to install pyproject.toml-based projects
´´´

You should first do the following in terminal:

´´´
mkdir "/path/to/temp/directory"
chmod +x "/path/to/temp/directory"
export TMPDIR="/path/to/temp/directory"
´´´

Now try to reinstall `capfinder`

!!! failure "What if you get this error"
In case you encounter the following error:

´´´
note: use option -std=c99 or -std=gnu99 to compile your code
error: command '/usr/bin/gcc' failed with exit code 1
[end of output]

note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed building wheel for pysam
Failed to build pysam
ERROR: Could not build wheels for pysam, which is required to install pyproject.toml-based projects
´´´

You should first do the following in terminal:

´´´
export CFLAGS="-std=c99 -D_GNU_SOURCE $CFLAGS"
export LDFLAGS="-std=c99 $LDFLAGS"
´´´

Now try to reinstall `capfinder`
