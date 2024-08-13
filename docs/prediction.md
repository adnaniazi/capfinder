Capfinder provides a command-line interface to predict RNA cap types using BAM and POD5 files. Here's how to use the `predict-cap-types` function:

#### Usage

```bash
capfinder predict-cap-types [OPTIONS]
```

#### Description

This command predicts RNA cap types using BAM and POD5 files.

#### Required Options

- `--bam_filepath` or `-b`: Path to the BAM file generated using the preprocessing step

- `--pod5_dir` or `-p`: Path to directory containing POD5 files

- `--output_dir` or `-o`: Path to the output directory for prediction results and logs

#### Additional Options

- `--n_cpus` or `-n`: Number of CPUs to use for parallel processing. Default is 1

:   Multiple CPUs are used during processing for POD5 file and BAM data (Step 1/5). Increasing this number speeds up POD5 and BAM processing. For inference (Step 4/5), only a single CPU is used no matter how many CPUs you have specified. For faster inference, have a GPU available (it will be detected automatically) and set dtype to `float16`


- `--dtype` or `-d`: Data type for model input. Valid values are `float16`, `float32`, or `float64`. Default is `float16`

:   Without a GPU, use `float32` or `float64` for better performance. If you have a GPU, then use `float16` for faster inference

- `--batch_size` or `-bs`: Batch size for model inference. Default is `128`

:   Larger batch sizes can speed up inference but require more memory. If the code crashes during step 4/5, you have probably set too high a batch size.

- `--plot-signal` / `--no-plot-signal`: Whether to plot extracted cap signal or not. Default is `--no-plot-signal`

- `--custom-model-path` or `-m`: Path to a custom model (.keras) file. If not provided, the default pre-packaged model will be used.

:   Saving plots can help you plot the read's signal, and plot the signal for cap and flanking bases(&#177;5).

- `--debug` / `--no-debug`: Enable debug mode for more detailed logging. Default is `--no-debug`

:   The option can prints which function is creating a particular log output. This is helpful during code debugging.

- `--refresh-cache` / `--no-refresh-cache`: Refresh the cache for intermediate results. Default is `--no-refresh-cache`

:   If you input data has changed (for example you added one more POD5 file in your POD5 directory) then you must use `--refresh-cache` to compute all steps again and not load them from cache that hold results from your previous run.

- `--help`: Show the help message and exit


!!! example

    ```
    capfinder predict-cap-types \
        --bam_filepath /path/to/sorted.bam \
        --pod5_dir /path/to/pod5_dir \
        --output_dir /path/to/output_dir \
        --n_cpus 100 \
        --dtype float16 \
        --batch_size 256 \
        --no_plot_signal \
        --no-debug \
        --no-refresh-cache
    ```

!!! tip "Tips"

    1. **CPU Usage**:

        - Increase `--n_cpus` for faster processing of POD5 and BAM data
        - CPU count doesn't affect inference speed (Step 4/5)

    2. **GPU Acceleration**:

        - If you have a GPU, use `--dtype float16` for faster inference
        - Without a GPU, `float32` or `float64` may perform better

    3. **Batch Size**:

        - Larger batch sizes can speed up inference but require more memory
        - Adjust `--batch_size` based on your system's capabilities

    4. **Plotting**:

        - Use `--no-plot-signal` to skip signal plotting for faster processing

    5. **Debugging**:

        - Enable `--debug` for detailed logging when troubleshooting

    6. **Caching**:

        - Use `--refresh-cache` if you've made changes to input data and need to regenerate intermediate results

    For more detailed information, run `capfinder predict-cap-types --help`.
