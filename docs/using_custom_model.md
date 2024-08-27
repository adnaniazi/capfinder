Custom trained models can be used with `custom_model_path` parameter and specifiying the path to the custom trained model.

!!! example

    ```sh hl_lines="8"
    capfinder predict-cap-types \
        --bam_filepath /path/to/sorted.bam \
        --pod5_dir /path/to/pod5_dir \
        --output_dir /path/to/output_dir \
        --n_cpus 100 \
        --dtype float16 \
        --batch_size 256 \
        --custom_model_path /path/to/.keras/file/for/custom/trained/model \
        --no_plot_signal \
        --no-debug \
        --no-refresh-cache
    ```

!!! note

    Before running the above command, please ensure that the new cap type is present in the cap map. Please read more about it in [Extending cap mapping](../extend_capmap) section of the documentation.
