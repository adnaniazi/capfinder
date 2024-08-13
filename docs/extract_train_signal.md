After synthesizing and sequencing your custom oligos, the next step is to collate BAM and POD5 files and extract the cap signal for the cap and flanking bases. This process is crucial for training the model to recognize new cap types.

### Signal Extraction Process

Capfinder extracts the signal for the cap and 5 flanking bases, which is then used to train the model. For example, with a Cap0-m6A oligo, Capfinder will extract the signal for `ACUGUm6A1N2N3N4N5N6` (i.e., the cap and flanking 5 bases).

This approach is consistent with the method used for cap 0, cap 1, cap 2, and cap 2-1 classes in the pretrained model.

### Command for Signal Extraction

To extract the cap signal, use the `capfinder extract-cap-signal` command. Here's an example command for our new class (Cap0-m6A):

```bash
capfinder extract-cap-signal \
    --bam_filepath /path/to/sorted.bam \
    --pod5_dir /path/to/pod5_dir \
    --reference GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGTANNNNNCGATGTAACTGGGACATGGTGAGCAATCAGGGAAAAAAAAAAAAAAA \
    --cap_class 4 \
    --cap_n1_pos0 52 \
    --train_or_test train \
    --output_dir /path/to/output_dir \
    --n_workers 10 \
    --no_plot_signal \
    --no-debug
```

#### Command Parameters Explained

- `--bam_filepath`: Path to the sorted BAM file containing aligned reads.
- `--pod5_dir`: Directory containing POD5 files with raw signal data.
- `--reference`: The reference sequence of the oligo. Note that `ANNNNN` represents the cap and variable bases.
- `--cap_class`: Integer representing the new cap class (4 in this example because we added this class label previously using `capmap add` command).
- `--cap_n1_pos0`: 0-based position of the first cap base `N1` in the reference sequence (52 in this case corresponding to the `A` of the `m6A` moeity).
- `--train_or_test`: Set to `train` for creating training data.
- `--output_dir`: Directory where the extracted data will be saved.
- `--n_workers`: Number of CPU cores to use for parallel processing.
- `--no-plot-signal`: Option to skip generating signal plots (speeds up processing).
- `--no-debug`: Disable debug mode for more detailed logging

## Output

The command will generate two main CSV files in the specified output directory:

1. `data__cap_0m6A.csv`: Contains the extracted ROI signal data.
2. `metadata__cap_0m6A.csv`: Contains complete metadata information.

Additionally, a log file (`capfinder_vXYZ_datetime.log`) will be created with program execution details.
