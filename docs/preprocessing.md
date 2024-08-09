Capfinder requires detailed information about base movements during sequencing, which is captured in the "moves table". This information is crucial for accurate cap type prediction.
#### Requirements

- Raw POD5 files from your sequencing run
- Dorado basecaller (version 0.7.0 or later recommended)
- Samtools (version 1.18 or later recommended)
- A transcriptome reference

#### Important Note

Dorado Live Basecalling does not generate the necessary moves information. Therefore, post-data acquisition standalone basecalling and alignment process is required to produce the moves information in the alignment SAM/BAM file.

#### Basecalling Process

To generate the required data, we use a custom script that performs the following steps:

1. Basecalls the raw POD5 files using Dorado

2. Emits SAM format output with moves information

3. Aligns the basecalled reads to a reference genome

4. Converts the output to a sorted and indexed BAM file

#### Basecalling Script

Below is a script that automates this process. You'll need to adjust the paths and settings to match your environment:

```bash
#!/bin/bash
# Please edit to reflect your settings
DORADO="/path/to/bin/dorado"
SAMTOOLS="/path/to/samtools"
POD5_DIR="/path/to/pod5_dir"
REF="/path/to/ref.fa"
MODEL_NAME="rna004_130bps_sup@v5.0.0"
MODEL_DIR="/path/to/download/doardo/model"
OUTPUT_DIR="/path/to/save/basecalled/data"
DEVICE="cuda:all" # For Dorado to use GPU

#---------------------- DO NOT EDIT BELOW THIS LINE ---------------------#

# Function to check and download the model if necessary
check_and_download_model() {
    local model_path="$MODEL_DIR/$MODEL_NAME"
    if [ ! -d "$model_path" ]; then
        echo "Model not found. Downloading..."
        "$DORADO" download --directory "$MODEL_DIR" --model "$MODEL_NAME"
    else
        echo "Model already exists. Skipping download."
    fi
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check and download the model if necessary
check_and_download_model

# Run dorado basecaller and pipe directly to samtools for sorting and indexing
echo "Starting basecalling and BAM creation..."
"$DORADO" basecaller "$MODEL_DIR/$MODEL_NAME" "$POD5_DIR/" \
    --recursive \
    --emit-sam \
    --emit-moves \
    --device "$DEVICE" \
    --reference "$REF" | \
"$SAMTOOLS" view -bS - | \
"$SAMTOOLS" sort -o "$OUTPUT_DIR/sorted.bam" -

# Index the sorted BAM file
echo "Indexing the sorted BAM file..."
"$SAMTOOLS" index "$OUTPUT_DIR/sorted.bam"

echo "Basecalling and processing completed. Output files are in $OUTPUT_DIR"
echo "Generated files: sorted.bam and sorted.bam.bai"
```

To use this script:

1. Save it to a file (e.g., `basecall.sh`).
2. Make it executable:
   ```bash
   chmod +x basecall.sh
   ```
3. Edit the script directly to change any paths or settings as needed.
4. Run the script:
   ```bash
   ./basecall.sh
   ```
