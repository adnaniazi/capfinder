### Sequencing Kit

The pretrained classifier has been trained using data generated from the SQK-RNA004 kit. This kit offers significantly improved data quality compared to its predecessor, SQK-RNA002. To ensure compatibility and optimal performance:

- **Required Kit**: Use the SQK-RNA004 kit for sequencing your newly designed synthetic oligos.
- **Rationale**: The superior data quality of SQK-RNA004 is crucial for accurate cap type classification.

### Read Depth

To achieve robust and reliable learning outcomes:

- **Minimum Read Count**: Acquire at least 4 million reads.
- **Recommendation**: More reads generally lead to better model performance. If resources allow, consider generating more than the minimum requirement.

### Data Processing

For proper preparation of your sequencing data:

1. **Basecalling**: Convert raw signal data to nucleotide sequences.
2. **Alignment**: Map the basecalled reads to your reference sequence.

Detailed instructions for these steps can be found in the [Preprocessing](preprocessing.md) section of our documentation.

> **Note**: Adhering to these requirements ensures that your new data is consistent with data the pretrained model has been trained on.
