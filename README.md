# Capfinder - Advanced RNA Cap Type Prediction Framework


[![PyPI](https://img.shields.io/pypi/v/capfinder?style=flat-square)](https://pypi.python.org/pypi/capfinder/)
[![PyPi Downloads](https://img.shields.io/pypi/dm/capfinder)](https://pypistats.org/packages/capfinder)
[![CI/CD](https://github.com/adnaniazi/capfinder/actions/workflows/release.yml/badge.svg)](https://github.com/adnaniazi/capfinder/actions/workflows/release.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/capfinder?style=flat-square)](https://pypi.python.org/pypi/capfinder/)
[![PyPI - License](https://img.shields.io/pypi/l/capfinder?style=flat-square)](https://pypi.python.org/pypi/capfinder/)


---

**Documentation**: [https://adnaniazi.github.io/capfinder](https://adnaniazi.github.io/capfinder)

**PyPI**: [https://pypi.org/project/capfinder/](https://pypi.org/project/capfinder/)

---

Capfinder is a cutting-edge deep learning framework designed for accurate prediction of RNA cap types in mRNAs sequenced using Oxford Nanopore Technologies (ONT) SQK-RNA004 chemistry. By leveraging the power of native RNA sequencing data, Capfinder predicts the cap type on individual transcript molecules with high accuracy.

## Key Features

- **Pre-trained Model**: Ready-to-use classifier for immediate cap type prediction on ONT RNA-seq data.
- **Extensible Architecture**: Advanced users can train the classifier on additional cap classes, allowing for customization and expansion.
- **Comprehensive ML Pipeline**: Includes data preparation, hyperparameter tuning, and model training.
- **High Accuracy**: State-of-the-art performance in distinguishing between various cap types.
- **Flexibility**: Supports both CNN-LSTM, CNN-LSTML-Attention, ResNet, and Transformer-based Encoder model architectures.
- **Scalability**: Designed to efficiently handle large-scale RNA sequencing datasets.

## Supported Cap Types

Capfinder's pre-trained model offers accurate out-of-the-box predictions for the following RNA cap structures:

1. **Cap0**: Unmodified cap structure
2. **Cap1**: Methylated at the 2'-O position of the first nucleotide
3. **Cap2**: Methylated at the 2'-O position of the first and second nucleotides
4. **Cap2,-1**: Methylated at the 2'-O position of the first and second nucleotides, with an additional methylation at the -1 position

These cap types represent the most common modifications found in eukaryotic mRNAs. Capfinder's ability to distinguish between these structures enables researchers to gain valuable insights into RNA processing and regulation.

For advanced users, Capfinder's extensible architecture allows for training on additional cap types, expanding its capabilities to meet specific research needs.

---

# Pre-requisite for using Capfinder

The mRNA for Capfinder analysis must be prepared as follows:

1. Decap mRNA: Remove m7G from 5' end.
2. Ligate the following 52-nt OTE sequence to 5' end:

    ```sh
    5'-GCUUUCGUUCGUCUCCGGACUUAUCGCACCACCUAUCCAUCAUCAGUACUGU-3'
    ```

3. Sequence using ONT SQK-RNA004 chemistry.

These steps are crucial for accurate Capfinder predictions of mRNA cap types.

# Installing Capfinder
First create and acitvate a new Python environment and then install Capfinder as following:

1. For CPU

    ```
    pip install capfinder -U jax
    ```

2. For GPU

    ```
    pip install capfinder -U "jax[cuda12]"
    ```

3. For TPU

    ```
    pip install capfinder -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    ```

## Quick Start

The main command for cap type prediction is `predict-cap-types`:

```bash
capfinder predict-cap-types [OPTIONS]
```

### Example usage:

```bash
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

## Documentation
Please read the Capfinder's [comprehensive documentation](https://adnaniazi.github.io/capfinder/) for detailed information on using Capfinder.

## Contributing
Contributions to Capfinder are welcome! Please refer to the contribution guidelines for more information.

## License
Capfinder is released under the MIT License.

## Want to Collaborate
Please [contact](https://www.mn.uio.no/ibv/personer/vit/edv/) Eivind D. Valen at University of Oslo.
