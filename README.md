# Capfinder - A Tool for mRNA Cap type Prediction

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

Capfinder is a tool for predicting RNA cap types in mRNAs sequenced using Oxford Nanopore Technologies (ONT) RNA004 chemistry. It analyzes native RNA sequencing data to determine the cap structure of individual transcripts.

### Supported Cap Types
Currently, Capfinder can predict the following cap types:

- Cap0
- Cap1
- Cap2
- Cap2,-1

### Requirements for mRNA data
For Capfinder to work correctly, the  m7G moiety in mRNA samples must be first be removed from the 5' end of the mRNA (decapping).
The following 52-nucleotide oligonucleotide extension (OTE) must be ligated to the 5' end of each mRNA molecule:
```sh
5'-GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT-3'
```
---
# Installing Capfinder

### 1. Make and activate a new Python Environment
- [Creating new environment](docs/environment.md)



### 2. Install Capfinder package

- [Installation](docs/installation.md)

---
# Usage

### 1. Preprocessing: Basecalling and alignment

- [Data Preprocessing](docs/preprocessing.md)


### 2. Predicting Cap Types with Capfinder

- [Usage Guide](docs/prediction.md)


# Updating Capfinder

- [Updating Capfinder](docs/updating.md)


## Development

- [Development](docs/development.md)
