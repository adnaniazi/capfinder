# Capfinder - A tool for mRNA Cap Type Prediction

[![PyPI](https://img.shields.io/pypi/v/capfinder?style=flat-square)](https://pypi.python.org/pypi/capfinder/)
[![PyPi Downloads](https://img.shields.io/pypi/dm/capfinder)](https://pypistats.org/packages/capfinder)
[![CI/CD](https://github.com/adnaniazi/capfinder/actions/workflows/release.yml/badge.svg)](https://github.com/adnaniazi/capfinder/actions/workflows/release.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/capfinder?style=flat-square)](https://pypi.python.org/pypi/capfinder/)
[![PyPI - License](https://img.shields.io/pypi/l/capfinder?style=flat-square)](https://pypi.python.org/pypi/capfinder/)

---

Capfinder is a specialized tool designed for predicting RNA cap types in mRNAs sequenced using Oxford Nanopore Technologies (ONT) SQK-RNA004 chemistry. By analyzing native RNA sequencing data, Capfinder can determine the cap structure of individual transcripts with high accuracy.

### Supported Cap Types

Capfinder currently supports the prediction of the following cap types:

 1. Cap0
 2. Cap1
 3. Cap2
 4. Cap2,-1

### mRNA Sample Preparation Requirements
To ensure optimal performance of Capfinder, mRNA samples must be prepared according to the following specifications:

- **Decapping:** The m7G moiety at the 5' end of the mRNA must be removed (decapping process).
- **Oligonucleotide Extension (OTE):** A specific 52-nucleotide sequence must be ligated to the 5' end of each mRNA molecule. The OTE sequence is as follows:
```sh
5'-GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT-3'
```
- **Sequencing:** Samples should be sequenced using Oxford Nanopore Technologies (ONT) SQK-RNA004 chemistry.
