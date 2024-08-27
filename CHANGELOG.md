# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2024-08-27
### Fixed
- Fixed cli arguments
- Fixed readme and documentation errors.

## [0.4.1] - 2024-08-23
### Added
- Option to use time-warped augmented data during training

### Fixed
- Padding and truncation algorithm is now fixed such that equal amounts of time samples are padded or truncated from both ends of the classifier examples

## [0.4.0] - 2024-08-19
### Fixed
- Fixed bugs that caused slow training pipeline
- Fixed logic for uploading large dataset objects to Comet ML in small chunks

## [0.3.9] - 2024-08-16
### Fixed
- Fixed a bug where all the class data files were not being used to make the dataset

## [0.3.8] - 2024-08-16
### Fixed
- Fixed a bug where if the dataset dir had no dataset previously, new dataset was not being created.
- Increased CSV field size in train_etl to fix capfinder crashing during when encoutering large csv fields

### Added
- Added logging info to standalone train ETL pipeline

## [0.3.7] - 2024-08-14
### Fixed
- Fixed bugs in collate function that caused hogging of memory for large BAM file

## [0.3.6] - 2024-08-13
### Fixed
- Fixed missing headers from some commands in the cli

### Added
- Added option to specify custom models during inference
- Added more documentation

## [0.3.5] - 2024-08-11
### Fixed
- Fixed string formatting issue

## [0.3.4] - 2024-08-11
### Added
- Added ability to add more cap types to training
- Added a new train ETL pipeline that can handle larger than memory datasets
- Added improved interface for training pipeline

### Fixed
- Old CLI app to reflect changes in API

## [0.3.3] - 2024-08-08
### Fixed
- Issues with pip installation by removing the yanked package (types-pkg-resources)

## [0.3.2] - 2024-08-08
### Fixed
- Issues with pip installation

## [0.3.1] - 2024-08-08
### Fixed
- Issues with pip installation

## [0.3.0] - 2024-08-08
### Fixed
- Issues with pip installation

## [0.2.9] - 2024-08-02
### Fixed
- Fixed issue with batch inference not working

## [0.2.8] - 2024-08-02
### Fixed
- Fixed loading the entire inference dataset in memory

### Added
- Added more information to the README file

## [0.2.7] - 2024-08-01
### Fixed
- Problems with cli not displaying capfinder version info
- Fixed issue with API docs generation
- Added max_examples parameter to ETL to limit the number of examples to process in a dataset (use during training)

## [0.2.6] - 2024-08-01
### Fixed
- Slow report generation
- Problems with refreshing of cache

## [0.2.5] - 2024-07-31
### Added
- Report generation
- Cli for inference

## [0.2.4] - 2024-07-26
### Added
- Added functions for performaing inference

## [0.2.3] - 2024-07-17
### Added
- Cosine annealing cyclic learning rate scheduler with resets, decay, and progressive lengthing of cycles

## [0.2.2] - 2024-07-15
### Added
- Added Cyclical learning rate scheduler

### Removed
- Removed reduce learning rate on plateau

## [0.2.1] - 2024-07-11
### Added
- Added an attention-augmented CNN-LSTM model

## [0.2.0] - 2024-07-10
### Fixed
- Fixed issue with ml_libs module not found

## [0.1.9] - 2024-07-10
### Fixed
- Fixing issue with ml_libs module not found

## [0.1.8] - 2024-07-10
### Added
- Added support for resnet model

## [0.1.7] - 2024-07-08
### Fixed
- Fixed encoder model hogging all available GPU memory and crashing

## [0.1.6] - 2024-07-07
### Fixed
- Fixed bug with not writing dataset version to the file

## [0.1.5] - 2024-07-07
### Fixed
- Bugs with using the train config functionality

## [0.1.4] - 2024-07-07
### Added
- Added pipeline for making training data
- Added CNN-LSTM and encoder models

## [0.1.3] - 2023-09-23
### Added
- Functions to align OTE with reads in FASTQ
- Function for pretty printing alignment for debugging purposes
- Function for finding the start and end location of ROI in training datasets

## [0.1.2] - 2023-08-18
### Changed
- Changed function arg names again for testing version bump

## [0.1.1] - 2023-08-16
### Changed
- Changed function arg names for testing version bump

## [0.1.0] - 2023-08-15
### Added
- Basic skeleton of the package and tested it

[Unreleased]: https://github.com/adnaniazi/capfinder/compare/0.4.2...master
[0.4.2]: https://github.com/adnaniazi/capfinder/compare/0.4.1...0.4.2
[0.4.1]: https://github.com/adnaniazi/capfinder/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/adnaniazi/capfinder/compare/0.3.9...0.4.0
[0.3.9]: https://github.com/adnaniazi/capfinder/compare/0.3.8...0.3.9
[0.3.8]: https://github.com/adnaniazi/capfinder/compare/0.3.7...0.3.8
[0.3.7]: https://github.com/adnaniazi/capfinder/compare/0.3.6...0.3.7
[0.3.6]: https://github.com/adnaniazi/capfinder/compare/0.3.5...0.3.6
[0.3.5]: https://github.com/adnaniazi/capfinder/compare/0.3.4...0.3.5
[0.3.4]: https://github.com/adnaniazi/capfinder/compare/0.3.3...0.3.4
[0.3.3]: https://github.com/adnaniazi/capfinder/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/adnaniazi/capfinder/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/adnaniazi/capfinder/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/adnaniazi/capfinder/compare/0.2.9...0.3.0
[0.2.9]: https://github.com/adnaniazi/capfinder/compare/0.2.8...0.2.9
[0.2.8]: https://github.com/adnaniazi/capfinder/compare/0.2.7...0.2.8
[0.2.7]: https://github.com/adnaniazi/capfinder/compare/0.2.6...0.2.7
[0.2.6]: https://github.com/adnaniazi/capfinder/compare/0.2.5...0.2.6
[0.2.5]: https://github.com/adnaniazi/capfinder/compare/0.2.4...0.2.5
[0.2.4]: https://github.com/adnaniazi/capfinder/compare/0.2.3...0.2.4
[0.2.3]: https://github.com/adnaniazi/capfinder/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/adnaniazi/capfinder/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/adnaniazi/capfinder/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/adnaniazi/capfinder/compare/0.1.9...0.2.0
[0.1.9]: https://github.com/adnaniazi/capfinder/compare/0.1.8...0.1.9
[0.1.8]: https://github.com/adnaniazi/capfinder/compare/0.1.7...0.1.8
[0.1.7]: https://github.com/adnaniazi/capfinder/compare/0.1.6...0.1.7
[0.1.6]: https://github.com/adnaniazi/capfinder/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/adnaniazi/capfinder/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/adnaniazi/capfinder/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/adnaniazi/capfinder/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/adnaniazi/capfinder/compare/0.1.1...0.1.2
[0.1.1]: https://pypi.org/manage/project/capfinder/release/0.1.1/
[0.1.0]: https://pypi.org/manage/project/capfinder/release/0.1.0/

