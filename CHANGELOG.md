# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/adnaniazi/capfinder/compare/0.2.6...master
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

