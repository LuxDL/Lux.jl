# Changelog

All notable changes to this project since the release of v1 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2024-07-28

### Fixed

  - Tracker gradients with ComponentArrays. (#24)

## [1.1.0] - 2024-07-28

### Added

  - `@test_softfail` macro marks a test as broken if it fails else it passes. (#23)
  - `soft_fail` kwarg introdced in `test_gradients` to mark a test as broken if it
    fails. (#23)

### Changed

  - `skip_backends` use `skip` kwarg in `@test` macro and show up as broken in the test
    summary. (#23)
  - If `Enzyme.jl` fails to load, then Enzyme tests will be skipped. (#23)

## [1.0.1] - 2024-07-27

### Fixed

  - GPU device detection in `test_gradients`.
