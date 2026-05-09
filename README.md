# Pedestrian Impact Kinematic Reconstruction Framework (Phase 1)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository provides the official Phase 1 technical implementation and numerical verification benchmarks for the research: **"A Physics-Grounded Multi-Modal Sensor Fusion Framework for Pedestrian Impact Kinematic Reconstruction Under Uncertainty: Phase 1 Design and Theoretical Evaluation"**.

### Graphical Abstract
![Graphical Abstract](assets/graphical_abstract.png) 
*Proposed bimodal reconstruction architecture integrating LiDAR, NIR stereo, and IMU data.*

## 1. Research Objectives
Forensic reconstruction of pedestrian–vehicle collisions is a state-estimation problem: recovering a dynamical trajectory from incomplete, noisy, and retrospective observables. This framework integrates 128-channel LiDAR, NIR stereo cameras, and a 2 kHz IMU via asynchronous Kalman filtering and noise-robust Savitzky–Golay (SG) polynomial differentiation.

### Core Contributions:
* **Uncertainty-Aware Integration:** A unified framework for propagating sensor-level noise through kinematic state-estimation and biomechanical injury mapping.
* **Noise-Robust Signal Recovery:** Mathematical justification of SG differentiation for suppressing high-frequency noise amplification in power-law transformations like the Head Injury Criterion (HIC).
* **Vehicle-Class Parameterisation:** Inversion of throw-distance relationships using class-specific coefficients to mitigate systematic reconstruction errors.

## 2. Repository Contents
* `reproducibility_benchmark.py`: High-precision Python script to reproduce the core mathematical results.
    * **Monte Carlo Engine:** Executes 10,000 draws to propagate throw-distance uncertainty.
    * **DSP Verification:** Generates frequency response analysis for SG filters vs. central finite differences.
    * **Biomechanical Mapping:** Implements the Mertz–Prasad lognormal probit model and Viano hepatic stress transformations.

## 3. Reproducibility Guide

### Requirements
* Python 3.8+
* `numpy`, `scipy`, `matplotlib`

### Execution
To verify the Phase 1 theoretical results and generate benchmarking plots:
```bash
python reproducibility_benchmark.py
