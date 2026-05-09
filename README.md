# Pedestrian Impact Kinematic Reconstruction Framework (Phase 1)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Overview
This repository provides the official Phase 1 technical implementation and numerical verification benchmarks for the research: **"A Physics-Grounded Multi-Modal Sensor Fusion Framework for Pedestrian Impact Kinematic Reconstruction Under Uncertainty: Phase 1 Design and Theoretical Evaluation"**.

### Graphical Abstract
![Graphical Abstract](assets/graphical_abstract.png)
*Figure GA: Proposed bimodal reconstruction architecture integrating high-resolution spatial sensing, high-frequency inertial measurements, and ambient scene audio.*

---

## 2. Framework Architecture & Roadmap
The system utilizes a three-stage processing pipeline to recover kinematic states and map them to physically interpretable biomechanical quantities.

| Figure 1: System Architecture | Figure 2: Validation Roadmap |
| :---: | :---: |
| ![Figure 1](assets/figure_1.png) | ![Figure 2](assets/figure_2.png) |
| *Bimodal reconstruction architecture from Stage 1 (Fusion) to Stage 3 (Classification).* | *Five-phase progression from FEA simulation to independent multi-site replication.* |

---

## 3. Core Contributions
* **Uncertainty-Aware Integration:** A unified framework for propagating sensor-level noise through kinematic state-estimation and biomechanical injury mapping.
* **Noise-Robust Signal Recovery:** Mathematical justification of Savitzky–Golay (SG) differentiation for suppressing noise amplification in power-law transformations like the Head Injury Criterion (HIC).
* **Vehicle-Class Parameterisation:** Inversion of throw-distance relationships using class-specific coefficients to mitigate systematic reconstruction errors.

## 4. Repository Contents
* `reproducibility_benchmark.py`: High-precision Python script to reproduce core mathematical results.
    * **Monte Carlo Engine:** Executes 10,000 draws to propagate throw-distance uncertainty.
    * **DSP Verification:** Generates frequency response analysis for SG filters vs. central finite differences.
    * **Biomechanical Mapping:** Implements Mertz–Prasad lognormal probit and Viano hepatic stress transformations.

## 5. Reproducibility Guide

### Requirements
* Python 3.8+
* `numpy`, `scipy`, `matplotlib`

### Execution
To verify the Phase 1 theoretical results and generate benchmarking plots:
```bash
python reproducibility_benchmark.py
