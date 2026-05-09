# Pedestrian Impact Kinematic Reconstruction Framework (Phase 1)

## 1. Overview
This repository provides the official Phase 1 technical implementation and numerical verification benchmarks for the research: **"A Physics-Grounded Multi-Modal Sensor Fusion Framework for Pedestrian Impact Kinematic Reconstruction Under Uncertainty: Phase 1 Design and Theoretical Evaluation"**.

### Graphical Abstract
![Graphical Abstract](graphical_abstract.png)
*Figure GA: Proposed bimodal reconstruction architecture integrating high-resolution spatial sensing, high-frequency inertial measurements, and ambient scene audio.*

---

## 2. Framework Architecture & Roadmap
The system utilizes a three-stage processing pipeline to recover kinematic states and map them to physically interpretable biomechanical quantities.

| Figure 1: System Architecture | Figure 2: Validation Roadmap |
| :---: | :---: |
| ![Figure 1](figure_1.png) | ![Figure 2](figure_2.png) |
| *Bimodal reconstruction architecture.* | *Five-phase validation roadmap.* |

---

## 3. Core Contributions
* **Uncertainty-Aware Integration:** A unified framework for propagating sensor-level noise through kinematic state-estimation and biomechanical injury mapping.
* **Noise-Robust Signal Recovery:** Mathematical justification of Savitzky–Golay (SG) differentiation for suppressing noise amplification in power-law transformations.
* **Vehicle-Class Parameterisation:** Inversion of throw-distance relationships using class-specific coefficients to mitigate systematic reconstruction errors.

## 4. Repository Contents
* **`reproducibility_benchmark.py`**: High-precision Python script to reproduce core mathematical results.
* **Monte Carlo Engine:** Executes 10,000 draws to propagate throw-distance uncertainty.
* **DSP Verification:** Generates frequency response analysis for SG filters vs. central finite differences.
* **Biomechanical Mapping:** Implements Mertz–Prasad lognormal probit and Viano hepatic stress transformations.

## 5. Reproducibility Guide

### Requirements
* Python 3.8+
* `numpy`, `scipy`, `matplotlib`

### Execution
To verify the Phase 1 theoretical results and generate benchmarking plots:
`python reproducibility_benchmark.py`

### Expected Numerical Outputs (Phase 1 Benchmark)
* **Impact Velocity (v):** 49.3 ± 1.16 km/h (95% CI: [47.0, 51.6])
* **Hepatic Stress (σ_liver):** 283 ± 6.7 kPa (95% CI: [270, 296])
* **Injury Risk (P(AIS ≥ 4)):** ≈ 0.40

---

## 6. Technical Benchmarks

| Figure S1: DSP Proof | Figure S2: Uncertainty Propagation |
| :---: | :---: |
| ![Figure S1](figure_s1.png) | ![Figure S2](figure_s2.png) |
| *Verification of SG noise suppression.* | *End-to-end Monte Carlo results.* |

---

## 7. Academic Status & Citation
This research is currently **submitted** to *Sensors* for peer review.

> Barua, N.; Hitosugi, M. A Physics-Grounded Multi-Modal Sensor Fusion Framework for Pedestrian Impact Kinematic Reconstruction Under Uncertainty: Phase 1 Design and Theoretical Evaluation. Sensors 2026 (Submitted).

## 8. Intellectual Property & License
* **Code License:** This project is licensed under the **MIT License**.
* **Patents:** Aspects of this framework are protected under Japanese Patent Application **No. 2025-167440** (filed 3 October 2025).
* **Disclosures:** N.B. is a co-author of the MFCC-CNN architecture (Reference [6]) adapted for the acoustic branch of this framework.

## 9. Acknowledgments
Supported by the Department of Legal Medicine, Shiga University of Medical Science.
