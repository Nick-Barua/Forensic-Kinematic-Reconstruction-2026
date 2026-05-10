# Pedestrian Impact Kinematic Reconstruction Framework (Phase 1)

<p align="left">
  <a href="https://doi.org/10.5281/zenodo.20096887">
    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20096887-blue.svg" alt="DOI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Status-Under%20Review-orange" alt="Status: Under Review">
  <img src="https://img.shields.io/badge/Phase-1%20Design-darkblue" alt="Phase 1 Design">
</p>

---

## 1. Overview

This repository provides the official Phase 1 technical implementation and numerical verification benchmarks for:

> **"A Physics-Grounded Multi-Modal Sensor Fusion Framework for Pedestrian Impact Kinematic Reconstruction Under Uncertainty: Phase 1 Design and Theoretical Evaluation"**
> Barua, N.; Hitosugi, M. — *Sensors* (MDPI), 2026 (Under Review)

The framework addresses forensic pedestrian–vehicle collision reconstruction as a **state-estimation problem**: recovering the dynamical trajectory of a physical system from incomplete, retrospective, noisy observables. It integrates 128-channel LiDAR, NIR stereo cameras, and a 2 kHz IMU via Kalman filtering and Savitzky–Golay differentiation, with end-to-end Monte Carlo uncertainty propagation.

### Graphical Abstract

![Graphical Abstract](graphical_abstract.png)
*Proposed bimodal reconstruction architecture integrating high-resolution spatial sensing, high-frequency inertial measurements, and ambient scene audio.*

---

## 2. Framework Architecture

The system utilises a three-stage processing pipeline to recover kinematic states and map them to physically interpretable biomechanical quantities.

| Figure 1: System Architecture | Figure 2: CNN–BiLSTM Design Specification |
| :---: | :---: |
| ![Figure 1](figure_1.png) | ![Figure 2](figure_2.png) |
| *Bimodal reconstruction architecture — Stage 1 and Stage 2 implemented in simulation.* | *Phase 1 design specification — training and evaluation pending FEA completion.* |

| Figure 3: Five-Phase Validation Roadmap | Figure 4: Phase 1 FEA Simulation Setup |
| :---: | :---: |
| ![Figure 3](figure_3.png) | ![Figure 4](figure_4.png) |
| *Evidence accumulates sequentially; no forensic deployment proposed until Phase 5 is complete.* | *THUMS v5 / LS-DYNA factorial scenario matrix — ≥150 primary runs.* |

---

## 3. Core Contributions

| Contribution | Description |
|---|---|
| **Uncertainty-Aware Integration** | Physics-constrained framework propagating sensor-level noise through kinematic reconstruction to biomechanical injury mapping via end-to-end Monte Carlo (n = 10,000; seed = 42) |
| **Noise-Robust Signal Recovery** | Mathematical justification of Savitzky–Golay differentiation over central finite differences — a 10% noise spike theoretically amplifies HIC₁₅ by 26.9% under the Wayne State 2.5 power-law exponent |
| **Vehicle-Class Parameterisation** | Inversion of throw-distance relationships using class-specific k-coefficients — a class-agnostic model produces +36.1 km/h systematic error for cab-over truck scenarios |
| **Bimodal Architecture** | Proposed integration of MFCC-CNN acoustic branch for pre-impact physiological state classification from CCTV/dashcam recordings (design specification; experimental validation reserved for Phase 3) |

---

## 4. Repository Contents

| File | Description |
|---|---|
| `reproducibility_benchmark.py` | Main simulation script — reproduces all Phase 1 numerical results |
| `requirements.txt` | Python dependencies |
| `CITATION.cff` | Machine-readable citation file |
| `LICENSE` | MIT Licence |
| `figure_1.png` | System architecture diagram |
| `figure_2.png` | CNN–BiLSTM design specification schematic |
| `figure_3.png` | Five-phase validation roadmap |
| `figure_4.png` | Phase 1 FEA simulation setup (THUMS v5 / LS-DYNA) |
| `figure_s1.png` | Supplementary Figure S1 — SG vs. CFD frequency response |
| `figure_s2.png` | Supplementary Figure S2 — Monte Carlo uncertainty propagation |
| `graphical_abstract.png` | Graphical abstract |

---

## 5. Reproducibility Guide

### Requirements

```bash
