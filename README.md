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
pip install -r requirements.txt
```

- Python 3.8+
- `numpy`, `scipy`, `matplotlib`

### Execution

```bash
python reproducibility_benchmark.py
```

### Expected Numerical Outputs

All outputs are verified against the manuscript (Tables 4–8):

| Output | Value | 95% CI |
|---|---|---|
| Impact velocity | 49.3 ± 1.16 km/h | [47.0, 51.6] |
| Hepatic stress σ_liver | 283 ± 6.7 kPa | [270, 296] |
| HIC₁₅ | 820 | [727, 917] |
| P(AIS ≥ 4, cranial) | ≈ 0.40 | [0.28, 0.52] |

Simulation specification: d ~ N(14.2, 0.5²) m · k = 0.041 (bonnet-type sedan, fixed) · seed = 42 · n = 10,000 · covariance = diagonal

---

## 6. Technical Benchmarks

| Figure S1: DSP Verification | Figure S2: Monte Carlo Uncertainty Propagation |
| :---: | :---: |
| ![Figure S1](figure_s1.png) | ![Figure S2](figure_s2.png) |
| *Savitzky–Golay noise suppression advantage over central finite differences — ~15 dB in the 10–30 Hz band.* | *End-to-end propagation from throw-distance measurement error through velocity, HIC₁₅, σ_liver, and P(AIS≥4).* |

---

## 7. Monte Carlo Simulation Specification

As reported in manuscript Table 3:

| Parameter | Distribution / Value | Basis |
|---|---|---|
| Throw distance *d* | N(14.2, 0.5²) m | ±0.5 m tape-measurement uncertainty |
| Vehicle coefficient *k* | 0.041 (fixed) | Deterministic bonnet-type sedan |
| Random seed | 42 | Computational reproducibility |
| Covariance | Diagonal | Zero correlation between inputs |
| *n* | 10,000 draws | — |

---

## 8. Academic Status & Citation

This research is currently **under review** at *Sensors* (MDPI).

**BibTeX:**
```bibtex
@article{barua2026forensic,
  author    = {Barua, Nick and Hitosugi, Masahito},
  title     = {A Physics-Grounded Multi-Modal Sensor Fusion Framework for
               Pedestrian Impact Kinematic Reconstruction Under Uncertainty:
               Phase 1 Design and Theoretical Evaluation},
  journal   = {Sensors},
  publisher = {MDPI},
  year      = {2026},
  note      = {Under review},
  doi       = {10.5281/zenodo.20096887}
}
```

---

## 9. Intellectual Property & Disclosures

- **Code Licence:** MIT — see `LICENSE`
- **Patent:** Aspects of this framework are covered under Japanese Patent Application **No. 2025-167440** (filed 3 October 2025, Japan Patent Office)
- **Coauthorship disclosure:** N.B. is a co-author of the MFCC-CNN architecture (Rezaul et al. 2024, Reference [6]) on which the acoustic branch of this framework is based; this overlap is disclosed in the interest of transparency

---

## 10. Validation Roadmap

This repository covers **Phase 1 only**. No forensic deployment is proposed until Phase 5 is complete.

| Phase | Setting | Target N | Status |
|---|---|---|---|
| 1 — Simulation & Algorithm | THUMS v5 / LS-DYNA | ≥150 scenarios | 🔄 In progress |
| 2 — Instrumented Tests | Crash dummy trials | ≥50 trials | ⏳ Pending |
| 3 — Real-World Data | CCTV/dashcam (IRB) | ≥30 cases | ⏳ Pending |
| 4 — Clinical Validation | IRB-approved cohort | ≥100 cases | ⏳ Pending |
| 5 — Deployment Readiness | Multi-site replication | ≥200 cases | ⏳ Pending |

---

## 11. Acknowledgments

Supported by the Department of Legal Medicine, Shiga University of Medical Science, Otsu, Shiga, Japan.
