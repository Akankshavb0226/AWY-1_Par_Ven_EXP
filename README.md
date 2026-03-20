# AWY-1: Machine Learning Based Detection of Objects Moving Away from FIUS Sensor with Workspace Clearance Prediction

**Frankfurt University of Applied Sciences (FRA-UAS)**  
Course: Information Technology — Module: Individual Project  
Under the guidance of **Prof. Dr. Andreas Pech**

| Student | Matr. No. | Email |
|---|---|---|
| Akanksha Venkatesh Baitipuli | 1568316 | akanksha.venkatesh-baitipuli@stud.fra-uas.de |
| Ayush Naresh Parmar | 1565427 | ayush.parmar@stud.fra-uas.de |

**Project Duration:** 02 Dec 2025 – 20 Mar 2026

---

## Overview

AWY-1 is a machine learning pipeline built on real distance recordings from the **Frankfurt Intelligent Ultrasonic Sensor (FIUS)**. It goes beyond simple presence detection by:

1. **Classifying object motion** into three states — *Approaching*, *Moving Away*, or *Stationary* — using 300-reading sliding windows (~7.9 s at 38 Hz).
2. **Predicting Time-to-Clear (TtC)** — the number of seconds until an object exits the 0.90 m safety zone — for Away objects still inside the zone.

The primary focus is detecting that an object is **leaving** and estimating **how soon the zone will be free**, enabling downstream systems (e.g., collaborative robots, AGVs) to act proactively rather than waiting for a binary threshold to flip.

---

## Results Summary

| Model | Accuracy |
|---|---|
| Threshold-only baseline | 32.59% |
| Trend-based baseline | 57.78% |
| Logistic Regression | 67.41% |
| K-Nearest Neighbors | 71.11% |
| SVM (RBF kernel) | 66.67% |
| Decision Tree | 78.52% |
| **Random Forest** | **85.93% ✓** |

**TtC Regression (MAE):**

| Model | MAE (s) |
|---|---|
| Trend-based baseline | 313.04 |
| Linear Regression | 16.22 |
| **Random Forest** | **6.38** |

Random Forest met the ≥85% classification accuracy target and achieved the lowest TtC prediction error.

---

## Repository Structure

```
AWY-1/
│
├── awy1_project.py          # Main ML pipeline (training, evaluation, figures)
├── streamlit_app.py         # Interactive demo application
├── simulation.py            # Physics-based synthetic trajectory generator
├── requirements.txt         # Python dependencies
│
├── data/                    # FIUS CSV recordings (not included — see Data section)
│   ├── Day1/
│   └── Day2/
│
└── results/                 # Auto-generated on pipeline run
    ├── figures/             # All plots and confusion matrices
    ├── tables/              # Metric tables
    └── models/              # Saved .joblib model files
        ├── rf_classifier.joblib
        └── rf_regressor.joblib
```

---

## Setup

### Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd AWY-1

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` includes:

```
numpy>=1.24
pandas>=1.5
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
streamlit>=1.28
plotly>=5.18
joblib>=1.2
```

---

## Data

Data was collected using the FIUS sensor mounted horizontally in the laboratory, recording one distance measurement every 26.3 ms (38 Hz). Each recording is stored as a CSV file; **column index 10** contains the distance in metres.

### Dataset Composition

| Class | Total | Day 1 | Day 2 |
|---|---|---|---|
| Approaching | 15,988 | 9,055 | 6,933 |
| Away | 37,374 | 9,125 | 28,249 |
| Stationary | 21,550 | 12,397 | 9,153 |
| **Total** | **74,912** | **30,577** | **44,335** |

- **Day 1:** Single object (chair), single lighting condition
- **Day 2:** Three object types (human, chair, metal plate), two lighting conditions (daylight/night), three motion speeds (slow/medium/fast)

Place your CSV files in `data/Day1/` and `data/Day2/` before running the pipeline. Update the data path constants at the top of `awy1_project.py` if needed.

---

## Running the Pipeline

### Train and Evaluate All Models

```bash
python awy1_project.py
```

This will:
- Load and clean all CSV recordings
- Extract 16 features per 300-reading window
- Train all 5 classifiers + 2 regressors
- Print classification and regression metrics to the terminal
- Save all figures to `results/figures/`
- Save trained Random Forest models to `results/models/`

### Generate Simulation Samples (Optional)

```bash
python simulation.py
```

Generates synthetic Away / Approaching / Stationary trajectories and saves a sample plot to `results/figures/00_simulation_samples.png`.

---

## Interactive Demo (Streamlit App)

The demo loads the saved `rf_classifier.joblib` and `rf_regressor.joblib` and runs live inference in a simulated warehouse safety corridor.

```bash
streamlit run streamlit_app.py
```

### How it works

- Drag the distance slider to position the object between 0.05 m and 3.00 m.
- Each interaction adds 30 noisy readings to a rolling 300-reading buffer.
- Once 300 readings are available, the same 16 features used during training are extracted and passed to the classifier.
- The app displays **four live metrics**: current sensor reading, ML motion prediction, TtC estimate, and AGV decision state.

### AGV Decision States

| State | Condition |
|---|---|
| `WAITING` | Zone occupied, motion unclear |
| `WAIT — CLEARANCE PREDICTED` | Away detected, TtC available |
| `RESUME OPERATIONS` | Zone cleared (d ≥ 0.90 m) |
| `EMERGENCY HOLD` | Approaching detected |

> **Note:** The Streamlit app requires `rf_classifier.joblib` and `rf_regressor.joblib` to be present in `results/models/`. Run `awy1_project.py` first to generate them.

---

## System Pipeline

```
Raw CSV  →  Signal Cleaning  →  Feature Extraction  →  Labelling
                                                             ↓
                                          Train/Test Split (75/25 temporal)
                                                             ↓
                                         Model Training & Evaluation
                                                             ↓
                                       Saved .joblib Models + Streamlit Demo
```

### Key Configuration Constants

| Parameter | Value |
|---|---|
| Zone threshold | 0.90 m |
| Window size | 300 readings (~7.9 s) |
| Training stride | 50 readings |
| Test stride | 100 readings |
| Measurement rate | 38 Hz |
| Train/test split | 75% / 25% (temporal, per file) |

### Features (16 total)

| Group | Features |
|---|---|
| Statistical | mean, std, min, max, range |
| Kinematic | velocity, acceleration, net displacement, normalised velocity, trend noise |
| Directional | frac_increasing, frac_decreasing |
| Positional | start_dist, center_dist, end_dist, in_zone flag |

---

## Known Limitations

- **TtC MAE target not fully met:** Random Forest achieves 6.38 s MAE vs. the 0.6 s target. The 7.9 s observation window is too short to accurately predict clearance for objects still 60–120 s from the boundary.
- **Class imbalance:** Away class has 558 training windows vs. 238 for Approaching. No SMOTE or class weighting was applied.
- **Small regression dataset:** Only 169 regression training windows, limiting generalisation for long TtC values.
- **Zone threshold revised:** Original proposal specified 3.0 m; revised to 0.90 m due to laboratory space constraints.
- **Temporal leakage in regression:** Due to the small regression dataset, the 75/25 regression split is random rather than strictly temporal.

---

## References

1. J. Mainprice and D. Berenson, "Human-robot collaborative manipulation planning using early prediction of human motion," 2013.
2. T. Nesti, S. Boddana, and B. Yaman, "Ultra-sonic sensor based object detection for autonomous vehicles," CVPRW, 2023.
3. R. Tapu et al., "When ultrasonic sensors and computer vision join forces for efficient obstacle detection," Sensors, 2016.
4. D. Hunter, "Analysis of the measurement error from a low-cost ultrasonic sensor," Edward Waters University, 2023.
5. V. Murugesan et al., "Digital-twin evaluation for proactive human-robot collision avoidance," 2025.
6. S. Wang et al., "Denoising, outlier/dropout correction, and sensor selection in range-based positioning systems," IEEE Trans. Instrum. Meas., 2021.
7. M. E. Mohamed et al., "A machine learning approach for detecting ultrasonic echoes in noisy environments," VTC Spring, 2019.
8. M. E. Mohamed et al., "A CNN-based approach for ultrasonic noise suppression with minimal distortion," IEEE IUS, 2019.
