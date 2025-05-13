# Quantum‑Wavelength Correlation Research

This repository contains datasets and code produced for the **DOE‑funded InterQ‑net initiative**.  
Our goal is to determine whether **widely separated optical carriers (≈ 1350 nm and ≈ 1550 nm) preserve useful phase correlations when they co‑propagate in a single fiber**, informing future quantum‑network multiplexing designs.

---

## Repository Layout

```
quantum/
├── Data/
│   ├── S_params/          # Raw scattering‑parameter CSVs (S11, S21, …)
│   └── Angles/            # Same measurements converted to phase angles
│
├── Direct_prediction_*/   # Model‑centric experiments (one per dataset)
│   ├── a.ipynb            # Reproducible notebook with results & graphs
│   └── models/            # Saved PyTorch checkpoints
│
├── Conditional_recalibration/
│   └── a.ipynb            # Alternative calibration approach
│
├── requirements.txt       # Optional pip‑install spec
└── README.me              # You are here
```

### Why two *Data* folders?
Measurements are recorded as **S‑parameters** (common in microwave photonics).  
Phase‑angle views are easier for ML, so `utils.py` provides a safe conversion.

---

## Datasets

| Tag | Date (2024) | Span (nm) | Duration | Folder |
|-----|-------------|-----------|----------|--------|
| **Data 1** | Jun 29 – Jul 1 | 1550 ↔ 1350 | 48 h | `Direct_prediction_data1/` |
| **Data 2** | Mar 12 | 1550 ↔ 1350 | 3 h | `Direct_prediction_data2/` |
| **Data 3** | Nov (wk‑long) | 1550 ↔ 1350 | 168 h | `Direct_prediction_data3/` |
| **Data 4‑7** | Apr 7 | 1551.7 ± {+1…+10} | 3 h each | `Direct_prediction_1552…1561/` |
| **Data 8** | Apr 22 | 1551 ↔ 1531 | 3 h | `Direct_prediction_1531/` |

Each folder hosts a self‑contained `a.ipynb` that loads its slice, trains a small 1‑D CNN, and plots results.

---

## Quick‑Start

### 1. Clone & enter the repo

```bash
git clone https://github.com/HyouinSchoolAcc/quantum.git
cd quantum
```

### 2. Create an isolated environment (Conda recommended)

```bash
conda create -n quantum python=3.10 -y
conda activate quantum
```

### 3. Install dependencies

```bash
# strict conda channel install
conda install -y numpy pandas matplotlib pytorch torchvision torchaudio -c pytorch

# or via pip
pip install -r requirements.txt
```

### 4. Launch a notebook

```bash
jupyter lab
# open Direct_prediction_1552/a.ipynb and run all
```

AMP mixed‑precision (`torch.cuda.amp`) is enabled by default; disable if running on CPU.

---

## Experimental Highlights

* **Direct Prediction vs Conditional Recalibration**  
  Direct regression on raw S‑parameter tensors consistently outperforms the conditional approach when phase drift exceeds the training window.

* **Window‑Size Sensitivity**  
  See `window_size_comparison.png` for a quick sweep of MSE vs time‑window length.

---

## Contributing

Improved data loaders, new loss functions, or fresh wavelength regimes are welcome.  
Fork → Feature branch → Pull request.

---

## License

Unless noted otherwise, the repository is released under the **MIT License** (see `LICENSE`).

---

## Acknowledgements

Supported by the U.S. **Department of Energy, Office of Science, InterQ‑net Program** (Award DE‑SC002462*).  
Any opinions, findings, and conclusions are those of the authors and do not necessarily reflect the views of the DOE.
