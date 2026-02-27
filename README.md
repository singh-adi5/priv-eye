# Priv-Eye — ML-Driven Linux Privilege Posture Risk (Low / Medium / High)

Priv-Eye is a defensive Security + Machine Learning project that models **Linux privilege posture risk** using structured host telemetry.

The system collects kernel metadata, SUID attack surface indicators, and sudo configuration signals, then classifies a host into:

- LOW risk  
- MEDIUM risk  
- HIGH risk  

This project focuses on **practical ML security engineering**, not toy ML.

---

## Why This Matters (Real-World Security Context)

Privilege escalation in enterprise environments frequently stems from:

- Over-permissive sudo rules (`(ALL : ALL) ALL`, `NOPASSWD`)
- Large SUID attack surface (excess privileged binaries)
- Lab / pentest / unmanaged baseline hosts inside production networks
- Configuration drift across servers

Instead of signature detection, Priv-Eye models **posture risk** using engineered security features.

This aligns with:

- Hardening programs
- Baseline deviation monitoring
- Detection engineering support
- Internal red/blue validation
- Secure configuration auditing

---

## Project Architecture

         ┌──────────────────────┐
         │   recon_engine.py     │
         │  (telemetry capture)  │
         └─────────┬────────────┘
                   │
                   ▼
         raw_state.json  (local only; ignored by git)
                   │
                   ▼
         ┌──────────────────────┐
         │    ml_pipeline.py     │
         │ (feature engineering) │
         └─────────┬────────────┘
                   │
                   ▼
         feature_vector.csv  (local only; ignored by git)
                   │
                   ▼
         ┌──────────────────────────────┐
         │      train_model.py           │
         │  3-Class Random Forest Model  │
         │   LOW / MEDIUM / HIGH risk    │
         └─────────┬────────────────────┘
                   │
                   ▼
     Prediction + Probabilities + Feature Importances


---

## Feature Engineering

Current security features include:

### Kernel Signals
- `kernel_major`
- `kernel_minor`
- `kernel_patch`
- `kernel_flavor`

### SUID High-Sensitivity Indicators
- `pkexec`
- `su`
- `sudo`
- `mount`
- `passwd`
- `chsh`

### Aggregate Attack Surface
- `suid_total_count`

### Sudo Posture Signals
- `sudo_has_all`
- `sudo_has_nopasswd`

### Collection Integrity
- `euid_is_root`  
  (prevents label leakage / biased telemetry)

---

## Model Design

- Algorithm: Random Forest (interpretable, non-neural)
- Multi-class classification:
  - LOW
  - MEDIUM
  - HIGH
- Synthetic baseline training with enterprise-style constraints
- Feature importance reporting for explainability

This project prioritizes:

- Deterministic feature engineering
- Clear decision boundaries
- Interpretable outputs
- Security context mapping

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
