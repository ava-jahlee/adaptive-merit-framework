# Adaptive Merit Framework (AMF)

A merit-anchored fairness mechanism for educational admissions that integrates SES-based corrections while preserving performance thresholds.

## Paper

**Adaptive Merit Framework: Merit-Anchored Fairness via SES Correction**

- arXiv: [Link will be added after submission]
- Author: Jung-Ah Lee (ava.jahlee@gmail.com)

## Overview

AMF introduces a two-phase selection mechanism that:
1. Preserves regular admits (those who exceed raw score thresholds)
2. Enables threshold-based conditional admission for high-potential students from disadvantaged backgrounds

The correction formula:

```
M_i* = M_i + alpha * (0.5 - S_i)
```

Where:
- `M_i`: Raw score
- `S_i`: SES percentile (0-1)
- `alpha`: Correction strength parameter
- `M_i*`: Corrected score

## Repository Structure

```
adaptive-merit-framework/
├── README.md
├── LICENSE
├── requirements.txt
└── scripts/
    ├── amf_engine.py           # Core AMF implementation
    ├── run_amf_all_results.py  # Main analysis script
    ├── amf_weighted_analysis.py # Population-weighted estimation
    └── pisa2022_korea.csv      # PISA 2022 Korea data
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ava-jahlee/adaptive-merit-framework.git
cd adaptive-merit-framework

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Complete Analysis

```bash
cd scripts
python run_amf_all_results.py
```

This generates:
- PISA-based AMF simulation results
- Synthetic population Monte Carlo simulations
- Robustness checks (SES noise, score variance, threshold sensitivity)
- Dynamic Bayesian Network trajectory analysis

### Run Weighted Population Estimation

```bash
cd scripts
python amf_weighted_analysis.py
```

This generates population-weighted estimates using PISA sampling weights.

## Generated Output Files

| File | Description |
|------|-------------|
| `pisa_amf_summary.csv` | Summary statistics for alpha = 5, 10, 15 |
| `pisa_additional_alpha*.csv` | Detailed conditional admits per alpha |
| `synthetic_mc_summary.csv` | Monte Carlo simulation summary |
| `robustness_*.csv` | Robustness check results |
| `dbn_*.csv` | Long-run trajectory analysis |

## Data

- **Source**: OECD PISA 2022 Korea
- **URL**: https://www.oecd.org/pisa/data/2022database/
- **Sample**: N = 6,377 (14 outliers removed via 1.5x IQR rule)
- **Variables**:
  - PV1MATH to PV10MATH (plausible values for mathematics) - available in the raw PISA dataset.
  - ESCS (Economic, Social, and Cultural Status index)
  - W_FSTUWT (sampling weights)
 
Note on plausible values and actual analysis
- Although the PISA dataset provides PV1MATH–PV10MATH, this repository's analyses (as implemented in the scripts) use PV1MATH only for the primary score variable. The reason for using PV1MATH in the current analysis is to preserve consistency with the originally reported results and to keep the processing pipeline straightforward and reproducible for readers.
- If you prefer to account for plausible-value uncertainty in downstream inference (recommended for more rigorous variance estimation), one can:
  - (Simple) use the mean of PV1–PV10 as a single score per student, or
  - (Recommended) run analyses separately for each PV and combine estimates using multiple-imputation-style combining rules (Rubin's rules) to correctly reflect between-PV variance.
- The codebase includes clear places to extend the pipeline to a full PV-based combining approach; see comments in `scripts/amf_weighted_analysis.py` and `scripts/amf_engine.py` for where to implement PV looping and combination.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scipy
- seaborn

## Runtime

Expected runtime: ~2-3 minutes on a standard laptop.

## License

- Code: MIT License
- Paper: CC BY 4.0

## Citation

If you use this code, please cite

## Contact

For questions or collaboration inquiries: ava.jahlee@gmail.com

