# Model card: medical-xai-uncertainty

> Transparency document for the models in this repository, following Mitchell et al.,
> *Model Cards for Model Reporting* (FAccT 2019, DOI 10.1145/3287560.3287596).

## Model details
- **Author**: Michele Maria De Giulio, 2026.
- **Code license**: MIT (see `LICENSE`).
- **Tasks**: binary classification of (a) **pneumonia vs normal** on paediatric chest X-rays and
  (b) **glioma vs non-glioma** on contrast-enhanced T1w brain MRI slices.
- **Architecture**: ResNet-18 (ImageNet weights) with a 2-class head and **dropout (p=0.3)** kept
  active at inference to estimate epistemic uncertainty via **MC-Dropout** (Gal & Ghahramani,
  ICML 2016, arXiv:1506.02142).
- **Calibration**: post-hoc single-parameter **temperature scaling** (Guo et al., ICML 2017,
  arXiv:1706.04599).
- **Explainability**: **Grad-CAM** (Selvaraju et al., ICCV 2017, DOI 10.1109/ICCV.2017.74).
- **Training**: see `scripts/train.py` and `scripts/evaluate_tumor.py` (few epochs, no early
  stopping: a demonstration, not an optimised training).
- **Baseline**: logistic regression on pixels.

## Intended use and out-of-scope uses
- **Intended use**: **demonstration and teaching** only, to show a clinical-ML pipeline that
  combines uncertainty, calibration and explainability, compared with a linear baseline.
- **Out of scope**: this is **NOT a medical device**, it is **NOT** clinically validated and must
  **NOT** be used for screening, diagnosis, triage or any decision about patients.

## Data
- **PneumoniaMNIST** (MedMNIST v2, Yang et al., *Scientific Data* 10:41, 2023,
  DOI 10.1038/s41597-022-01721-8), CC BY 4.0: paediatric chest X-rays, resized to 64x64 and
  replicated to 3 channels; official MedMNIST split.
- **Cheng brain tumour dataset** (figshare, DOI 10.6084/m9.figshare.1512427), CC BY 4.0:
  3064 contrast-enhanced T1w slices from 233 patient IDs (209 patients after grouping), resized to 64x64.
  **Split by patient**, with IDs differing only by a trailing letter grouped together (train/val/test = 148/20/41 patients), stratified, seed 0.
- Both datasets are downloaded by script and are not included in the repository.

## Metrics
Chest X-ray (`results/metrics.json`, n_test = 624):

| Metric | ResNet-18 (deep) | Logistic baseline |
|---|:--:|:--:|
| AUROC | **0.963** | 0.927 |
| F1 | 0.858 | 0.886 |
| Accuracy @0.5 | 0.793 | 0.841 |

ECE 0.172 -> 0.146 (T = 1.81); mean uncertainty 0.0033 (correct) vs 0.0104 (wrong).

Brain MRI (`results/tumor_metrics.json`, n_test = 613 slices from 41 unseen patients, 13 with glioma):

| Metric | ResNet-18 (deep) | Logistic baseline |
|---|:--:|:--:|
| AUROC | **0.942** | 0.846 |
| F1 | 0.876 | 0.744 |
| Accuracy @0.5 | 0.883 | 0.759 |

ECE 0.076 -> 0.028 (T = 2.50); mean uncertainty 0.0048 (correct) vs 0.0190 (wrong).

Chest track: the deep model ranks better (AUROC) while accuracy and F1 at the fixed 0.5
threshold are below the linear baseline. Brain track: The deep model is better than the baseline on every metric here, including accuracy and F1 at the fixed 0.5 threshold, and calibration reduces the ECE. Training runs a fixed 5 epochs without early stopping, with one seed and one split, so the threshold-dependent metrics are not stable from run to run.

## Ethical considerations and limitations
- **Chest track**: paediatric population, low resolution; results do not transfer to adults or
  to real high-resolution X-rays.
- **Brain track**: 2D slices at 64x64, a single patient-level split, short fixed training; wide
  uncertainty on every number.
- **No diagnostic use**: the output is not a report.
- **Uncertainty as a triage flag, not a guarantee**: MC-Dropout approximates epistemic
  uncertainty only.
- **No subgroup analysis** (age, sex, site): possible unmeasured bias.
- **Threshold not tuned**: metrics at 0.5 are sensitive to class imbalance.

## How to reproduce
```
uv sync --extra dev
uv run python scripts/download_data.py && uv run python scripts/train.py && uv run python scripts/evaluate.py
uv run python scripts/download_tumor.py && uv run python scripts/evaluate_tumor.py
```
