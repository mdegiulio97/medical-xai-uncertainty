# Brain-tumour classification (glioma vs non-glioma) with explainability and uncertainty

Short report on the real-data track of this repository. The chest-track method (ResNet18 +
Grad-CAM + MC-Dropout uncertainty + calibration) is applied, **with the model unchanged**, to real
brain MRI.

## Data

**Cheng brain tumour dataset** (figshare), CC BY 4.0 [3]: 3064 contrast-enhanced T1w slices
from 233 patient IDs (209 patients after grouping), labelled meningioma, glioma or pituitary tumour. The task is
binary: **glioma vs non-glioma**. Images are normalised and resized to 3x64x64. Raw data are not
redistributed; `scripts/download_tumor.py` downloads all four parts and checks their MD5.

**Patient-level split.** Train / validation / test = 148 / 20 / 41 patients
(2144 / 307 / 613 slices; 46 / 6 / 13 patients with glioma), from a
stratified group K-fold with seed 0. No patient appears in more than one split; patient IDs that differ only by a trailing letter (for example MR040240 and MR040240B, separate series of the same patient) are grouped together. Grouping is by ID only, so a patient listed under unrelated IDs would not be caught. An earlier version
of this repository split by slice and used one quarter of the dataset; with that setup slices of the
same patient fell on both sides and the scores were not meaningful.

## Method

- **ResNet18** (ImageNet weights) with dropout, trained for 5 epochs, no early stopping.
- **Explainability**: Grad-CAM [1].
- **Uncertainty**: MC-Dropout [2] (20 samples; standard deviation as epistemic uncertainty).
- **Calibration**: post-hoc temperature scaling; **ECE** as the metric.
- **Baseline**: logistic regression on pixels.

## Results

From `results/tumor_metrics.json` (test = 613 slices from 41 unseen patients):

| | ResNet18 | Logistic baseline |
|---|---|---|
| **AUROC** | **0.942** | 0.846 |
| F1 | 0.876 | 0.744 |
| Accuracy @0.5 | 0.883 | 0.759 |

- **Calibration**: ECE **0.076 -> 0.028** with temperature scaling (T = 2.50).
- **Uncertainty**: mean **0.0048** on correct predictions vs **0.0190** on errors.

**Reading the numbers.** The deep model is better than the baseline on every metric here, including accuracy and F1 at the fixed 0.5 threshold, and calibration reduces the ECE. Training runs a fixed 5 epochs without early stopping, with one seed and one split, so the threshold-dependent metrics are not stable from run to run. Figures:
`results/tumor_reliability.png`, `results/tumor_gradcam.png`, `results/tumor_uncertainty.png`.

## Limitations

- **One split** with 13 glioma patients in the test set: wide uncertainty on every number.
- **Simplified binary task** (glioma vs the rest); the dataset has 3 classes.
- 2D slices resized to 64x64: resolution and 3D context are lost.
- **Not a clinical tool**: research demonstration on public CC BY data.

## References

1. Selvaraju RR, Cogswell M, Das A, et al. *Grad-CAM: Visual Explanations from Deep Networks
   via Gradient-based Localization.* ICCV 2017. DOI:
   [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)
2. Gal Y, Ghahramani Z. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in
   Deep Learning.* ICML 2016. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)
3. Cheng J, et al. *Brain tumor dataset.* figshare. DOI:
   [10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427) (CC BY 4.0)
