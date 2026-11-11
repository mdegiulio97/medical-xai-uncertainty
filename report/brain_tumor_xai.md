# Brain-tumour classification (glioma vs non-glioma) with explainability and uncertainty

Short report on the real-data track of this repository. The chest-track method (ResNet18 +
Grad-CAM + MC-Dropout uncertainty + calibration) is applied, **with the model unchanged**, to real
brain MRI.

## Data

**Cheng brain tumour dataset** (figshare), CC BY 4.0 [3]: 3064 contrast-enhanced T1w slices
from 233 patients, labelled meningioma, glioma or pituitary tumour. The task is
binary: **glioma vs non-glioma**. Images are normalised and resized to 3x64x64. Raw data are not
redistributed; `scripts/download_tumor.py` downloads all four parts and checks their MD5.

**Patient-level split.** Train / validation / test = 164 / 23 / 46 patients
(2145 / 306 / 613 slices; 62 / 9 / 18 patients with glioma), from a
stratified group K-fold with seed 0. No patient appears in more than one split. An earlier version
of this repository split by slice and used one quarter of the dataset; with that setup slices of the
same patient fell on both sides and the scores were not meaningful.

## Method

- **ResNet18** (ImageNet weights) with dropout, trained for 5 epochs, no early stopping.
- **Explainability**: Grad-CAM [1].
- **Uncertainty**: MC-Dropout [2] (20 samples; standard deviation as epistemic uncertainty).
- **Calibration**: post-hoc temperature scaling; **ECE** as the metric.
- **Baseline**: logistic regression on pixels.

## Results

From `results/tumor_metrics.json` (test = 613 slices from 46 unseen patients):

| | ResNet18 | Logistic baseline |
|---|---|---|
| **AUROC** | **0.961** | 0.881 |
| F1 | 0.572 | 0.774 |
| Accuracy @0.5 | 0.719 | 0.791 |

- **Calibration**: ECE **0.237 -> 0.070** with temperature scaling (T = 7.54).
- **Uncertainty**: mean **0.0056** on correct predictions vs **0.0135** on errors.

**Reading the numbers.** The deep model ranks slices much better than the baseline (AUROC), but at
the fixed 0.5 threshold its accuracy and F1 are below the baseline, and before calibration it is
strongly over-confident (large fitted temperature). The validation loss rises at the last of the
5 epochs, so a checkpoint chosen on validation loss would likely behave differently. Figures:
`results/tumor_reliability.png`, `results/tumor_gradcam.png`, `results/tumor_uncertainty.png`.

## Limitations

- **One split** with 18 glioma patients in the test set: wide uncertainty on every number.
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
