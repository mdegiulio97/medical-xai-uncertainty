# Classificazione di tumore cerebrale (glioma vs non-glioma) con spiegabilità e incertezza

Mini-report del *track dati reali* di `cxr-xai-uncertainty`. La stessa metodologia del
track chest (ResNet18 + Grad-CAM + incertezza MC-Dropout + calibrazione) viene applicata,
**senza modifiche al modello**, a MRI cerebrale reale.

## Dati

**Cheng brain tumor dataset** (figshare), licenza **CC BY 4.0** [3]. Slice T1w con mezzo di
contrasto, etichettate in 3 classi (meningioma, glioma, adenoma ipofisario). Si usa la
**parte 1/4** pubblica (766 slice) e si imposta un task **binario clinicamente rilevante**:
**glioma** (la classe maligna) **vs non-glioma**. Le immagini sono normalizzate e portate al
formato della pipeline (3×64×64), così da riusare modello, Grad-CAM, incertezza e
calibrazione. I dati grezzi non sono ridistribuiti: si scaricano con `scripts/download_tumor.py`.

## Metodo

- **ResNet18** (pre-addestrato ImageNet) con dropout, addestrato 5 epoche.
- **Spiegabilità**: Grad-CAM [1].
- **Incertezza**: MC-Dropout [2] (20 campioni; deviazione standard come incertezza epistemica).
- **Calibrazione**: temperature scaling post-hoc; **ECE** come metrica.
- **Baseline**: regressione logistica sui pixel.

## Risultati

Numeri reali da `results/tumor_metrics.json` (test = 154 slice held-out):

| | ResNet18 | Baseline logistica |
|---|---|---|
| **AUROC** | **0.980** | 0.859 |
| F1 | 0.621 | 0.571 |
| Accuracy @0.5 | 0.929 | 0.961 |

- **Calibrazione**: ECE **0.039 → 0.013** con temperature scaling (T = 1.24).
- **Incertezza**: media **0.0127** sulle predizioni corrette vs **0.0573** sugli errori
  (**~4.5× più alta sugli errori**) — l'incertezza MC-Dropout è un segnale utile per
  l'abstention/triage.

**Lettura onesta.** Come nel track chest, il modello profondo domina in **AUROC** (0.980 vs
0.859) ma a soglia fissa 0.5 la sua accuracy è leggermente sotto la baseline (0.929 vs 0.961):
la classe glioma è minoritaria e la soglia non è ottimizzata (F1 modesto). Il valore del
modello sta in **capacità di ranking + incertezza + calibrazione**, non nell'accuracy grezza.
Figure: `results/tumor_reliability.png`, `results/tumor_gradcam.png`, `results/tumor_uncertainty.png`.

## Limiti

- **Sottoinsieme** del dataset (766/3064 slice, una parte) e **un solo split**: demo, non studio.
- **Task binario semplificato** (glioma vs resto); il dataset è in origine a 3 classi.
- Slice 2D ridimensionate a 64×64: si perde risoluzione e contesto 3D.
- **Non è uno strumento clinico**: dimostrazione di ricerca su dati pubblici CC BY.

## Riferimenti

1. Selvaraju RR, Cogswell M, Das A, et al. *Grad-CAM: Visual Explanations from Deep Networks
   via Gradient-based Localization.* ICCV 2017. DOI:
   [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)
2. Gal Y, Ghahramani Z. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in
   Deep Learning.* ICML 2016. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)
3. Cheng J, et al. *Brain tumor dataset.* figshare. DOI:
   [10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427) (CC BY 4.0)
