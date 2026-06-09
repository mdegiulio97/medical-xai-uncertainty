# Model Card — cxr-xai-uncertainty

> Documento di trasparenza sul modello. Segue lo spirito di Mitchell et al., *Model Cards
> for Model Reporting* (FAccT 2019, DOI 10.1145/3287560.3287596).

## Dettagli del modello
- **Compito**: classificazione binaria (polmonite vs normale) su radiografie del torace.
- **Architettura**: ResNet-18 (pesi ImageNet pre-addestrati) con testa a 2 classi e
  dropout (p=0.3) attivo anche in inferenza per stima dell'incertezza (MC-Dropout).
- **Calibrazione**: temperature scaling post-hoc.
- **Spiegabilità**: Grad-CAM (Selvaraju et al., ICCV 2017, DOI 10.1109/ICCV.2017.74).
- **Config di training**: vedi `scripts/train.py` (default lr=1e-3, batch=128; epoche basse
  per esecuzione su CPU — è una demo, non un training ottimizzato).
- **Baseline di confronto**: regressione logistica sui pixel.

## Dati
- **PneumoniaMNIST** (MedMNIST v2, Yang et al., *Scientific Data* 2023,
  DOI 10.1038/s41597-022-01721-8), licenza **CC BY 4.0**.
- Immagini 28×28 in scala di grigi, **dataset benchmark di sanity-check**, NON immagini
  cliniche reali ad alta risoluzione (no DICOM, no artefatti reali, popolazione pediatrica).
- Scaricato a runtime da `scripts/download_data.py`; non incluso nel repository.

## Metriche (riproducibili, `results/metrics.json`, n_test=624)
| | Modello | Baseline logistico |
|---|---|---|
| AUROC | 0.963 | 0.927 |
| Accuracy @0.5 | 0.793 | 0.841 |
- Calibrazione **ECE 0.172 → 0.146** dopo temperature scaling (T=1.81).
- Incertezza media **3× più alta sugli errori** (0.0104) che sui corretti (0.0033).

## Uso previsto
- **Esclusivamente dimostrativo/didattico**: mostrare un pipeline ML clinico con incertezza,
  calibrazione e spiegabilità integrate.
- **NON è un dispositivo medico**, NON è validato clinicamente, NON deve essere usato per
  decisioni diagnostiche.

## Limiti e rischi
- Dataset giocattolo a bassa risoluzione → le prestazioni **non** si trasferiscono a CXR reali.
- A soglia 0.5 l'accuracy del deep model è sotto la baseline (sbilanciamento di classe,
  soglia non ottimizzata): il valore del modello è in **incertezza + calibrazione**, non
  nell'accuracy grezza.
- Nessuna analisi per sottogruppi (età/sesso/sito) → possibili bias non misurati.
- MC-Dropout è una stima approssimata dell'incertezza epistemica, non una garanzia.

## Come riprodurre
```
uv sync --extra dev
uv run python scripts/download_data.py
uv run python scripts/train.py
uv run python scripts/evaluate.py
```
