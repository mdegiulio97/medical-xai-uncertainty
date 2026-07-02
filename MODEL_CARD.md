# Model Card — cxr-xai-uncertainty

> Documento di trasparenza sul modello, in italiano. Segue lo spirito di Mitchell et al.,
> *Model Cards for Model Reporting* (FAccT 2019, DOI 10.1145/3287560.3287596).

## Dettagli del modello
- **Autore**: Michele De Giulio, 2026.
- **Licenza del codice**: MIT (vedi `LICENSE`).
- **Compito**: classificazione binaria **pneumonia vs normal** su radiografie del torace.
- **Architettura**: ResNet-18 (pesi ImageNet pre-addestrati) con testa a 2 classi e
  **dropout (p=0.3)** mantenuto attivo anche in inferenza per la stima dell'incertezza
  epistemica via **MC-Dropout** (Gal & Ghahramani, ICML 2016, arXiv:1506.02142).
- **Calibrazione**: **temperature scaling** post-hoc a singolo parametro (Guo et al.,
  ICML 2017, arXiv:1706.04599).
- **Spiegabilità**: **Grad-CAM** (Selvaraju et al., ICCV 2017, DOI 10.1109/ICCV.2017.74).
- **Config di training**: vedi `scripts/train.py` (default lr=1e-3, batch=128; poche
  epoche per esecuzione anche su CPU/GPU entry-level — è una demo, non un training
  ottimizzato).
- **Baseline di confronto**: regressione logistica sui pixel.

## Uso previsto e usi fuori ambito
- **Uso previsto**: esclusivamente **dimostrativo e didattico**, per illustrare un pipeline
  di ML clinico che integra incertezza, calibrazione e spiegabilità e per confrontarlo con
  una baseline lineare.
- **Usi fuori ambito**: **NON è un dispositivo medico**, **NON** è validato clinicamente e
  **NON** deve essere usato per screening, diagnosi, triage reale o qualsiasi decisione che
  riguardi pazienti. Non è destinato a immagini cliniche reali (DICOM, alta risoluzione,
  artefatti di acquisizione) né a popolazioni diverse da quella del dataset di benchmark.

## Dati
- **PneumoniaMNIST** (MedMNIST v2, Yang et al., *Scientific Data* 10:41, 2023,
  DOI 10.1038/s41597-022-01721-8), licenza **CC BY 4.0**, download automatico senza
  registrazione tramite il pacchetto `medmnist`.
- **Radiografie pediatriche** del torace, classi *normal* / *pneumonia*.
- Immagini **ridimensionate a 64×64** e **replicate a 3 canali** per il backbone ResNet-18.
- Dataset benchmark standardizzato per sanity-check riproducibile: **non** immagini cliniche
  reali ad alta risoluzione.
- Scaricato a runtime da `scripts/download_data.py`; non incluso nel repository.

## Metriche (riproducibili, `results/metrics.json`, n_test = 624)
| Metrica | ResNet-18 (deep) | Baseline logistica |
|---|:--:|:--:|
| AUROC ↑ | **0.963** | 0.927 |
| F1 ↑ | 0.858 | 0.886 |
| Accuracy @0.5 ↑ | 0.793 | 0.841 |

- **Calibrazione**: ECE **0.172 → 0.146** dopo temperature scaling (**T = 1.81**).
- **Incertezza (MC-Dropout)**: media **0.0104** sugli errori vs **0.0033** sulle predizioni
  corrette → incertezza **~3× più alta dove il modello sbaglia**, utile come **flag di triage**.
- **Lettura onesta**: il modello profondo ha AUROC nettamente superiore (miglior ranking),
  ma alla soglia fissa 0.5 accuracy e F1 sono leggermente sotto la baseline lineare, perché
  la soglia non è ottimizzata e le classi sono sbilanciate. Il valore distintivo è il segnale
  di incertezza e la calibrazione, non lo squeezing dell'accuracy.

## Considerazioni etiche e limiti
- **Popolazione pediatrica**: i dati sono radiografie di bambini; le prestazioni non sono
  rappresentative di adulti né di altri contesti clinici.
- **Bassa risoluzione**: le immagini 64×64 sono standardizzate; le prestazioni **non** si
  trasferiscono a CXR reali ad alta risoluzione.
- **Nessun uso diagnostico**: qualsiasi impiego per decisioni cliniche è escluso; l'output
  non è un referto.
- **Incertezza come flag di triage**, non come garanzia: MC-Dropout stima l'incertezza
  *epistemica* in modo approssimato e non copre l'incertezza aleatoria del dato.
- **Nessuna analisi per sottogruppi** (età, sesso, sito di acquisizione): possibili bias non
  misurati.
- **Soglia non ottimizzata**: metriche a soglia 0.5, sensibili allo sbilanciamento di classe.

## Caveat
- I numeri dipendono da seed, numero di epoche e ambiente; rigenerabili con
  `scripts/evaluate.py`.
- Il pesi del modello e i dati non sono versionati nel repository: vanno rigenerati
  localmente seguendo la procedura sotto.
- Model card e metriche descrivono il track **chest X-ray** (PneumoniaMNIST); il track MRI
  cerebrale (Cheng, glioma vs non-glioma) è documentato separatamente in
  `report/brain_tumor_xai.md`.

## Come riprodurre
```
uv sync --extra dev
uv run python scripts/download_data.py
uv run python scripts/train.py
uv run python scripts/evaluate.py
```
