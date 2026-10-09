# Oitavén: Label-Efficient Regional Transfer

Research code by **Ali Güneş**, Istanbul Atlas University. ORCID: [0000-0003-3116-1184](https://orcid.org/0000-0003-3116-1184).

This repository accompanies the manuscript **İHA Multispektral Görüntülerinde Az Etiketli Bölgesel Aktarımın Mekânsal Değerlendirilmesi** (*Spatial Evaluation of Regional Transfer with Limited Labels in UAV Multispectral Imagery*). The manuscript is a draft; no journal publication or acceptance is claimed.

## Main study

`watershed_validation/` is the experiment reported in the manuscript:

- Three spatial partitions, five training seeds.
- Spectral MLP, patch CNN, joint spectral–spatial CNN and Random Forest.
- Source-only references and nested target-training budgets of 100,300,700 labels.
- Fixed40 source epochs and30 target fine-tuning epochs; no extra target early-stopping labels.
- Source-only normalization, matched target sample IDs and buffered spatial blocks.
- 240 development conditions and200 unique checkpoints in a predeclared final test batch.

`watershed_transfer/` and `watershed_experiments/` preserve earlier exploratory comparisons. They are not the principal manuscript results. Their conclusions and annotation accounting differ; see each protocol.

## Installation

Python3.9 was used for the original runs. Install in a separate environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

CNN execution uses Apple MPS when available and CPU otherwise. Current runners do not select CUDA automatically. Seeds and numeric results can vary with hardware/library differences; do not assume bitwise-identical trained weights.

## Data and preparation

The raw dataset is maintained by the High-Performance Computing for Remote Sensing Laboratory at CiTIUS:

[Domain Adaptation Dataset](https://gitlab.citius.gal/HPC4RS/DomainAdaptationDataset)

Pinned raw-data commit: `e17fff566e4f77bf6b4d7cd3fa995e25627c177f`. Original data is CC BY4.0. Raw images, model weights, environments and manuscript Word/PDF files are not distributed here.

```bash
python bootstrap.py --download
```

If the dataset is already downloaded, place `DomainAdaptationDataset/` beside `bootstrap.py` and run `python bootstrap.py`. The bootstrap checks raw-file SHA-256 hashes and reconstructs arrays from `frozen_sampling/`. These fixtures contain sample coordinates,block assignments and evaluation sampling weights, not image bands or class-label arrays. Labels are reconstructed locally from the released reference maps. This avoids requiring local diagnostic artifacts from the original workspace and avoids solver-dependent changes to published partitions.

The shared source preparation,15×15 patches and source normalization are reconstructed before experiments. Frozen sampling metadata is derivative dataset documentation and retains CC BY4.0 attribution; the Python code is MIT-licensed.

## Run the manuscript experiment

```bash
python watershed_validation/test_integrity.py
python -u watershed_validation/continue.py
```

Do not start two controllers. The controller validates preparation, trains and resumes epoch checkpoints, audits development predictions, freezes every model hash, and runs the final test batch. Progress is written to `watershed_validation/controller_status.json` and `status.json`; generated reports are in `watershed_validation/results/`.

Running the full controller includes the already reported test evaluation. If you plan new tuning, use only development stages and obtain a new independent test; the published test is not an untouched resource.

## Results and limitations

`reference_results/watershed_validation/` contains the original lightweight metric tables and audit receipts. With700 target labels, development macro-F1 was0.6246 for patch CNN,0.6242 for joint CNN and0.5652 for Random Forest. The extra spectral branch did not provide a clear average benefit.

All partitions share one acquisition and can overlap. Non-test benchmark labels were used to enforce class support in partitions/reference pools. The final2000-sample test has199 water,10 concrete and1791 vegetation samples; four classes are absent. Supported-class test macro-F1 and mIoU do not confirm independent seven-class superiority. Concrete recall is low. No speed,ecological-cost or new-method superiority claim is made.

## Related work and citation

Dataset publications:

- Argüello et al. (2021), *Watershed Monitoring in Galicia from UAV Multispectral Imagery Using Advanced Texture Methods*. [DOI](https://doi.org/10.3390/rs13142687).
- Garea et al. (2023), *A hybrid CUDA,OpenMP,and MPI parallel TCA-based domain adaptation for classification of very high-resolution remote sensing images*. [DOI](https://doi.org/10.1007/s11227-022-04961-y).

The author's related active-learning study is [AdaRisk-Agent](https://doi.org/10.3390/drones10070547). The main comparison in this repository does not use active learning,LLMs or asymmetric cost coefficients. Cite this software using `CITATION.cff`; a manuscript DOI will be added only if published.
