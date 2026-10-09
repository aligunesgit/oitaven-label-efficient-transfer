# Architecture, target-budget and spatial robustness validation

Run `python -u watershed_validation/continue.py`. Do not start another instance. Progress: `controller_status.json`, `status.json`, `run.log`. Results: `results/REPORT.md`; final test: `results/final_test`.

Three predeclared spatial partitions, five training seeds, four architectures, target-label budgets100/300/700 plus source-only references.240 development results,200 unique frozen checkpoints evaluated in one final test batch. Spectral MLP uses the center spectrum; patch CNN uses a5-band patch without a separate spectral branch; joint CNN combines both. Random Forest uses center/mean/std from the same15×15 footprint.

Unlike the earlier pilot, no700-label validation set is used for CNN early stopping. Source40 epochs and target30 epochs are fixed; final checkpoints are selected. Budgets are actual total target training counts. Source weights are shared across partitions. Target IDs are nested random prefixes and shared by every architecture. Non-test reference samples are benchmark class-supported; this does not simulate blind field annotation.2000 development labels per partition are evaluation overhead.

Original512px test blocks and2000 test samples remain fixed. Only non-test256px blocks are redistributed, with32px buffers, using benchmark class-support information. All patches fit inside their assigned blocks. Source-only mean/std normalization remains fixed. Three partitions overlap and share one acquisition; they are not independent scenes. No second scene is available in the downloaded repository.

After development auditing, all checkpoint/protocol/test-file hashes are frozen before test labels are read. Every predeclared model is reported, with no model selection or fitting after test access. Absent test classes are explicitly unsupported; supported-class metrics supplement fixed-seven-class metrics. If test lacks a class, it cannot confirm general seven-class superiority. After this final batch the test is no longer an untouched resource for further tuning.

Source hashes, histories, resumable epoch checkpoints, models, sampled IDs, predictions and per-class support are preserved. An architecture comparison is not a capacity-matched causal proof; parameter counts are recorded. No new-method novelty or journal acceptance is assumed.
