# Architecture, budget and spatial-partition validation

Three spatial partitions × five training seeds × three budgets × four architectures. Source pretraining40 epochs; target fine-tuning30 epochs. No target validation/early-stopping labels. Budgets100/300/700 are the total target training labels.2000 development references per partition are evaluation overhead. Nested sample IDs are matched across architectures. Same source normalization and patch footprint.

patch_cnn removes the explicit center-spectrum branch but still sees all five spectral bands. Parameter counts are recorded; architectures are not capacity matched. Non-test benchmark labels enforce class-supported partitions; this is not a label-free deployment sampler.

Spatial partitions reuse one scene and can overlap across repetitions; do not treat the15 split/seed combinations as15 independent scenes. Reserved test is read only after all models are frozen.

| Architecture | Target labels | Runs | Macro-F1 | mIoU | Accuracy |
|---|---:|---:|---:|---:|---:|
| joint_cnn | 0 | 15 | 0.3793 | 0.3059 | 0.6707 |
| joint_cnn | 100 | 15 | 0.4978 | 0.4130 | 0.8047 |
| joint_cnn | 300 | 15 | 0.5854 | 0.4874 | 0.8280 |
| joint_cnn | 700 | 15 | 0.6242 | 0.5290 | 0.8511 |
| patch_cnn | 0 | 15 | 0.3857 | 0.3114 | 0.6763 |
| patch_cnn | 100 | 15 | 0.4848 | 0.3997 | 0.8018 |
| patch_cnn | 300 | 15 | 0.5752 | 0.4758 | 0.8229 |
| patch_cnn | 700 | 15 | 0.6246 | 0.5305 | 0.8502 |
| random_forest | 0 | 15 | 0.4444 | 0.3470 | 0.7440 |
| random_forest | 100 | 15 | 0.4988 | 0.3977 | 0.7875 |
| random_forest | 300 | 15 | 0.5395 | 0.4331 | 0.8128 |
| random_forest | 700 | 15 | 0.5652 | 0.4573 | 0.8338 |
| spectral_mlp | 0 | 15 | 0.3778 | 0.2890 | 0.6220 |
| spectral_mlp | 100 | 15 | 0.4438 | 0.3487 | 0.6797 |
| spectral_mlp | 300 | 15 | 0.4990 | 0.3900 | 0.7175 |
| spectral_mlp | 700 | 15 | 0.5471 | 0.4285 | 0.7424 |
