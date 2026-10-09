# Spectral–spatial transfer pilot

No active learning or risk weights. Argmax7-class decisions. Same source/target spatial partition; source-only patch normalization. Five seeds; target700 labels inform CNN checkpoint selection and class reliability. Direct-training references use these700 labels for fitting; they choose a fixed final checkpoint.2000 development labels are separate evaluation overhead. Reserved test is not evaluated.

Prototype and self-training implementations are experimental baselines, not exact reproductions or verified novelty. Wilson bounds are heuristic gating signals, not coverage guarantees under spatial dependence. Source-continued control separates additional optimization from adaptation.

| Method | Seeds | Macro-F1 | mIoU | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| cnn_filtered_self_training | 5 | 0.3741 | 0.2888 | 0.5931 | 0.5114 |
| cnn_prototype_all | 5 | 0.3749 | 0.2891 | 0.5953 | 0.5252 |
| cnn_selective_alignment | 5 | 0.3774 | 0.2927 | 0.5978 | 0.5214 |
| cnn_self_training | 5 | 0.3753 | 0.2901 | 0.5964 | 0.5258 |
| cnn_source | 5 | 0.3842 | 0.2919 | 0.5914 | 0.5364 |
| cnn_source_continued | 5 | 0.3771 | 0.2913 | 0.5961 | 0.5174 |
| cnn_target700 | 5 | 0.6474 | 0.5380 | 0.8433 | 0.7589 |
| rf_source | 5 | 0.4528 | 0.3637 | 0.7719 | 0.4884 |
| rf_target700 | 5 | 0.5863 | 0.4934 | 0.8696 | 0.5856 |
