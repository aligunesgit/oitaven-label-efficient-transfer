# Spectral–spatial transfer comparison

Run with the isolated experiment interpreter:

```sh
python -u watershed_transfer/continue.py
```

Do not start a second controller. Progress: `controller_status.json`, `status.json`, `run.log`. Completed results: `results/REPORT.md`. Final audit reconstructs every model prediction and score before generating comparison plots.

Frozen class-supported spatial split and sampling weights from the previous pilot are reused, including the untouched reserved test.15×15 five-band patches stay inside their assigned256px blocks and32px buffers. Mean/std use only source training patches. RF and CNN see the same patch footprint; RF uses center/mean/std features and CNN uses the full patch plus center spectral branch.

Nine methods × five seeds. Random Forest source-only and source+700 target-label references; CNN source-only; extra source training control; ordinary self-training; reliability-filtered self-training; all-class prototype alignment; selective prototype alignment; source+700-label supervised fine-tuning reference. There is no active learning and no ecological cost matrix. All decisions are argmax.

The700-label target controller selects CNN checkpoints and estimates class precision. It is neither development nor test. Direct-supervision references use these same700 labels for fitting instead. Consequently this is a label-informed transfer pilot, not zero-target-label UDA. CNN target700 uses a fixed30-epoch final checkpoint, avoiding validation on training labels. Development uses2000 distinct references and inclusion-probability weights.

Pseudo-labels are frozen from the same source checkpoint for every branch. No target-pool ground truth is loaded. Confidence≥.9, at most200 samples per class, at least20 candidates. Filtered variants require at least10 controller predictions, Wilson precision lower bound≥.6, and source-prototype cosine similarity above the source class10th percentile. Bounds are heuristics under spatial dependence, not guarantees. Prototype loss compares normalized target batch means with fixed source prototypes. This is an experimental classwise-alignment baseline, not an exact paper reproduction or validated novelty.

Best source models, branch initial states, pseudo-label IDs, reliability reports, epoch histories, resumable epoch checkpoints, final weights and development probabilities are retained. A source-only continued-training branch controls for additional optimization. Five seeds share one spatial partition; independent-scene and split sensitivity validation remain required.
