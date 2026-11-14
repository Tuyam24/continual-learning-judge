# Split MNIST results (test split)

Mean ± std over seeds. ACC = final average accuracy over all 5 tasks; forgetting = mean drop from best to final accuracy on tasks 1-4 (Chaudhry et al. 2018). Rows below the rule are tagged variants/ablations.

## Class-incremental (class-IL)

| Method | File | ACC | Forgetting | Seeds | grad steps | rollbacks |
|---|---|---:|---:|---:|---:|---:|
| Naive fine-tuning | `naive_class` | 19.64 ± 0.04% | 99.23 ± 0.09% | 3 | 1,694 ± 0 | – |
| Online EWC | `ewc_class` | 22.80 ± 1.44% | 95.39 ± 1.84% | 3 | 1,694 ± 0 | – |
| Replay | `replay_class` | 84.76 ± 1.61% | 17.36 ± 1.99% | 3 | 1,694 ± 0 | – |
| DER++ | `derpp_class` | 89.90 ± 0.81% | 11.40 ± 0.94% | 3 | 1,694 ± 0 | – |
| Ours | `ours_class` | 89.05 ± 0.44% | 7.08 ± 0.82% | 3 | 2,824 ± 42 | 43.0 ± 4.2 |
| *variants* |  |  |  |  |  |  |
| Replay [epochs4] | `replay_class_epochs4` | 85.43 ± 0.95% | 17.24 ± 1.23% | 3 | 3,388 ± 0 | – |
| DER++ [epochs4] | `derpp_class_epochs4` | 89.64 ± 0.52% | 12.14 ± 0.69% | 3 | 3,388 ± 0 | – |
| Ours [no-judge] | `ours_class_no-judge` | 87.27 ± 0.60% | 14.45 ± 0.63% | 3 | 2,394 ± 0 | 0.0 ± 0.0 |
| Ours [no-sleep] | `ours_class_no-sleep` | 86.09 ± 0.30% | 13.64 ± 0.25% | 3 | 2,467 ± 85 | 77.3 ± 8.5 |
| Ours [no-surprise] | `ours_class_no-surprise` | 87.65 ± 1.95% | 5.25 ± 1.79% | 3 | 2,764 ± 67 | 37.0 ± 6.7 |
| Ours [no-targeted] | `ours_class_no-targeted` | 88.44 ± 1.18% | 7.95 ± 2.18% | 3 | 2,744 ± 22 | 35.0 ± 2.2 |

## Task-incremental (task-IL)

| Method | File | ACC | Forgetting | Seeds | grad steps | rollbacks |
|---|---|---:|---:|---:|---:|---:|
| Naive fine-tuning | `naive_task` | 98.29 ± 0.26% | 1.51 ± 0.28% | 3 | 1,694 ± 0 | – |
| Online EWC | `ewc_task` | 99.16 ± 0.07% | 0.07 ± 0.04% | 3 | 1,694 ± 0 | – |
| Replay | `replay_task` | 98.85 ± 0.07% | 0.55 ± 0.02% | 3 | 1,694 ± 0 | – |
| DER++ | `derpp_task` | 99.49 ± 0.05% | 0.21 ± 0.02% | 3 | 1,694 ± 0 | – |
| Ours | `ours_task` | 99.01 ± 0.09% | 0.74 ± 0.11% | 3 | 2,394 ± 0 | 0.0 ± 0.0 |

