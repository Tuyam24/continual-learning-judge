# Split CIFAR-10 results (test split)

Mean ± std over seeds. ACC = final average accuracy over all 5 tasks; forgetting = mean drop from best to final accuracy on tasks 1-4 (Chaudhry et al. 2018). Rows below the rule are tagged variants/ablations.

## Class-incremental (class-IL)

| Method | File | ACC | Forgetting | Seeds | grad steps | rollbacks |
|---|---|---:|---:|---:|---:|---:|
| Naive fine-tuning | `naive_class` | 18.69 ± 0.26% | 86.89 ± 1.22% | 3 | 3,525 ± 0 | – |
| Online EWC | `ewc_class` | 16.30 ± 0.52% | 76.55 ± 3.85% | 3 | 3,525 ± 0 | – |
| Replay | `replay_class` | 25.56 ± 2.11% | 78.75 ± 2.66% | 3 | 3,525 ± 0 | – |
| DER++ | `derpp_class` | 38.64 ± 2.03% | 58.24 ± 1.70% | 3 | 3,525 ± 0 | – |
| Ours | `ours_class` | 28.56 ± 1.69% | 45.36 ± 2.74% | 3 | 7,235 ± 37 | 226 ± 4 |
| *variants* |  |  |  |  |  |  |
| Replay [epochs11] | `replay_class_epochs11` | 27.94 ± 0.29% | 79.18 ± 0.30% | 3 | 7,755 ± 0 | – |
| DER++ [epochs11] | `derpp_class_epochs11` | 43.04 ± 1.05% | 54.21 ± 1.55% | 3 | 7,755 ± 0 | – |
| Ours [no-judge] | `ours_class_no-judge` | 24.54 ± 9.72% | 59.05 ± 9.40% | 3 | 4,975 ± 0 | 0.0 ± 0.0 |
| Ours [relaxed-any_d8] | `ours_class_relaxed-any_d8` | 27.90 ± 2.39% | 47.80 ± 5.63% | 3 | 7,002 ± 169 | 203 ± 17 |

