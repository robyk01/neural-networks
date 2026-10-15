### Testing different optimizations (Ablation study)

| Baseline (Mini-Batch SGD + CE) | Train on full data + lr decay | L2 Regularization | Scaling (mean centering) | L2 + scaling | Momentum + scaling (50 ep) | Momentum + scaling (70 ep) | Multi-Seed Weight Averaging (5 seeds) |
| - | - | - | - | - | - | - | - |
| 0.9218 | 0.9226 | 0.9220 (Train: 0.9290) | 0.9246 (Train: 0.9327) | Train: 0.9299 | 0.9246 (Train: 0.9331) | 0.9238 (Train: 0.9356) | **0.9252** (Train: 0.9330) |

L2 made the model underfitting by pulling the weights back and preventing them from descending down to loss.
Scaling by subtracting the mean improved the test score from 0.9226 to 0.9246.
Adding momentum (50 epochs, loss 0.2442) converged to the exact same minimum as mean centering (0.9246).
Training longer with slower lr decay (70 epochs, loss 0.2352, train acc 0.9356) started overfitting and dropped the score to 0.9238.
Multi-Seed Weight Averaging (averaging 5 independent runs into a single W_avg) canceled out random mini-batch jitter on borderline digits and pushed the score to 0.9252 (tied for #2 on the Kaggle leaderboard).

Raw SGD took 48 epochs to find the minimum.
Mean centering cut that to 30 epochs.
Momentum + batch 128 reached the lowest loss (0.2880) in 22 epochs.
Momentum + batch 64 reached the peak in just 14 epochs (3.4× faster convergence than baseline SGD).