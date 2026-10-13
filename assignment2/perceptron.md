### Testing different optimizations (Ablation study)

| Baseline (Mini-Batch SGD + CE) | Train on full data + lr decay | L2 Regularization | Scaling (mean centering) | L2 + scaling | Momentum + scaling (50 ep) | Momentum + scaling (70 ep) |
| - | - | - | - | - | - | - |
| 0.9218 | 0.9226 | 0.9220 (Train: 0.9290) | 0.9246 (Train: 0.9327) | Train: 0.9299 | 0.9246 (Train: 0.9331) | 0.9238 (Train: 0.9356) |

L2 made the model underfitting by pulling the weights back and preventing them from descending down to loss.
Scaling by subtracting the mean improved the test score from 0.9226 to 0.9246.
Adding momentum (50 epochs, loss 0.2442) converged to the exact same minimum as mean centering (0.9246).
Training longer with slower lr decay (70 epochs, loss 0.2352, train acc 0.9356) started overfitting and dropped the score to 0.9238.
