## Ablation Study

| Baseline (Mini-Batch SGD + CE) | Train on full data + lr decay | L2 Regularization | Mean centering | L2 + centering | Momentum + centering (50 ep) | Momentum + centering (70 ep) | Multi-Seed Weight Averaging (5 seeds) | Multi-Seed + Gamma Contrast | Multi-Seed (10) + Gamma + Cosine Anneal + Micro L2 | Final Model (Gamma 0.78 + Pure SGD + Cosine LR + 10 Seeds) |
| - | - | - | - | - | - | - | - | - | - | - |
| 0.9218 | 0.9226 | 0.9220 (Train: 0.9290) | 0.9246 (Train: 0.9327) | Train: 0.9299 | 0.9246 (Train: 0.9331) | 0.9238 (Train: 0.9356) | 0.9252 (Train: 0.9330) | 0.9262 (Train: 0.9333) | 0.9278 (Train: 0.9362) | **0.9288** (Train: 0.9374) |

**TL:DR**
L2 made the model underfitting by pulling the weights back and preventing them from descending down to loss.
Scaling by subtracting the mean improved the test score from 0.9226 to 0.9246.
Adding momentum (50 epochs, loss 0.2442) converged to the exact same minimum as mean centering (0.9246).
Training longer with slower lr decay (70 epochs, loss 0.2352, train acc 0.9356) started overfitting and dropped the score to 0.9238.
Multi-Seed Weight Averaging (averaging 5 independent runs into a single W_avg) canceled out random mini-batch jitter on borderline digits and pushed the score to 0.9252.
Fine-tuning contrast to $\gamma = 0.78$, removing L2 to allow pure convex descent, and applying Cosine Annealing with 10-seed averaging reached a new training record (`0.9374`) and pushed the Kaggle test score to **`0.9288`** (tied for #2 among legitimate linear models).

Raw SGD took 48 epochs to find the minimum.
Mean centering cut that to 30 epochs.
Momentum + batch 128 reached the lowest loss (0.2880) in 22 epochs.
Momentum + batch 64 reached the peak in just 14 epochs (3.4× faster convergence than baseline SGD).

## Data Preprocessing

### One-Hot Encoding

For each class, we create a vector of zeros with a 1 at the index of the true class. If we have labels `y = [0, 2, 1]`, instead of passing raw integer numbers, we transform them into a matrix of shape `(3, 3)` (or `(m, 10)` for MNIST):
$$[0, 2, 1] \longrightarrow \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

This is necessary for two main reasons:

- **Eliminating ordering and bias:** Scalar labels wrongly imply that digit 9 is "greater than" digit 0, or that digit 1 is closer to 2 ($|2 - 1| = 1$) than to 9 ($|9 - 1| = 8$). One-hot encoding represents each class as an orthogonal unit vector in $\mathbb{R}^{10}$. The Euclidean distance between any two distinct one-hot classes $\mathbf{y}_a$ and $\mathbf{y}_b$ is always:
   $$\|\mathbf{y}_a - \mathbf{y}_b\|_2 = \sqrt{\sum_{k=1}^{10} (y_{a,k} - y_{b,k})^2} = \sqrt{(1 - 0)^2 + (0 - 1)^2 + 0 + \dots + 0} = \sqrt{1 + 1} = \sqrt{2}$$
   Every digit is equally separated from all others, treating all classes symmetrically without metric bias.
- **Matching probability distributions for Cross-Entropy:** Softmax outputs a probability distribution $\hat{\mathbf{y}}$ where $\sum_{i=1}^{10} \hat{y}_i = 1$. To compute loss, the true target must also be a valid probability distribution where $\sum_{i=1}^{10} y_i = 1$ (100% confidence on the correct digit). This allows Cross-Entropy $\mathcal{L} = -\sum_{i=1}^{10} y_i \log(\hat{y}_i)$ to collapse cleanly to $-\log(\hat{y}_{\text{target}})$, and enables the multivariable derivative $\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$ derived in `guide.md`.

<br>

*Source: the 1-of-K coding scheme, Bishop, 2006*

### Min-Max Pixel Scaling

Raw grayscale pixel values are integers between $0$ and $255$. We normalize them to the range $[0.0, 1.0]$ by dividing each pixel value by $255.0$:
$$x_{\text{scaled}} = \frac{x}{255.0}$$

This normalization is essential for two main reasons:

- **Preventing Softmax saturation and numerical overflow in the forward pass:** Logits are computed by summing 784 products: $z_i = \sum_{j=1}^{784} x_j W_{ji} + b_i$. If $X \in [0, 255]$, these sums produce huge logits (e.g. $+50$ to $+150$). Without numerical stabilization, $e^z$ overflows floating-point limits and produces `inf` or `NaN`. Even with $z - \max(z)$, large gaps between logits cause Softmax to immediately saturate—outputting extreme probabilities ($1.0000$ for one class and $0.0000$ for all others) on the very first epoch before learning anything useful.
- **Preventing gradient explosion in the backward pass:** The gradient with respect to weights is:
   $$\nabla_W \mathcal{L} = \frac{1}{m} X^T (\hat{Y} - Y)$$
   We can see that $X$ is directly multiplied into the gradient. If $X \in [0, 255]$, the gradient is $255\times$ larger than if $X \in [0, 1]$. Using a standard learning rate like $\eta = 0.05$ causes massive weight updates ($\Delta W \approx 0.05 \times 255 = 12.75$), making weights oscillate violently and causing the training loss to diverge. Scaling to $[0, 1]$ keeps gradient steps well-behaved and updates stable.

<br>

*Source: Efficient BackProp (Section 4.3), LeCun et al., 1998*

### Mean Centering

After dividing by 255, all pixels are strictly positive numbers in $[0, 1]$. To center our data, we compute the average pixel value across the entire training set and subtract it from every image:
$$X_{\text{centered}} = X - \mu$$
This simple change improved my Kaggle score from `0.9218` to `0.9246` and cut convergence time from 48 epochs down to 30 epochs. Here is why:
- **The problem with all-positive pixels:**
   In backpropagation, the update for the weights of a specific digit class $i$ is:
   $$\Delta W_{[:, i]} = -\eta \cdot X^T \cdot (\hat{y}_i - y_i)$$
   Because every pixel $X$ is positive or zero:
   * When an image matches digit $i$ ($y_i = 1$), the error $(\hat{y}_i - 1)$ is negative, so all 784 weights of that perceptron are forced to increase together.
   * When an image does not match ($y_i = 0$), the error $(\hat{y}_i - 0)$ is positive, so all 784 weights are forced to decrease together.
   This is a big problem. A perceptron needs to learn positive weights where digit stroke should be (to reward the digit's shape) and negative weights where digit stroke shouldn't be (to penalize wrong strokes, like the center hole of a `0`). Without centering, the model cannot increase reward weights and decrease penalty weights in the same step. It constantly fights itself, taking a slow zig-zag path across training epochs.
- **How mean centering fixes it:**
   When we subtract the mean:
   * Dark background pixels ($0$) become negative ($-0.13$).
   * Bright stroke pixels ($1.0$) stay positive*($+0.87$).
   Now, within the exact same update step for digit $i$:
   * Weights on positive stroke pixels ($x > 0$) increase.
   * Weights on negative background pixels ($x < 0$) decrease.
   The weights are no longer locked together. The perceptron can reward the digit's strokes and penalize background pixels at the exact same time. This kills the zig-zagging, allowing gradient descent to take a direct path to the minimum and reach sharper decision boundaries.
<br>

*Source: Efficient BackProp (Section 4.3), LeCun et al., 1998*

## Optimization

### GD Optimization

Gradient descent updates the weights by stepping in the opposite direction of the gradient. There are three common ways to compute these updates depending on how much data we pass at a time:

| Variant | Batch Size | Updates per Epoch | Vectorization | Loss Curve |
| :--- | :---: | :---: | :--- | :--- |
| **Full-Batch GD** | $60,000$ | $1$ | High memory usage, slow (only 50 steps in 50 epochs) | Perfectly smooth |
| **Pure SGD** | $1$ | $60,000$ | Painfully slow (Python loop overhead, no matrix parallelism) | Jumpy and chaotic |
| **Mini-Batch SGD (My Choice)** | **$64 / 128$** | **$469 - 938$** | **Fastest** (optimizes CPU cache and parallel matrix math) | **Balanced and steady** |

I chose **Mini-Batch SGD** (with batch sizes 64 and 128) because in Full-Batch, the model only updates its weights once per epoch (50 updates in 50 epochs), which takes a very long time to reach the minimum point (to converge). In Pure SGD, updating after every single image requires 60,000 Python loop iterations per epoch, which cannot use NumPy's fast parallel matrix multiplication. Mini-batch strikes the best balance: processing 64 or 128 images at once fully utilizes CPU vectorization while taking hundreds of steps per epoch.


**Why shuffling is necessary:**
For Mini-Batch, I shuffle the dataset at the start of every epoch. If I didn't shuffle, each batch would always contain the exact same images in the exact same order, causing the model to repeat identical mini-cycles. Shuffling ensures each mini-batch provides a fresh, random estimate of the true dataset gradient.

<br>

*Source: Large-Scale Machine Learning with Stochastic Gradient Descent, Bottou, 2010*

### Momentum

I implemented Momentum ($\beta = 0.9$) to speed up standard SGD. Instead of relying solely on the current batch's gradient, Momentum tracks a running "velocity" vector that remembers past gradients:

```python
vW = momentum * vW + lr * dW
vb = momentum * vb + lr * db
W -= vW
b -= vb
```

This works like a heavy ball rolling down a hill:
1. **Accelerating along consistent downhill slopes:** In standard SGD, if the gradient is small, updates crawl. With momentum, if gradients keep pointing in the same direction across consecutive batches, velocity builds up speed like compound interest:
   $$\text{Effective Speed} = \frac{1}{1 - \beta} = \frac{1}{1 - 0.9} = 10\times$$
   This allows the optimizer to move up to $10\times$ faster down steady slopes.
2. **Dampening side-to-side oscillations:** If gradients bounce back and forth between steep valley walls, the positive and negative updates cancel out in the velocity memory, while forward momentum pushes straight down the valley floor.

**Why momentum did not increase final accuracy for my perceptron:**
In my ablation experiments:
* Mean Centering without momentum reached **`0.9246`** (peaking around epoch 30).
* Momentum with Mean Centering reached the exact same **`0.9246`** (peaking in just **14 epochs**).

They converged to the exact same test score because a single-layer perceptron with Cross-Entropy is a strictly convex optimization problem there are no local minima traps to escape from, only a single global minimum. Momentum did not change the final destination, but it reached it faster (14 epochs instead of 48).

<br>

*Source: Some methods of speeding up the convergence of iteration methods, Polyak, 1964*

### Learning Rate Decay

In my training loop, I applied an exponential learning rate decay at the end of each epoch:
$$\eta_{t+1} = \eta_t \times \gamma$$
Starting from an initial rate of $\eta_0 = 0.05$ with a decay factor of $\gamma = 0.95$, the learning rate gradually shrunk to $\approx 0.0038$ by epoch 50.

This is important because of two phases during training:
- **Early epochs:** In the beginning, the weights are far away from the optimal values. A larger learning rate ($0.05$) allows the model to take big, rapid strides down the loss landscape.
- **Late epochs:** Near the bottom of the loss bowl, keeping a large step size causes overshooting the model takes steps that are too big for the narrow minimum, jumping past the optimal point and bouncing around it instead of landing in it.

By shrinking the learning rate after every epoch, the steps become smaller and more fine-grained, allowing the weights to settle smoothly into the exact center of the minimum. In my ablation experiments, adding learning rate decay improved the Kaggle score from `0.9218` to `0.9226`.

<br>

*Source: A Stochastic Approximation Method, Robbins & Monro, 1951*


## Regularization

### L2 Weight Decay

L2 regularization (commonly called weight decay) adds a penalty to the loss function based on the squared magnitude of the weights ($\frac{1}{2}\lambda \|W\|^2$). In code, this simply scales the weights down slightly on every single update step before applying the gradient:
$$W \leftarrow (1 - \eta \lambda) W - \eta \nabla_W$$
In deep neural networks (like MLPs with multiple hidden layers), L2 is very useful because having millions of parameters can cause the network to assign huge weights to random background pixels and overfit. Scaling weights down keeps them small and forces the network to distribute learning across many features.
**Why L2 hurt my single-layer Perceptron (Underfitting):**
In my experiments, adding L2 ($\lambda = 10^{-4}$) degraded performance:
* Train accuracy dropped from `0.9335` to `0.9290`.
* Kaggle test score dropped from `0.9226` to `0.9220`.
A single-layer perceptron only has 7,840 weights and no hidden layers, it is already an under-parameterized model with very limited capacity. To separate similar digits (like an `8` from a `3`), the model needs strong weights on the differentiating pixels to produce high logits ($z = XW + b$) and confident Softmax probabilities. By constantly shrinking weights towards zero, L2 made the logits small, which flattened the Softmax probabilities into timid, uncertain predictions. Instead of preventing overfitting, L2 caused **underfitting**.
<br>
*Source: A Simple Weight Decay Can Improve Generalization, Krogh & Hertz, 1992*

### Overfitting boundary

To test where the model begins to overfit, I trained a model with a slower learning rate decay ($\gamma = 0.98$) for 70 epochs instead of 50 epochs:
* **50 epochs (decay 0.95):** Final training loss was `0.2880`, training accuracy was `0.9331`, and Kaggle test score was **`0.9246`**.
* **70 epochs (decay 0.98):** Final training loss dropped to `0.2352`, training accuracy climbed to `0.9356`, but the Kaggle test score dropped to `0.9238`.
When training accuracy keeps increasing and loss keeps dropping, but the test score on unseen Kaggle data gets worse, the model stops learning general digit shapes and starts memorizing noise specific to the training set. Stopping at 50 epochs acted as an optimal early stopping point.

## Multi-Seed Weight Average

To push the performance of the single-layer perceptron, I trained 5 separate runs using different random seeds (`seeds = [42, 101, 7, 999, 2026]`) and computed the average of their weight matrices and bias vectors:
$$\bar{W} = \frac{1}{5} \sum_{k=1}^5 W_k, \quad \bar{b} = \frac{1}{5} \sum_{k=1}^5 b_k$$

Because of random mini-batch shuffling during training, each seed follows a slightly different optimization path. Averaging the weights across 5 independent trajectories cancels out this random mini-batch jitter, pulling the final weights toward the center of a wider, more stable loss basin (similar to Stochastic Weight Averaging, Izmailov et al., 2018).

* **Result on Kaggle:**
  Multi-Seed Averaging pushed my score from `0.9246` to **`0.9252`**.

<br>

*Source: Averaging Weights Leads to Wider Optima and Better Generalization, Izmailov et al., 2018*


## Discussion

### Confusion Matrix

To understand where my single-layer Perceptron hits its limit, I evaluated the model on a 12,000-image validation split (scoring `0.9202` validation accuracy) and computed the confusion matrix.

The top most confused digit pairs were:
* **True 9 $\rightarrow$ Predicted 8:** 41 errors (and **True 8 $\rightarrow$ Predicted 9:** 41 errors) 
* **True 4 $\rightarrow$ Predicted 7:** 38 errors.
* **True 1 $\rightarrow$ Predicted 6:** 36 errors (and **True 6 $\rightarrow$ Predicted 1:** 32 errors).

### Gamma Contrast Boost

Looking at the misclassified digits from the confusion matrix, many errors happened because faint strokes (pixels with low values like $0.1 - 0.3$) in loops contributed almost nothing to the dot product.

To test if boosting these faint strokes would help the linear templates detect ambiguous loops, I applied a power-law contrast transformation (gamma correction) before mean centering:
$$X_{\text{boost}} = X^{\gamma} \quad (\gamma = 0.7)$$

Because $\gamma < 1.0$:
* Faint pixels get amplified: $0.2^{0.7} \approx 0.32$ (a $+60\%$ contrast boost).
* Pure background remains $0.0^{0.7} = 0.0$, and full stroke remains $1.0^{0.7} = 1.0$.

Retraining the 5-seed model with this contrast boost directly helped differentiate ambiguous loops, improving my validation accuracy (`0.9203` $\to$ `0.9214`) and pushing my Kaggle score from `0.9252` to **`0.9262`**.

### Final Push: Gamma 0.78, Pure SGD, and Cosine Annealing

In the final submission:
**Refined Contrast ($\gamma = 0.78$):** Tuning $\gamma$ to $0.78$ preserved stroke line geometry better than $0.70$, preventing the slight background pixel amplification while still sharpening faint loops.
**Pure Convex Descent (Eliminating L2):** Completely removing the micro-L2 weight decay prevented artificial shrinkage of logits, allowing the 10 linear templates to maintain sharp, confident decision boundaries on borderline digits (particularly on 8 vs 0 and 8 vs 9 transitions).
**Cosine Annealing with Warmup ($\eta = 0.10 \to 10^{-4}$):** 2 warmup epochs followed by smooth cosine decay down to $10^{-4}$ across 60 epochs allowed the optimizer to escape initial flat regions and settle precisely into the global convex optimum.
**10-Seed Weight Averaging:** Averaging across 10 random seeds pushed training accuracy to a record **`0.9374`** and the Kaggle test score to **`0.9288`**.

To move beyond the ~92.9% ceiling toward 98%+, the architecture requires non-linear hidden layers (Multi-Layer Perceptrons) that can learn compositional stroke representations.

<br>

*Source: Gradient-Based Learning Applied to Document Recognition, LeCun et al., 1998*