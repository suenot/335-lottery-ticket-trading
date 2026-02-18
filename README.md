# Chapter 205: Lottery Ticket Trading

## 1. Introduction: The Lottery Ticket Hypothesis

In 2019, Jonathan Frankle and Michael Carlin published a landmark paper, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Networks," which fundamentally changed how researchers think about neural network architectures. The central claim is both simple and profound: **a randomly initialized dense neural network contains sparse subnetworks (winning tickets) that, when trained in isolation from the same initialization, can match the test accuracy of the original dense network in a comparable number of training iterations.**

This idea has far-reaching implications for machine learning in general, but it holds particular promise for quantitative trading. Trading models must operate under severe constraints: they must generalize across market regimes, resist overfitting to noise, and often run under strict latency requirements. If we can identify a small subnetwork within a large model that captures the essential trading signals while discarding the noise-fitting capacity of the full network, we gain models that are simultaneously more robust, more interpretable, and faster to execute.

The analogy to actual lottery tickets is apt. When you buy a stack of lottery tickets, most are losers, but hidden among them is a winner. Similarly, within the vast parameter space of a randomly initialized neural network, most parameter configurations are unhelpful, but a specific sparse subset — the winning ticket — is sufficient to learn the target function. The key insight is that this winning subnetwork is determined at initialization, before any training occurs.

This chapter explores how to apply the Lottery Ticket Hypothesis (LTH) to build better trading systems. We will cover the mathematical foundations, the iterative magnitude pruning algorithm for finding winning tickets, practical considerations for financial time series, and a complete Rust implementation that integrates with Bybit market data.

## 2. Mathematical Foundation

### 2.1 Formal Statement

Let $f(x; \theta)$ denote a neural network with input $x$ and parameters $\theta \in \mathbb{R}^d$. The parameters are initialized as $\theta_0 \sim \mathcal{D}_\theta$, where $\mathcal{D}_\theta$ is the initialization distribution (e.g., Xavier or Kaiming). After training for $T$ iterations using optimizer $\mathcal{A}$, we obtain $\theta_T$.

A **mask** $m \in \{0, 1\}^d$ defines a subnetwork $f(x; m \odot \theta)$, where $\odot$ denotes element-wise multiplication. The sparsity of the mask is $s = 1 - \frac{\|m\|_0}{d}$, representing the fraction of pruned parameters.

**Lottery Ticket Hypothesis:** There exists a mask $m^*$ with sparsity $s \gg 0$ such that $f(x; m^* \odot \theta_0)$ trained for at most $T$ iterations achieves test accuracy $\geq$ that of $f(x; \theta_0)$ trained for $T$ iterations.

### 2.2 Winning Ticket Identification

The winning ticket is identified through a post-hoc process: we first train the dense network, then determine which weights were "important" based on their trained magnitudes, and finally verify that the sparse subnetwork defined by these important weights can train successfully from the original initialization.

Formally, given trained parameters $\theta_T$, we construct the mask:

$$m^*_j = \begin{cases} 1 & \text{if } |\theta_{T,j}| \geq \text{quantile}(|\theta_T|, s) \\ 0 & \text{otherwise} \end{cases}$$

This means we keep the top $(1-s)$ fraction of weights by magnitude and prune the rest.

### 2.3 Iterative Magnitude Pruning (IMP)

Rather than pruning all weights at once, the IMP algorithm prunes a fraction $p$ per round over $n$ rounds, achieving a final sparsity of $1 - (1-p)^n$. This iterative approach finds better tickets than one-shot pruning.

**Algorithm: Iterative Magnitude Pruning**

```
Input: Network f(x; θ), pruning rate p, rounds n
1. Initialize θ₀ ~ D_θ
2. Save initial weights θ₀
3. For round i = 1 to n:
   a. Train f(x; mᵢ₋₁ ⊙ θ₀) for T iterations → θ_T
   b. Compute magnitude scores: sⱼ = |θ_{T,j}| for all j where mᵢ₋₁,ⱼ = 1
   c. Prune bottom p fraction: mᵢ,ⱼ = mᵢ₋₁,ⱼ · 𝟙[sⱼ ≥ quantile(s, p)]
   d. Rewind weights to θ₀
4. Return mask mₙ (the winning ticket mask)
```

The key step is **rewinding**: after each pruning round, instead of continuing from the trained weights, we reset to the original initialization $\theta_0$. This is what distinguishes lottery ticket identification from standard pruning — we are finding subnetworks that can train from scratch, not merely compressing an already-trained model.

## 3. Finding Lottery Tickets

### 3.1 The Train-Prune-Rewind-Retrain Cycle

The practical workflow for finding lottery tickets consists of four steps repeated iteratively:

1. **Train**: Train the current subnetwork (initially the full network) to convergence or for a fixed number of epochs.
2. **Prune**: Remove a fraction of the remaining weights with the smallest magnitudes. Typical pruning rates are 20% per round.
3. **Rewind**: Reset all surviving weights to their values at initialization (or at an early training epoch — see Section 5 on late rewinding).
4. **Retrain**: Train the pruned subnetwork from the rewound weights to verify it can still achieve full accuracy.

This cycle continues until the desired sparsity level is reached or until performance begins to degrade significantly.

### 3.2 Early-Bird Tickets

A significant practical concern with IMP is its computational cost: each round requires full training of the network. You et al. (2020) introduced the concept of **early-bird tickets** — winning tickets that can be identified early in training, typically within the first 10-30% of the training epochs.

The early-bird phenomenon exploits the observation that the mask structure (which weights are important) stabilizes long before the weights themselves converge. By monitoring the Hamming distance between masks drawn at consecutive early checkpoints, we can detect when the mask has stabilized and stop training early.

For trading applications, early-bird tickets are especially valuable because they dramatically reduce the computational cost of lottery ticket search, making it feasible to re-identify tickets as market regimes shift.

### 3.3 Mask Stability and Ticket Quality

A robust winning ticket should exhibit **mask stability**: the same set of weights should be identified as important across multiple independent training runs. This stability metric provides a confidence measure for the identified ticket. In trading contexts, we can extend this to check mask stability across different market periods — a ticket that is stable across bull and bear markets is likely capturing genuine signal rather than regime-specific noise.

## 4. Trading Applications

### 4.1 Finding Minimal Trading Signal Extractors

In quantitative trading, input features often include hundreds of technical indicators, order book statistics, and alternative data signals. Many of these are redundant or noisy. The lottery ticket framework provides a principled way to identify the minimal set of model parameters — and by extension, the minimal set of input-feature interactions — needed to extract trading signal.

When we find a winning ticket in a trading model, the structure of the surviving weights reveals which input features and which feature interactions the model truly relies on. Pruned connections from specific input features indicate that those features contribute little to the trading signal, even if they appear correlated with returns in isolation.

### 4.2 Robust Sparse Models

Overfitting is the central challenge in ML-based trading. Dense networks have enormous capacity to memorize the training data, including its noise. Lottery tickets offer a structural form of regularization: by constraining the model to use only a sparse subset of its parameters, we limit its capacity to fit noise while preserving its ability to capture genuine patterns.

Empirical results consistently show that winning tickets generalize better than their dense counterparts, especially in low signal-to-noise ratio settings typical of financial data. The sparsity acts as an implicit regularizer, complementing explicit regularization techniques like dropout and weight decay.

### 4.3 Latency Benefits

In high-frequency and medium-frequency trading, inference latency matters. A model pruned to 10% of its original parameters performs roughly 10x fewer multiply-accumulate operations during inference. While the theoretical speedup depends on hardware support for sparse computation, even without specialized hardware, smaller models benefit from better cache utilization and reduced memory bandwidth requirements.

### 4.4 Regime Adaptation

Markets undergo regime changes — periods of high vs. low volatility, trending vs. mean-reverting behavior. A practical approach is to maintain a library of winning tickets identified across different market regimes. When a regime change is detected, the system can switch to the appropriate ticket without retraining from scratch, since each ticket needs only its initial weights and mask.

## 5. Late Rewinding and Linear Mode Connectivity

### 5.1 Late Rewinding

Frankle et al. (2020) discovered that for larger networks, rewinding to iteration 0 often fails — the winning ticket cannot train successfully from the exact initialization. However, rewinding to an early training iteration $k$ (typically $k$ corresponding to 0.1-7% of total training) restores the lottery ticket property. This is called **late rewinding** or **rewinding to iteration $k$**.

The late rewinding modification changes the algorithm:
- Instead of saving $\theta_0$, save $\theta_k$ after $k$ steps of training.
- After each pruning round, rewind to $\theta_k$ instead of $\theta_0$.

This works because the first few iterations of training move the parameters into a region of the loss landscape from which the sparse subnetwork can successfully optimize. The initial random weights may place some subnetwork parameters in adversarial positions that only the full network's redundancy can overcome.

### 5.2 Linear Mode Connectivity

Frankle et al. further showed that successful lottery tickets exhibit **linear mode connectivity** with the dense model: for any $\alpha \in [0, 1]$, the linearly interpolated model $\alpha \cdot \theta_{\text{dense}} + (1-\alpha) \cdot \theta_{\text{sparse}}$ achieves loss comparable to both endpoints. This means the winning ticket and the dense model converge to the same basin of the loss landscape.

This property has practical implications: it confirms that the winning ticket is truly learning the same function as the dense model, not a different function that happens to achieve similar accuracy. For trading, this means the winning ticket captures the same market patterns, just with fewer parameters.

## 6. Implementation Walkthrough (Rust)

Our Rust implementation provides a complete lottery ticket framework for trading models. The core components are:

### 6.1 Network with Mask Support

We implement a feedforward neural network where each layer has an associated binary mask. The forward pass computes:

```rust
// Masked forward pass: output = activation(input * (W ⊙ mask) + b)
let masked_weights = weights * mask; // element-wise
let output = (input.dot(&masked_weights) + bias).mapv(|x| x.max(0.0)); // ReLU
```

### 6.2 Weight Rewinding

The network supports saving and restoring initial weights:

```rust
// Before any training
let initial_weights = network.save_weights();

// After pruning round
network.restore_weights(&initial_weights);
// Mask is preserved — only unmasked weights are restored
```

### 6.3 Iterative Magnitude Pruning

The IMP loop implements the full lottery ticket search:

```rust
for round in 0..num_rounds {
    // Train with current mask
    network.train(&data, epochs);

    // Prune bottom p% of remaining weights by magnitude
    network.prune_by_magnitude(pruning_rate);

    // Rewind to initial weights (keeping new mask)
    network.restore_weights(&initial_weights);
}

// The final mask defines the winning ticket
let winning_mask = network.get_mask();
```

### 6.4 Winning vs. Random Ticket Comparison

To validate that we found a genuine winning ticket (not just any sparse subnetwork), we compare against a random ticket with the same sparsity:

```rust
// Random ticket: same sparsity, random mask
let random_mask = generate_random_mask(network.param_count(), sparsity);
random_network.set_mask(&random_mask);
random_network.restore_weights(&initial_weights);
random_network.train(&data, epochs);

// Winning ticket should significantly outperform random ticket
assert!(winning_accuracy > random_accuracy);
```

### 6.5 Data Pipeline

The implementation includes a Bybit API client that fetches OHLCV candlestick data for any trading pair. Features are computed from raw candles (returns, volatility, momentum indicators), normalized, and fed into the network.

See `rust/src/lib.rs` for the complete implementation and `rust/examples/trading_example.rs` for a runnable example.

## 7. Bybit Data Integration

Our implementation connects to the Bybit public API to fetch real market data. The endpoint used is:

```
GET https://api.bybit.com/v5/market/kline
```

Parameters:
- `category`: "linear" (for USDT perpetual contracts)
- `symbol`: e.g., "BTCUSDT"
- `interval`: e.g., "60" (1-hour candles)
- `limit`: number of candles (max 200)

The raw OHLCV data is transformed into features suitable for the neural network:

1. **Log returns**: $r_t = \ln(C_t / C_{t-1})$ where $C_t$ is the close price at time $t$.
2. **Volatility**: Rolling standard deviation of returns over a lookback window.
3. **Momentum**: Rate of change of price over multiple lookback periods.
4. **Volume ratio**: Current volume relative to rolling average volume.

These features are normalized to zero mean and unit variance before being fed into the network. The target variable is the sign of the next-period return (binary classification: up or down).

### Data Considerations for Lottery Tickets

When applying the lottery ticket methodology to financial data, several considerations apply:

- **Non-stationarity**: Market data is non-stationary, so winning tickets may become "losing tickets" over time. Periodic re-identification of tickets (e.g., monthly or quarterly) is recommended.
- **Sample efficiency**: Financial datasets are small relative to image or text datasets. This actually favors the lottery ticket approach, as sparse models generalize better in low-data regimes.
- **Walk-forward validation**: Always use walk-forward (expanding or sliding window) validation rather than random train/test splits, to respect the temporal structure of the data.

## 8. Key Takeaways

1. **The Lottery Ticket Hypothesis states that dense neural networks contain sparse subnetworks that can train to full accuracy from their original initialization.** This has been empirically validated across architectures and domains.

2. **Iterative Magnitude Pruning (IMP) is the primary algorithm for finding winning tickets.** It alternates between training, pruning the smallest-magnitude weights, and rewinding to initial weights. Multiple rounds of moderate pruning outperform single-round aggressive pruning.

3. **Winning tickets are not just any sparse network.** They significantly outperform random subnetworks of the same sparsity, confirming that the specific structure identified by IMP is meaningful.

4. **For trading, lottery tickets offer three key benefits**: (a) structural regularization that reduces overfitting to market noise, (b) model interpretability through the structure of surviving connections, and (c) reduced inference latency for time-sensitive strategies.

5. **Late rewinding (to an early training iteration rather than iteration 0) is often necessary** for larger networks and more complex tasks. This is a practical consideration when implementing lottery ticket search for production trading systems.

6. **Early-bird tickets can be identified within the first 10-30% of training**, dramatically reducing the computational cost of lottery ticket search and making it feasible to periodically re-identify tickets as market conditions change.

7. **Mask stability across market regimes is a valuable diagnostic.** A winning ticket whose mask is consistent across bull markets, bear markets, and sideways markets is more likely to capture genuine trading signal.

8. **The Rust implementation provided in this chapter** demonstrates the complete workflow: fetching market data from Bybit, training a dense network, applying IMP to find winning tickets, and comparing winning tickets against random baselines.

## References

- Frankle, J., & Carlin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Networks. *ICLR 2019*.
- Frankle, J., Dziugaite, G. K., Roy, D. M., & Carlin, M. (2020). Linear Mode Connectivity and the Lottery Ticket Hypothesis. *ICML 2020*.
- You, H., Li, C., Xu, P., Fu, Y., Wang, Y., Chen, X., Baraniuk, R. G., Wang, Z., & Lin, Y. (2020). Drawing Early-Bird Tickets: Towards More Efficient Training of Deep Networks. *ICLR 2020*.
- Malach, E., Yehudai, G., Shalev-Shwartz, S., & Shamir, O. (2020). Proving the Lottery Ticket Hypothesis: Pruning is All You Need. *ICML 2020*.
