use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;

// ============================================================
// Bybit API data types
// ============================================================

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: String,
    pub list: Vec<Vec<String>>,
}

/// OHLCV candle data.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline (candlestick) data from Bybit V5 API.
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // API returns newest first; reverse to chronological order
    candles.reverse();
    Ok(candles)
}

// ============================================================
// Feature engineering
// ============================================================

/// Compute features and binary labels from candle data.
///
/// Features per sample (using lookback window):
///   0: log return
///   1: volatility (rolling std of returns, window=lookback)
///   2: momentum (price change over lookback periods)
///   3: volume ratio (current volume / rolling mean volume)
///
/// Label: 1.0 if next-period return > 0, else 0.0
pub fn prepare_features(candles: &[Candle], lookback: usize) -> (Array2<f64>, Array1<f64>) {
    let n = candles.len();
    assert!(n > lookback + 1, "Not enough candles for given lookback");

    let num_samples = n - lookback - 1; // -1 because we need next return for label
    let num_features = 4;

    let mut features = Array2::<f64>::zeros((num_samples, num_features));
    let mut labels = Array1::<f64>::zeros(num_samples);

    // Precompute log returns
    let log_returns: Vec<f64> = (1..n)
        .map(|i| (candles[i].close / candles[i - 1].close).ln())
        .collect();

    for i in 0..num_samples {
        let idx = lookback + i; // index into log_returns (0-based, corresponds to candle idx+1)

        // Feature 0: current log return
        features[[i, 0]] = log_returns[idx - 1];

        // Feature 1: rolling volatility
        let window_returns = &log_returns[(idx - lookback)..idx];
        let mean_ret: f64 = window_returns.iter().sum::<f64>() / lookback as f64;
        let var: f64 =
            window_returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / lookback as f64;
        features[[i, 1]] = var.sqrt();

        // Feature 2: momentum (price change over lookback)
        let candle_idx = idx + 1; // map back to candle index
        features[[i, 2]] =
            (candles[candle_idx].close / candles[candle_idx - lookback].close).ln();

        // Feature 3: volume ratio
        let vol_window: f64 = (candle_idx - lookback..candle_idx)
            .map(|j| candles[j].volume)
            .sum::<f64>()
            / lookback as f64;
        features[[i, 3]] = if vol_window > 0.0 {
            candles[candle_idx].volume / vol_window
        } else {
            1.0
        };

        // Label: sign of next return
        labels[i] = if log_returns[idx] > 0.0 { 1.0 } else { 0.0 };
    }

    // Normalize features to zero mean, unit variance
    for col in 0..num_features {
        let column = features.column(col).to_owned();
        let mean = column.mean().unwrap_or(0.0);
        let std = column.std(0.0);
        let std = if std < 1e-10 { 1.0 } else { std };
        for row in 0..num_samples {
            features[[row, col]] = (features[[row, col]] - mean) / std;
        }
    }

    (features, labels)
}

// ============================================================
// Neural network with lottery ticket support
// ============================================================

/// A single dense layer with mask support.
#[derive(Clone, Debug)]
pub struct MaskedLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub mask: Array2<f64>, // 1.0 = active, 0.0 = pruned
}

impl MaskedLayer {
    /// Create a new layer with Kaiming-like initialization.
    pub fn new(input_size: usize, output_size: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / input_size as f64).sqrt();
        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_size);
        let mask = Array2::ones((input_size, output_size));
        MaskedLayer {
            weights,
            biases,
            mask,
        }
    }

    /// Forward pass with mask applied.
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let masked_w = &self.weights * &self.mask;
        let mut output = input.dot(&masked_w);
        // Add biases (broadcast)
        for mut row in output.rows_mut() {
            row += &self.biases;
        }
        output
    }

    /// Count active (non-pruned) weights.
    pub fn active_weights(&self) -> usize {
        self.mask.iter().filter(|&&v| v > 0.5).count()
    }

    /// Total weight count.
    pub fn total_weights(&self) -> usize {
        self.mask.len()
    }
}

/// Feedforward neural network with lottery ticket support.
#[derive(Clone, Debug)]
pub struct LotteryTicketNetwork {
    pub layers: Vec<MaskedLayer>,
    pub learning_rate: f64,
}

/// Saved weights for rewinding.
#[derive(Clone, Debug)]
pub struct SavedWeights {
    pub layer_weights: Vec<Array2<f64>>,
    pub layer_biases: Vec<Array1<f64>>,
}

impl LotteryTicketNetwork {
    /// Build a network with given layer sizes. E.g., &[4, 32, 16, 1] creates 3 layers.
    pub fn new(layer_sizes: &[usize], learning_rate: f64, rng: &mut impl Rng) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(MaskedLayer::new(layer_sizes[i], layer_sizes[i + 1], rng));
        }
        LotteryTicketNetwork {
            layers,
            learning_rate,
        }
    }

    /// Save current weights (for rewinding).
    pub fn save_weights(&self) -> SavedWeights {
        SavedWeights {
            layer_weights: self.layers.iter().map(|l| l.weights.clone()).collect(),
            layer_biases: self.layers.iter().map(|l| l.biases.clone()).collect(),
        }
    }

    /// Restore weights from saved state (masks are preserved).
    pub fn restore_weights(&mut self, saved: &SavedWeights) {
        for (layer, (w, b)) in self.layers.iter_mut().zip(
            saved
                .layer_weights
                .iter()
                .zip(saved.layer_biases.iter()),
        ) {
            layer.weights = w.clone();
            layer.biases = b.clone();
        }
    }

    /// Forward pass through entire network. Hidden layers use ReLU; output uses sigmoid.
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < self.layers.len() - 1 {
                // ReLU for hidden layers
                x.mapv_inplace(|v| v.max(0.0));
            } else {
                // Sigmoid for output layer
                x.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));
            }
        }
        x
    }

    /// Train for one epoch using mini-batch SGD with backpropagation.
    /// Binary cross-entropy loss for binary classification.
    pub fn train_epoch(
        &mut self,
        features: &Array2<f64>,
        labels: &Array1<f64>,
        batch_size: usize,
    ) -> f64 {
        let n = features.nrows();
        let mut total_loss = 0.0;
        let mut batches = 0;

        let mut idx = 0;
        while idx < n {
            let end = (idx + batch_size).min(n);
            let batch_x = features.slice(ndarray::s![idx..end, ..]).to_owned();
            let batch_y = labels.slice(ndarray::s![idx..end]).to_owned();

            // Forward pass, storing activations
            let mut activations: Vec<Array2<f64>> = Vec::new();
            activations.push(batch_x.clone());

            let mut x = batch_x;
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward(&x);
                if i < self.layers.len() - 1 {
                    x.mapv_inplace(|v| v.max(0.0));
                } else {
                    x.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));
                }
                activations.push(x.clone());
            }

            // Compute loss (binary cross-entropy)
            let predictions = activations.last().unwrap();
            let bs = predictions.nrows();
            let mut loss = 0.0;
            for i in 0..bs {
                let p = predictions[[i, 0]].clamp(1e-7, 1.0 - 1e-7);
                let y = batch_y[i];
                loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            }
            total_loss += loss;
            batches += 1;

            // Backward pass
            // Output layer gradient: dL/dz = prediction - label (for sigmoid + BCE)
            let mut delta = Array2::zeros(predictions.raw_dim());
            for i in 0..bs {
                delta[[i, 0]] = predictions[[i, 0]] - batch_y[i];
            }
            delta /= bs as f64;

            // Backpropagate through layers
            for l in (0..self.layers.len()).rev() {
                let input_act = &activations[l];

                // Compute weight gradient
                let weight_grad = input_act.t().dot(&delta);

                // Compute bias gradient
                let bias_grad = delta.sum_axis(Axis(0));

                // Compute delta for previous layer (if not first layer)
                if l > 0 {
                    let masked_w = &self.layers[l].weights * &self.layers[l].mask;
                    let mut new_delta = delta.dot(&masked_w.t());
                    // ReLU derivative
                    let prev_act = &activations[l];
                    for ((r, c), val) in new_delta.indexed_iter_mut() {
                        if prev_act[[r, c]] <= 0.0 {
                            *val = 0.0;
                        }
                    }
                    delta = new_delta;
                }

                // Update weights (only where mask is active)
                let lr = self.learning_rate;
                let layer = &mut self.layers[l];
                for ((r, c), w) in layer.weights.indexed_iter_mut() {
                    if layer.mask[[r, c]] > 0.5 {
                        *w -= lr * weight_grad[[r, c]];
                    }
                }
                for (i, b) in layer.biases.iter_mut().enumerate() {
                    *b -= lr * bias_grad[i];
                }
            }

            idx = end;
        }

        total_loss / batches as f64
    }

    /// Train for multiple epochs.
    pub fn train(
        &mut self,
        features: &Array2<f64>,
        labels: &Array1<f64>,
        epochs: usize,
        batch_size: usize,
    ) -> Vec<f64> {
        let mut losses = Vec::new();
        for _ in 0..epochs {
            let loss = self.train_epoch(features, labels, batch_size);
            losses.push(loss);
        }
        losses
    }

    /// Compute accuracy on given data.
    pub fn accuracy(&self, features: &Array2<f64>, labels: &Array1<f64>) -> f64 {
        let preds = self.forward(features);
        let n = labels.len();
        let mut correct = 0;
        for i in 0..n {
            let predicted = if preds[[i, 0]] > 0.5 { 1.0 } else { 0.0 };
            if (predicted - labels[i]).abs() < 1e-5 {
                correct += 1;
            }
        }
        correct as f64 / n as f64
    }

    /// Get current sparsity (fraction of pruned weights).
    pub fn sparsity(&self) -> f64 {
        let total: usize = self.layers.iter().map(|l| l.total_weights()).sum();
        let active: usize = self.layers.iter().map(|l| l.active_weights()).sum();
        if total == 0 {
            return 0.0;
        }
        1.0 - (active as f64 / total as f64)
    }

    /// Total parameter count.
    pub fn total_params(&self) -> usize {
        self.layers.iter().map(|l| l.total_weights()).sum()
    }

    /// Active parameter count.
    pub fn active_params(&self) -> usize {
        self.layers.iter().map(|l| l.active_weights()).sum()
    }

    /// Prune the bottom `fraction` of remaining weights by magnitude (globally).
    pub fn prune_by_magnitude(&mut self, fraction: f64) {
        // Collect all active weight magnitudes
        let mut magnitudes: Vec<f64> = Vec::new();
        for layer in &self.layers {
            for ((r, c), &w) in layer.weights.indexed_iter() {
                if layer.mask[[r, c]] > 0.5 {
                    magnitudes.push(w.abs());
                }
            }
        }

        if magnitudes.is_empty() {
            return;
        }

        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let prune_count = (magnitudes.len() as f64 * fraction) as usize;
        if prune_count == 0 || prune_count >= magnitudes.len() {
            return;
        }
        let threshold = magnitudes[prune_count];

        // Apply pruning
        for layer in &mut self.layers {
            for ((r, c), mask_val) in layer.mask.indexed_iter_mut() {
                if *mask_val > 0.5 && layer.weights[[r, c]].abs() < threshold {
                    *mask_val = 0.0;
                }
            }
        }
    }

    /// Get all masks concatenated.
    pub fn get_mask_flat(&self) -> Vec<f64> {
        let mut mask = Vec::new();
        for layer in &self.layers {
            mask.extend(layer.mask.iter());
        }
        mask
    }

    /// Set masks from a flat vector.
    pub fn set_mask_flat(&mut self, mask: &[f64]) {
        let mut offset = 0;
        for layer in &mut self.layers {
            let size = layer.mask.len();
            for (i, val) in layer.mask.iter_mut().enumerate() {
                *val = mask[offset + i];
            }
            offset += size;
        }
    }

    /// Reset all masks to 1.0 (fully dense).
    pub fn reset_masks(&mut self) {
        for layer in &mut self.layers {
            layer.mask.fill(1.0);
        }
    }
}

// ============================================================
// Lottery ticket algorithms
// ============================================================

/// Result of the iterative magnitude pruning process.
#[derive(Debug, Clone)]
pub struct LotteryTicketResult {
    pub winning_mask: Vec<f64>,
    pub sparsity: f64,
    pub rounds_completed: usize,
    pub sparsity_history: Vec<f64>,
    pub loss_history: Vec<Vec<f64>>,
}

/// Run Iterative Magnitude Pruning (IMP) to find a winning lottery ticket.
///
/// Arguments:
///   - `network`: the network to prune (will be modified)
///   - `initial_weights`: saved weights from initialization
///   - `features`: training features
///   - `labels`: training labels
///   - `pruning_rate`: fraction of remaining weights to prune each round (e.g., 0.2)
///   - `num_rounds`: number of pruning rounds
///   - `epochs_per_round`: training epochs per IMP round
///   - `batch_size`: batch size for training
pub fn iterative_magnitude_pruning(
    network: &mut LotteryTicketNetwork,
    initial_weights: &SavedWeights,
    features: &Array2<f64>,
    labels: &Array1<f64>,
    pruning_rate: f64,
    num_rounds: usize,
    epochs_per_round: usize,
    batch_size: usize,
) -> LotteryTicketResult {
    let mut sparsity_history = Vec::new();
    let mut loss_history = Vec::new();

    for _round in 0..num_rounds {
        // Train with current mask
        let losses = network.train(features, labels, epochs_per_round, batch_size);
        loss_history.push(losses);

        // Prune bottom fraction of remaining weights
        network.prune_by_magnitude(pruning_rate);

        let current_sparsity = network.sparsity();
        sparsity_history.push(current_sparsity);

        // Rewind to initial weights (mask is preserved)
        network.restore_weights(initial_weights);
    }

    let winning_mask = network.get_mask_flat();
    let final_sparsity = network.sparsity();

    LotteryTicketResult {
        winning_mask,
        sparsity: final_sparsity,
        rounds_completed: num_rounds,
        sparsity_history,
        loss_history,
    }
}

/// Generate a random mask with the given sparsity level.
pub fn generate_random_mask(total_params: usize, sparsity: f64, rng: &mut impl Rng) -> Vec<f64> {
    let active_count = ((1.0 - sparsity) * total_params as f64).round() as usize;
    let mut mask = vec![0.0f64; total_params];

    // Randomly select positions to keep active
    let mut indices: Vec<usize> = (0..total_params).collect();
    // Fisher-Yates shuffle (partial)
    for i in 0..active_count.min(total_params) {
        let j = rng.gen_range(i..total_params);
        indices.swap(i, j);
    }
    for &idx in &indices[..active_count.min(total_params)] {
        mask[idx] = 1.0;
    }

    mask
}

/// Evaluate a winning ticket: restore initial weights, apply winning mask, train, measure accuracy.
pub fn evaluate_winning_ticket(
    network: &mut LotteryTicketNetwork,
    initial_weights: &SavedWeights,
    mask: &[f64],
    train_features: &Array2<f64>,
    train_labels: &Array1<f64>,
    test_features: &Array2<f64>,
    test_labels: &Array1<f64>,
    epochs: usize,
    batch_size: usize,
) -> f64 {
    network.restore_weights(initial_weights);
    network.set_mask_flat(mask);
    network.train(train_features, train_labels, epochs, batch_size);
    network.accuracy(test_features, test_labels)
}

/// Compare winning ticket vs random ticket at the same sparsity.
pub fn compare_tickets(
    layer_sizes: &[usize],
    learning_rate: f64,
    train_features: &Array2<f64>,
    train_labels: &Array1<f64>,
    test_features: &Array2<f64>,
    test_labels: &Array1<f64>,
    pruning_rate: f64,
    num_rounds: usize,
    epochs_per_round: usize,
    batch_size: usize,
    seed: u64,
) -> TicketComparison {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Build network and save initial weights
    let mut network = LotteryTicketNetwork::new(layer_sizes, learning_rate, &mut rng);
    let initial_weights = network.save_weights();

    // === Dense model baseline ===
    let mut dense_net = network.clone();
    dense_net.train(
        train_features,
        train_labels,
        epochs_per_round * num_rounds,
        batch_size,
    );
    let dense_accuracy = dense_net.accuracy(test_features, test_labels);

    // === Find winning ticket via IMP ===
    let imp_result = iterative_magnitude_pruning(
        &mut network,
        &initial_weights,
        train_features,
        train_labels,
        pruning_rate,
        num_rounds,
        epochs_per_round,
        batch_size,
    );

    // Evaluate winning ticket
    let mut winning_net = network.clone();
    let winning_accuracy = evaluate_winning_ticket(
        &mut winning_net,
        &initial_weights,
        &imp_result.winning_mask,
        train_features,
        train_labels,
        test_features,
        test_labels,
        epochs_per_round * num_rounds,
        batch_size,
    );

    // === Random ticket at same sparsity ===
    let random_mask =
        generate_random_mask(network.total_params(), imp_result.sparsity, &mut rng);
    let mut random_net =
        LotteryTicketNetwork::new(layer_sizes, learning_rate, &mut rng);
    // Use same initial weights for fair comparison
    random_net.restore_weights(&initial_weights);
    random_net.set_mask_flat(&random_mask);
    random_net.train(
        train_features,
        train_labels,
        epochs_per_round * num_rounds,
        batch_size,
    );
    let random_accuracy = random_net.accuracy(test_features, test_labels);

    TicketComparison {
        dense_accuracy,
        winning_ticket_accuracy: winning_accuracy,
        random_ticket_accuracy: random_accuracy,
        sparsity: imp_result.sparsity,
        active_params: network.active_params(),
        total_params: network.total_params(),
    }
}

/// Comparison results between different ticket types.
#[derive(Debug, Clone)]
pub struct TicketComparison {
    pub dense_accuracy: f64,
    pub winning_ticket_accuracy: f64,
    pub random_ticket_accuracy: f64,
    pub sparsity: f64,
    pub active_params: usize,
    pub total_params: usize,
}

impl std::fmt::Display for TicketComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Ticket Comparison ===")?;
        writeln!(
            f,
            "Sparsity:                {:.1}% ({}/{} params active)",
            self.sparsity * 100.0,
            self.active_params,
            self.total_params
        )?;
        writeln!(
            f,
            "Dense model accuracy:    {:.2}%",
            self.dense_accuracy * 100.0
        )?;
        writeln!(
            f,
            "Winning ticket accuracy: {:.2}%",
            self.winning_ticket_accuracy * 100.0
        )?;
        writeln!(
            f,
            "Random ticket accuracy:  {:.2}%",
            self.random_ticket_accuracy * 100.0
        )?;
        writeln!(
            f,
            "Winning vs Random delta: {:.2}pp",
            (self.winning_ticket_accuracy - self.random_ticket_accuracy) * 100.0
        )
    }
}

// We need StdRng for seeded reproducibility
use rand::SeedableRng;

// ============================================================
// Sparsity vs Accuracy tracking
// ============================================================

/// Track accuracy at various sparsity levels.
#[derive(Debug, Clone)]
pub struct SparsityAccuracyPoint {
    pub sparsity: f64,
    pub winning_accuracy: f64,
    pub random_accuracy: f64,
}

/// Run IMP incrementally and record accuracy at each sparsity level.
pub fn sparsity_vs_accuracy(
    layer_sizes: &[usize],
    learning_rate: f64,
    train_features: &Array2<f64>,
    train_labels: &Array1<f64>,
    test_features: &Array2<f64>,
    test_labels: &Array1<f64>,
    pruning_rate: f64,
    num_rounds: usize,
    epochs_per_round: usize,
    batch_size: usize,
    seed: u64,
) -> Vec<SparsityAccuracyPoint> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut network = LotteryTicketNetwork::new(layer_sizes, learning_rate, &mut rng);
    let initial_weights = network.save_weights();

    let mut results = Vec::new();

    for _round in 0..num_rounds {
        // Train with current mask
        network.train(train_features, train_labels, epochs_per_round, batch_size);

        // Prune
        network.prune_by_magnitude(pruning_rate);
        let current_sparsity = network.sparsity();

        // Rewind
        network.restore_weights(&initial_weights);

        // Evaluate winning ticket at this sparsity
        let mut eval_net = network.clone();
        eval_net.train(train_features, train_labels, epochs_per_round, batch_size);
        let winning_acc = eval_net.accuracy(test_features, test_labels);

        // Evaluate random ticket at same sparsity
        let random_mask =
            generate_random_mask(network.total_params(), current_sparsity, &mut rng);
        let mut rand_net = LotteryTicketNetwork::new(layer_sizes, learning_rate, &mut rng);
        rand_net.restore_weights(&initial_weights);
        rand_net.set_mask_flat(&random_mask);
        rand_net.train(train_features, train_labels, epochs_per_round, batch_size);
        let random_acc = rand_net.accuracy(test_features, test_labels);

        results.push(SparsityAccuracyPoint {
            sparsity: current_sparsity,
            winning_accuracy: winning_acc,
            random_accuracy: random_acc,
        });
    }

    results
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_network_forward_shape() {
        let mut rng = make_rng();
        let net = LotteryTicketNetwork::new(&[4, 8, 1], 0.01, &mut rng);
        let input = Array2::zeros((10, 4));
        let output = net.forward(&input);
        assert_eq!(output.shape(), &[10, 1]);
    }

    #[test]
    fn test_save_restore_weights() {
        let mut rng = make_rng();
        let mut net = LotteryTicketNetwork::new(&[4, 8, 1], 0.01, &mut rng);
        let saved = net.save_weights();

        // Modify weights
        net.layers[0].weights.fill(999.0);
        assert!((net.layers[0].weights[[0, 0]] - 999.0).abs() < 1e-10);

        // Restore
        net.restore_weights(&saved);
        assert!((net.layers[0].weights[[0, 0]] - 999.0).abs() > 1e-5);
    }

    #[test]
    fn test_pruning_increases_sparsity() {
        let mut rng = make_rng();
        let mut net = LotteryTicketNetwork::new(&[4, 16, 8, 1], 0.01, &mut rng);
        assert!((net.sparsity() - 0.0).abs() < 1e-10);

        net.prune_by_magnitude(0.3);
        assert!(net.sparsity() > 0.2);
        assert!(net.sparsity() < 0.4);
    }

    #[test]
    fn test_mask_preserved_after_restore() {
        let mut rng = make_rng();
        let mut net = LotteryTicketNetwork::new(&[4, 8, 1], 0.01, &mut rng);
        let saved = net.save_weights();

        // Prune
        net.prune_by_magnitude(0.5);
        let sparsity_before = net.sparsity();

        // Restore weights
        net.restore_weights(&saved);
        let sparsity_after = net.sparsity();

        // Mask should be preserved
        assert!((sparsity_before - sparsity_after).abs() < 1e-10);
    }

    #[test]
    fn test_random_mask_sparsity() {
        let mut rng = make_rng();
        let mask = generate_random_mask(1000, 0.7, &mut rng);
        let active: usize = mask.iter().filter(|&&v| v > 0.5).count();
        // Should have ~300 active params (30% of 1000)
        assert!((active as f64 - 300.0).abs() < 50.0);
    }

    #[test]
    fn test_training_reduces_loss() {
        let mut rng = make_rng();
        let mut net = LotteryTicketNetwork::new(&[2, 8, 1], 0.1, &mut rng);

        // Simple XOR-like data
        let features = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let losses = net.train(&features, &labels, 100, 4);
        assert!(losses.last().unwrap() < losses.first().unwrap());
    }

    #[test]
    fn test_imp_produces_sparse_network() {
        let mut rng = make_rng();
        let mut net = LotteryTicketNetwork::new(&[2, 16, 1], 0.1, &mut rng);
        let initial = net.save_weights();

        let features = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let result = iterative_magnitude_pruning(
            &mut net, &initial, &features, &labels, 0.2, 5, 50, 4,
        );

        assert!(result.sparsity > 0.5);
        assert_eq!(result.rounds_completed, 5);
    }

    #[test]
    fn test_set_get_mask_roundtrip() {
        let mut rng = make_rng();
        let mut net = LotteryTicketNetwork::new(&[4, 8, 1], 0.01, &mut rng);
        let original_mask = net.get_mask_flat();
        net.set_mask_flat(&original_mask);
        let recovered_mask = net.get_mask_flat();
        assert_eq!(original_mask, recovered_mask);
    }

    #[test]
    fn test_accuracy_in_valid_range() {
        let mut rng = make_rng();
        let net = LotteryTicketNetwork::new(&[2, 4, 1], 0.01, &mut rng);
        let features = Array2::ones((10, 2));
        let labels = Array1::ones(10);
        let acc = net.accuracy(&features, &labels);
        assert!(acc >= 0.0 && acc <= 1.0);
    }
}
