use lottery_ticket_trading::*;
use rand::SeedableRng;

fn main() -> anyhow::Result<()> {
    println!("=== Chapter 205: Lottery Ticket Trading ===\n");

    // -------------------------------------------------------
    // 1. Fetch market data from Bybit
    // -------------------------------------------------------
    println!("[1/4] Fetching BTCUSDT kline data from Bybit...");
    let candles = fetch_bybit_klines("BTCUSDT", "60", 200)?;
    println!("  Fetched {} candles", candles.len());
    if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
        println!(
            "  Time range: {} -> {}",
            first.timestamp, last.timestamp
        );
        println!(
            "  Price range: {:.2} -> {:.2}",
            first.close, last.close
        );
    }

    // -------------------------------------------------------
    // 2. Prepare features and train/test split
    // -------------------------------------------------------
    println!("\n[2/4] Preparing features...");
    let lookback = 10;
    let (features, labels) = prepare_features(&candles, lookback);
    let n = features.nrows();
    let train_size = (n as f64 * 0.7) as usize;

    let train_features = features.slice(ndarray::s![..train_size, ..]).to_owned();
    let train_labels = labels.slice(ndarray::s![..train_size]).to_owned();
    let test_features = features.slice(ndarray::s![train_size.., ..]).to_owned();
    let test_labels = labels.slice(ndarray::s![train_size..]).to_owned();

    println!("  Total samples:    {}", n);
    println!("  Training samples: {}", train_size);
    println!("  Test samples:     {}", n - train_size);
    println!("  Features per sample: {}", features.ncols());

    let positive_rate = labels.mean().unwrap_or(0.5);
    println!("  Positive label rate: {:.1}%", positive_rate * 100.0);

    // -------------------------------------------------------
    // 3. Run IMP at various sparsity levels
    // -------------------------------------------------------
    println!("\n[3/4] Running Iterative Magnitude Pruning...");
    let layer_sizes = &[4, 32, 16, 1];
    let learning_rate = 0.01;
    let epochs_per_round = 50;
    let batch_size = 32;
    let pruning_rate = 0.2;
    let num_rounds = 8;
    let seed = 42u64;

    let sa_results = sparsity_vs_accuracy(
        layer_sizes,
        learning_rate,
        &train_features,
        &train_labels,
        &test_features,
        &test_labels,
        pruning_rate,
        num_rounds,
        epochs_per_round,
        batch_size,
        seed,
    );

    println!("\n  Sparsity vs Accuracy (Winning vs Random tickets):");
    println!("  {:<12} {:<18} {:<18} {:<10}", "Sparsity", "Winning Acc", "Random Acc", "Delta");
    println!("  {}", "-".repeat(58));
    for point in &sa_results {
        let delta = point.winning_accuracy - point.random_accuracy;
        println!(
            "  {:<12.1}% {:<18.2}% {:<18.2}% {:<+10.2}pp",
            point.sparsity * 100.0,
            point.winning_accuracy * 100.0,
            point.random_accuracy * 100.0,
            delta * 100.0,
        );
    }

    // -------------------------------------------------------
    // 4. Full comparison: Dense vs Winning Ticket vs Random
    // -------------------------------------------------------
    println!("\n[4/4] Full ticket comparison (5 rounds of 20% pruning)...");
    let comparison = compare_tickets(
        layer_sizes,
        learning_rate,
        &train_features,
        &train_labels,
        &test_features,
        &test_labels,
        pruning_rate,
        5, // fewer rounds for the comparison
        epochs_per_round,
        batch_size,
        seed,
    );

    println!("\n{}", comparison);

    // Summary
    println!("--- Summary ---");
    println!(
        "The winning lottery ticket achieves {:.1}% accuracy with only {:.1}% of parameters,",
        comparison.winning_ticket_accuracy * 100.0,
        (1.0 - comparison.sparsity) * 100.0
    );
    println!(
        "compared to {:.1}% for the dense model and {:.1}% for a random sparse model.",
        comparison.dense_accuracy * 100.0,
        comparison.random_ticket_accuracy * 100.0,
    );
    println!(
        "\nThis demonstrates that structured pruning via IMP identifies genuinely"
    );
    println!(
        "important subnetworks, not just any sparse configuration."
    );

    Ok(())
}
