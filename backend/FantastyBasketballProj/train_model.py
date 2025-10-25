"""
Training Script for Fantasy Basketball CNN
Place this file in your project root directory
"""

import pandas as pd
import numpy as np
from fantasy_cnn import FantasyBasketballCNN, FantasyFeatureEngine
from sklearn.model_selection import train_test_split
import os


def main():
    print("=" * 60)
    print("FANTASY BASKETBALL CNN - TRAINING")
    print("=" * 60)

    # ====================================
    # 1. LOAD YOUR DATA
    # ====================================
    print("\n[1/6] Loading data...")

    # Check if data file exists
    data_path = 'data/player_games_2024.csv'
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Cannot find {data_path}")
        print("Please place your CSV file in the data/ folder")
        return

    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} game records")
    print(f"✅ Found {df['player_id'].nunique()} unique players")

    # Show sample of data
    print("\nFirst few rows of data:")
    print(df.head(3))

    # ====================================
    # 2. DATA VALIDATION
    # ====================================
    print("\n[2/6] Validating data...")

    required_columns = [
        'player_id', 'player_name', 'game_date', 'age', 'fantasy_points',
        'season_avg_fp', 'last_5_avg_fp'
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ ERROR: Missing required columns: {missing_cols}")
        return

    print("✅ All required columns present")

    # Fill missing values with defaults
    defaults = {
        'usage_rate': 0.20,
        'true_shooting_pct': 0.55,
        'per': 15.0,
        'mpg': 25.0,
        'fp_std_dev_15': 5.0,
        'games_missed_last_season': 0,
        'injury_status': 'healthy',
        'team_pace': 100.0,
        'is_starter': 1,
        'games_remaining': 82
    }
    df = df.fillna(defaults)
    print("✅ Filled missing values with defaults")

    # ====================================
    # 3. PREPARE DATA FOR MODEL
    # ====================================
    print("\n[3/6] Preparing sequences for training...")

    feature_engine = FantasyFeatureEngine()

    X_data = []
    y_high = []
    y_low = []
    y_expected = []
    y_confidence = []

    players_processed = 0
    sequences_created = 0

    for player_id in df['player_id'].unique():
        player_df = df[df['player_id'] == player_id].sort_values('game_date')

        # Need at least 16 games (15 for sequence + 1 for target)
        if len(player_df) < 16:
            continue

        players_processed += 1

        # Create sliding windows
        for i in range(len(player_df) - 15):
            sequence = player_df.iloc[i:i + 15]

            # Extract features for each game in sequence
            feature_vectors = []
            for _, game in sequence.iterrows():
                game_dict = game.to_dict()
                features = feature_engine.engineer_all_features(game_dict)
                feature_vectors.append(list(features.values()))

            X_data.append(feature_vectors)

            # Target: predict next game
            next_game = player_df.iloc[i + 15]
            next_game_dict = next_game.to_dict()
            features = feature_engine.engineer_all_features(next_game_dict)
            high, low, expected = feature_engine.calculate_high_low_projections(features)

            y_high.append(high)
            y_low.append(low)
            y_expected.append(expected)
            y_confidence.append(1.0)

            sequences_created += 1

        # Progress update every 50 players
        if players_processed % 50 == 0:
            print(f"  Processed {players_processed} players, created {sequences_created} sequences...")

    # Convert to numpy arrays
    X = np.array(X_data)
    y = {
        'high_end': np.array(y_high),
        'low_end': np.array(y_low),
        'expected_avg': np.array(y_expected),
        'confidence': np.array(y_confidence)
    }

    print(f"✅ Processed {players_processed} players")
    print(f"✅ Created {sequences_created} training samples")
    print(f"✅ Input shape: {X.shape}")

    if len(X) < 100:
        print("⚠️  WARNING: Very few training samples. Model may not train well.")
        print("   Recommendation: Get more data (multiple seasons)")

    # ====================================
    # 4. SPLIT DATA
    # ====================================
    print("\n[4/6] Splitting into train/validation sets...")

    # Simple split by index (keeps time series order mostly intact)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_val = X[split_idx:]

    y_train = {k: v[:split_idx] for k, v in y.items()}
    y_val = {k: v[split_idx:] for k, v in y.items()}

    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Validation samples: {len(X_val)}")

    # ====================================
    # 5. BUILD AND TRAIN MODEL
    # ====================================
    print("\n[5/6] Building model architecture...")

    n_features = X.shape[2]
    model = FantasyBasketballCNN(n_features=n_features, sequence_length=15)
    model.build_model()

    print(f"✅ Model built with {n_features} features")
    print("\nModel architecture:")
    model.model.summary()

    print("\n[6/6] Training model...")
    print("This may take 10-30 minutes depending on data size...")
    print("Press Ctrl+C to stop early (model will save best weights)\n")

    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )

    # ====================================
    # 6. SAVE MODEL
    # ====================================
    print("\nSaving model...")

    os.makedirs('models', exist_ok=True)
    model.model.save('models/saved_model')

    print("✅ Model saved to models/saved_model/")

    # ====================================
    # 7. TRAINING SUMMARY
    # ====================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total samples trained: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Model location: models/saved_model/")
    print("\nNext steps:")
    print("1. Run predict_player.py to make predictions")
    print("2. Run analyze_trade.py to analyze trades")
    print("=" * 60)


if __name__ == "__main__":
    main()