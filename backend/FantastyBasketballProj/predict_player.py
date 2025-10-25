"""
Prediction Script for Fantasy Basketball CNN
Predict future performance for any player
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from fantasy_cnn import FantasyFeatureEngine
import os
import sys


def predict_player(player_name, df, model, feature_engine):
    """
    Predict fantasy performance for a specific player
    """
    # Get player data
    player_df = df[df['player_name'] == player_name].sort_values('game_date')

    if len(player_df) == 0:
        print(f"❌ ERROR: Player '{player_name}' not found in data")
        print(f"Available players: {', '.join(df['player_name'].unique()[:10])}...")
        return None

    # Get last 15 games
    last_15 = player_df.tail(15)

    if len(last_15) < 15:
        print(f"❌ ERROR: Not enough data for {player_name}")
        print(f"   Need 15 games, but only have {len(last_15)}")
        return None

    # Prepare input
    feature_vectors = []
    for _, game in last_15.iterrows():
        game_dict = game.to_dict()
        features = feature_engine.engineer_all_features(game_dict)
        feature_vectors.append(list(features.values()))

    X_input = np.array([feature_vectors])  # Shape: (1, 15, n_features)

    # Make prediction
    predictions = model.predict(X_input, verbose=0)
    high_end = predictions[0][0][0]
    low_end = predictions[1][0][0]
    expected_avg = predictions[2][0][0]
    confidence = predictions[3][0][0]

    # Get current stats
    current_avg = player_df['season_avg_fp'].iloc[-1]
    last_5_avg = player_df['last_5_avg_fp'].iloc[-1]

    # Calculate trends
    trend = "📈 Trending UP" if last_5_avg > current_avg else "📉 Trending DOWN"
    trend_diff = last_5_avg - current_avg

    return {
        'player_name': player_name,
        'current_avg': current_avg,
        'last_5_avg': last_5_avg,
        'trend': trend,
        'trend_diff': trend_diff,
        'expected_avg': expected_avg,
        'high_end': high_end,
        'low_end': low_end,
        'confidence': confidence,
        'upside': high_end - expected_avg,
        'downside': expected_avg - low_end
    }


def display_prediction(result):
    """
    Display prediction results in a nice format
    """
    print(f"\n{'=' * 70}")
    print(f"🏀 FANTASY PROJECTION: {result['player_name']}")
    print(f"{'=' * 70}")
    print(f"\n📊 CURRENT PERFORMANCE:")
    print(f"   Season Average:     {result['current_avg']:6.1f} FP")
    print(f"   Last 5 Games Avg:   {result['last_5_avg']:6.1f} FP  {result['trend']}")
    print(f"   Trend Difference:   {result['trend_diff']:+6.1f} FP")

    print(f"\n🔮 MODEL PREDICTIONS:")
    print(f"   Expected Average:   {result['expected_avg']:6.1f} FP")
    print(f"   High End (Ceiling): {result['high_end']:6.1f} FP  (↑ {result['upside']:+.1f})")
    print(f"   Low End (Floor):    {result['low_end']:6.1f} FP  (↓ {result['downside']:-.1f})")
    print(f"   Confidence:         {result['confidence']:6.1%}")

    # Value assessment
    print(f"\n💡 ASSESSMENT:")
    if result['upside'] > 10:
        print(f"   🚀 HIGH UPSIDE - Ceiling is {result['upside']:.1f} points above expected")
    elif result['upside'] > 5:
        print(f"   ✅ SOLID UPSIDE - Good ceiling potential")
    else:
        print(f"   📊 LIMITED UPSIDE - Ceiling close to expected value")

    if result['downside'] > 10:
        print(f"   ⚠️  HIGH RISK - Floor is {result['downside']:.1f} points below expected")
    elif result['downside'] > 5:
        print(f"   🔸 MODERATE RISK - Some downside exposure")
    else:
        print(f"   ✅ SAFE FLOOR - Limited downside risk")

    if result['confidence'] > 0.75:
        print(f"   💪 HIGH CONFIDENCE - Model is {result['confidence']:.0%} confident")
    elif result['confidence'] > 0.5:
        print(f"   👌 MODERATE CONFIDENCE")
    else:
        print(f"   ⚠️  LOW CONFIDENCE - Prediction is uncertain")

    print(f"{'=' * 70}\n")


def main():
    print("=" * 70)
    print("FANTASY BASKETBALL CNN - PLAYER PREDICTIONS")
    print("=" * 70)

    # ====================================
    # 1. LOAD MODEL
    # ====================================
    print("\n[1/3] Loading trained model...")

    model_path = 'models/saved_model'
    if not os.path.exists(model_path):
        print("❌ ERROR: No trained model found!")
        print("Please run train_model.py first to train the model")
        return

    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return

    # ====================================
    # 2. LOAD DATA
    # ====================================
    print("\n[2/3] Loading player data...")

    data_path = 'data/player_games_2024.csv'
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Cannot find {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"✅ Loaded data for {df['player_id'].nunique()} players")

    feature_engine = FantasyFeatureEngine()

    # ====================================
    # 3. MAKE PREDICTIONS
    # ====================================
    print("\n[3/3] Making predictions...\n")

    # You can modify this list to predict multiple players
    players_to_predict = [
        "LaMelo Ball",
        "Nikola Jokic",
        "Cade Cunningham",
        "Franz Wagner"
    ]

    # Or get from command line argument
    if len(sys.argv) > 1:
        players_to_predict = sys.argv[1:]

    results = []
    for player_name in players_to_predict:
        result = predict_player(player_name, df, model, feature_engine)
        if result:
            display_prediction(result)
            results.append(result)

    # ====================================
    # 4. SUMMARY TABLE
    # ====================================
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY - TOP PLAYERS BY EXPECTED VALUE")
        print("=" * 70)
        results.sort(key=lambda x: x['expected_avg'], reverse=True)

        print(f"\n{'Player':<20} {'Current':<10} {'Expected':<10} {'Ceiling':<10} {'Floor':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['player_name']:<20} {